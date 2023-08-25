#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import sys
import time
import warnings

import torch
import torch.nn as nn
import tqdm

sys.path.append(f"{osp.dirname(__file__)}/..")

from model.referit3d.analysis.deepnet_predictions import analyze_predictions
from model.referit3d.in_out.arguments import parse_arguments
from model.referit3d.in_out.neural_net_oriented import (
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
    trim_scans_per_referit3d_data,
)
from model.referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from model.referit3d.models.referit3d_net import ReferIt3DNet_transformer
from model.referit3d.models.referit3d_net_utils import evaluate_on_dataset, single_epoch_train
from model.referit3d.models.utils import load_state_dicts, save_state_dicts
from model.referit3d.utils import create_logger, seed_training_code, set_gpu_to_zero_position
from model.referit3d.utils.tf_visualizer import Visualizer
from termcolor import colored
from torch import optim
from transformers import BertModel, BertTokenizer

"""Trains Karras et al. (2022) diffusion models."""

import argparse
import json
import os
import random

import numpy as np
import torch
from ipdb import set_trace
from model.point_e.configs.config import load_config, make_sample_density
from model.point_e.dataset.shapenet import ShapeNetCore
from model.point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from model.point_e.diffusion.sampler import PointCloudSampler
from model.point_e.evals.metrics import *
from model.point_e.models.configs import MODEL_CONFIGS, model_from_config
from model.point_e.models.download import load_checkpoint
from model.point_e.models.util import n_params
from model.point_e.util.common import get_linear_scheduler
from model.point_e.util.plotting import plot_point_cloud
from model.point_e.util.point_cloud import PointCloud
from torch import multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
from up.utils.env.dist_helper import env, finalize, setup_distributed
from up.utils.general.log_helper import default_logger as logger


def log_train_test_information():
    """Helper logging function.
    Note uses "global" variables defined below.
    """
    logger.info("Epoch:{}".format(epoch))
    for phase in ["train", "test"]:
        if phase == "train":
            meters = train_meters
        else:
            meters = test_meters

        info = "{}: Total-Loss {:.4f}, Listening-Acc {:.4f}".format(
            phase, meters[phase + "_total_loss"], meters[phase + "_referential_acc"]
        )

        if args.obj_cls_alpha > 0:
            info += ", Object-Clf-Acc: {:.4f}".format(meters[phase + "_object_cls_acc"])

        if args.lang_cls_alpha > 0:
            info += ", Text-Clf-Acc: {:.4f}".format(meters[phase + "_txt_cls_acc"])

        logger.info(info)
        logger.info("{}: Epoch-time {:.3f}".format(phase, timings[phase]))
    logger.info("Best so far {:.3f} (@epoch {})".format(best_test_acc, best_test_epoch))


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(
        args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb
    )
    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"

    device = torch.device("cuda")
    seed_training_code(args.random_seed)

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx["pad"]
    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = len(args.gpu.strip(",").split(","))

    if args.model == "referIt3DNet_transformer":
        mvt3dvg = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    else:
        assert False
    if gpu_num > 1:
        mvt3dvg = nn.DataParallel(mvt3dvg)
    mvt3dvg = mvt3dvg.to(device)
    print(mvt3dvg)

    # <1>
    if gpu_num > 1:
        param_list = [
            {"params": mvt3dvg.module.language_encoder.parameters(), "lr": args.init_lr * 0.1},
            {"params": mvt3dvg.module.refer_encoder.parameters(), "lr": args.init_lr * 0.1},
            {"params": mvt3dvg.module.object_encoder.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.module.obj_feature_mapping.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.module.box_feature_mapping.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.module.language_clf.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.module.object_language_clf.parameters(), "lr": args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append({"params": mvt3dvg.module.obj_clf.parameters(), "lr": args.init_lr})
    else:
        param_list = [
            {"params": mvt3dvg.language_encoder.parameters(), "lr": args.init_lr * 0.1},
            {"params": mvt3dvg.refer_encoder.parameters(), "lr": args.init_lr * 0.1},
            {"params": mvt3dvg.object_encoder.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.obj_feature_mapping.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.box_feature_mapping.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.language_clf.parameters(), "lr": args.init_lr},
            {"params": mvt3dvg.object_language_clf.parameters(), "lr": args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append({"params": mvt3dvg.obj_clf.parameters(), "lr": args.init_lr})

    # FIXME - Should we combine two model into one?
    config = load_config(open(args.config))

    # construct model
    point_e = model_from_config(MODEL_CONFIGS[name], device)
    point_e.to(device)

    # for multicard training
    point_e = DDP(point_e, device_ids=[env.local_rank], broadcast_buffers=False)

    # construct diffusion
    point_e_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[name])

    # construct optimizer for point_e
    # FIXME - we should have only one optimizer, so parameters should be reassigned
    # if opt_config["type"] == "adamw":
    #         opt = optim.AdamW(
    #         point_e.parameters(),
    #         lr=opt_config["lr"], # lr = 1e-4
    #         betas=tuple(opt_config["betas"]), # [0.95, 0.999]
    #         eps=opt_config["eps"], 1e-6
    #         weight_decay=opt_config["weight_decay"], # 1e-3
    #     )
    # elif opt_config["type"] == "sgd":
    #     opt = optim.SGD(
    #         point_e.parameters(),
    #         lr=opt_config["lr"],
    #         momentum=opt_config.get("momentum", 0.0),
    #         nesterov=opt_config.get("nesterov", False),
    #         weight_decay=opt_config.get("weight_decay", 0.0),
    #     )
    # else:
    #     raise ValueError("Invalid optimizer type")

    optimizer = optim.Adam(param_list, lr=args.init_lr)  # init_lr = 1e-4
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [40, 50, 60, 70, 80, 90], gamma=0.65
    )

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn("Resuming assumes that the BEST per-val model is loaded!")
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=mvt3dvg)
        print("Loaded a model stopped at epoch: {}.".format(loaded_epoch))
        if not args.fine_tune:
            print("Loaded a model that we do NOT plan to fine-tune.")
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            start_training_epoch = 0
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print(
                "Loaded model had {} test-accuracy in the corresponding dataset used when trained.".format(
                    best_test_acc
                )
            )
        else:
            print("Parameters that do not allow gradients to be back-propped:")
            ft_everything = True
            for name, param in mvt3dvg.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print("None, all wil be fine-tuned")
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = args.max_train_epochs + 1 - start_training_epoch
            print("Ready to *fine-tune* the model for a max of {} epochs".format(dummy))

    # Training.
    if args.mode == "train":
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info("Starting the training. Good luck!")

        with tqdm.trange(start_training_epoch, args.max_train_epochs + 1, desc="epochs") as bar:
            timings = dict()
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr())
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(
                    mvt3dvg,
                    point_e,
                    config,
                    data_loaders["train"],
                    criteria,
                    optimizer,
                    device,
                    pad_idx,
                    args=args,
                    tokenizer=tokenizer,
                    epoch=epoch,
                )
                toc = time.time()
                timings["train"] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(
                    mvt3dvg,
                    data_loaders["test"],
                    criteria,
                    device,
                    pad_idx,
                    args=args,
                    tokenizer=tokenizer,
                )
                toc = time.time()
                timings["test"] = (toc - tic) / 60

                eval_acc = test_meters["test_referential_acc"]

                last_test_acc = eval_acc
                last_test_epoch = epoch

                lr_scheduler.step()

                save_state_dicts(
                    osp.join(args.checkpoint_dir, "last_model.pth"),
                    epoch,
                    model=mvt3dvg,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )

                if best_test_acc < eval_acc:
                    logger.info(colored("Test accuracy, improved @epoch {}".format(epoch), "green"))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    save_state_dicts(
                        osp.join(args.checkpoint_dir, "best_model.pth"),
                        epoch,
                        model=mvt3dvg,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                    )
                else:
                    logger.info(
                        colored("Test accuracy, did not improve @epoch {}".format(epoch), "red")
                    )

                log_train_test_information()
                train_meters.update(test_meters)
                train_vis.log_scalars(
                    {k: v for k, v in train_meters.items() if "_acc" in k},
                    step=epoch,
                    main_tag="acc",
                )
                train_vis.log_scalars(
                    {k: v for k, v in train_meters.items() if "_loss" in k},
                    step=epoch,
                    main_tag="loss",
                )

                bar.refresh()

        with open(osp.join(args.checkpoint_dir, "final_result.txt"), "w") as f_out:
            f_out.write(
                ("Best accuracy: {:.4f} (@epoch {})".format(best_test_acc, best_test_epoch))
            )
            f_out.write(
                ("Last accuracy: {:.4f} (@epoch {})".format(last_test_acc, last_test_epoch))
            )

        logger.info("Finished training successfully.")

    elif args.mode == "evaluate":
        meters = evaluate_on_dataset(
            mvt3dvg, data_loaders["test"], criteria, device, pad_idx, args=args, tokenizer=tokenizer
        )
        print("Reference-Accuracy: {:.4f}".format(meters["test_referential_acc"]))
        print("Object-Clf-Accuracy: {:.4f}".format(meters["test_object_cls_acc"]))
        print("Text-Clf-Accuracy {:.4f}:".format(meters["test_txt_cls_acc"]))

        out_file = osp.join(args.checkpoint_dir, "test_result.txt")
        res = analyze_predictions(
            mvt3dvg,
            data_loaders["test"].dataset,
            class_to_idx,
            pad_idx,
            device,
            args,
            out_file=out_file,
            tokenizer=tokenizer,
        )
        print(res)
