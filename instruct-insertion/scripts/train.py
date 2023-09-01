#!/usr/bin/env python
# coding: utf-8

import json
import os.path as osp
import sys
import time
from pprint import pformat

import torch
import torch.nn as nn
import tqdm
from termcolor import colored
from torch import optim

sys.path.append(f"{osp.dirname(__file__)}/..")


#########################
#                       #
#    Import datasets    #
#                       #
#########################
# isort: split
from data.referit3d.datasets import make_data_loaders
from data.referit3d.in_out.neural_net_oriented import (
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
    trim_scans_per_referit3d_data_,
)

##################
#                #
#    Training    #
#                #
##################
# isort: split
import evaluate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from metrics import LOCAL_METRIC_PATHS
from torch import multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from train_utils import single_epoch_train

##################################
#                                #
#    Visual grounding related    #
#                                #
##################################
# isort: split
# from model.referit3d.analysis.deepnet_predictions import analyze_predictions
from model.referit3d_model.referit3d_net import ReferIt3DNet_transformer
from model.referit3d_model.utils import load_state_dicts, save_state_dicts
from transformers import BatchEncoding, BertModel, BertTokenizer

# from model.referit3d_model.utils.tf_visualizer import Visualizer

###########################
#                         #
#    Diffusion related    #
#                         #
###########################
# isort: split

from model.point_e_model.configs.config import load_config, make_sample_density
from model.point_e_model.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from model.point_e_model.diffusion.sampler import PointCloudSampler
from model.point_e_model.evals.metrics import *
from model.point_e_model.models.configs import MODEL_CONFIGS, model_from_config
from model.point_e_model.models.download import load_checkpoint
from model.point_e_model.util import n_params
from model.point_e_model.util.common import get_linear_scheduler
from model.point_e_model.util.plotting import plot_point_cloud
from model.point_e_model.util.point_cloud import PointCloud

#######################
#                     #
#    Miscellaneous    #
#                     #
#######################
# isort: split
from utils import seed_everything
from utils.arguments import parse_arguments
from utils.logger import get_logger, init_logger


def main():
    # Parse arguments
    args = parse_arguments()

    # TODO: add log file
    init_logger()
    logger = get_logger(__name__)
    acc_config = ProjectConfiguration(
        project_dir=osp.join(args.project_top_dir, args.project_name),
        logging_dir=osp.join(args.project_top_dir, "logs"),
    )
    accelerator = Accelerator(log_with="tensorboard", project_config=acc_config)
    device = accelerator.device
    seed_everything(args.random_seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)

    #######################
    #                     #
    #    Data pre-load    #
    #                     #
    #######################
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data_(referit_data, all_scans_in_dict)
    mean_rgb = compute_auxiliary_data(referit_data, all_scans_in_dict)
    data_loaders = make_data_loaders(args, referit_data, class_to_idx, all_scans_in_dict, mean_rgb)

    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx["pad"]
    # Object-type classification
    class_name_list = list(class_to_idx.keys())

    class_name_tokens: BatchEncoding = tokenizer(class_name_list, return_tensors="pt", padding=True)
    class_name_tokens = class_name_tokens.to(device)

    if args.mvt_model == "referIt3DNet_transformer":
        mvt3dvg = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    else:
        assert False
    mvt3dvg = mvt3dvg.to(device)
    print(mvt3dvg)

    # model params
    param_list = [
        {"params": mvt3dvg.language_encoder.parameters(), "lr": args.init_lr * 0.1},
        {"params": mvt3dvg.refer_encoder.parameters(), "lr": args.init_lr * 0.1},
        {"params": mvt3dvg.obj_encoder.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.obj_feature_mapping.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.box_feature_mapping.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.language_clf.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.object_language_clf.parameters(), "lr": args.init_lr},
    ]
    if not args.label_lang_sup:
        param_list.append({"params": mvt3dvg.obj_clf.parameters(), "lr": args.init_lr})

    # construct model
    point_e = model_from_config(MODEL_CONFIGS[args.point_e_model], device)
    point_e.to(device)

    # construct diffusion
    point_e_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[args.point_e_model])

    param_list.append({"params": point_e.parameters(), "lr": args.init_lr})
    optimizer = optim.AdamW(param_list, lr=args.init_lr)  # init_lr = 1e-4

    lr_scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=0,
        end_epoch=args.max_train_epochs,
        start_lr=args.init_lr,
        end_lr=args.min_lr,
    )

    # adapt with `accelerate`
    (
        mvt3dvg,
        point_e,
        data_loaders["train"],
        data_loaders["test"],
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        mvt3dvg, point_e, data_loaders["train"], data_loaders["test"], optimizer, lr_scheduler
    )

    aux_channels = ["R", "G", "B"]
    sampler = PointCloudSampler(
        device=device,
        models=[point_e],
        diffusions=[point_e_diffusion],
        num_points=[args.points_per_object],
        aux_channels=aux_channels,
        guidance_scale=[3.0],
        use_karras=[True],
        karras_steps=[64],
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
        model_kwargs_key_filter=[args.cond],
    )

    # Resume training
    if args.resume_path:
        accelerator.load_state(args.resume_path)

    start_training_epoch = lr_scheduler.last_epoch + 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    evaluate.load

    # Training.
    if args.mode == "train":
        logger.info("Starting the training. Good luck!", main_process_only=True)
        with accelerator.main_process_first():
            # load metrics
            metrics = {
                "referit3d_loc_acc": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                ),
                "referit3d_cls_acc": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                ),
                "referit3d_txt_acc": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                ),
            }

        with tqdm.tqdm(
            range(start_training_epoch, args.max_train_epochs + 1),
            desc="epochs",
            disable=not accelerator.is_main_process,
        ) as bar:
            timings = dict()
            for epoch in bar:
                logger.info(f"Current LR: {lr_scheduler.get_last_lr()}", main_process_only=True)
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(
                    accelerator=accelerator,
                    MVT3DVG=mvt3dvg,
                    point_e=point_e,
                    sampler=sampler,
                    data_loader=data_loaders["train"],
                    optimizer=optimizer,
                    device=device,
                    pad_idx=pad_idx,
                    args=args,
                    metrics=metrics,
                    epoch=epoch,
                )
                toc = time.time()
                timings["train"] = (toc - tic) / 60

                accelerator.log(train_meters)

                # TODO - Fix Evaluate
                # Evaluate:
                # tic = time.time()
                # test_meters = evaluate_on_dataset(
                #     mvt3dvg,
                #     data_loaders["test"],
                #     criteria,
                #     device,
                #     pad_idx,
                #     args=args,
                #     tokenizer=tokenizer,
                # )
                # toc = time.time()
                # timings["test"] = (toc - tic) / 60

                # eval_acc = test_meters["test_referential_acc"]

                # last_test_acc = eval_acc
                # last_test_epoch = epoch

                lr_scheduler.step()

                # TODO - Also save the best model
                accelerator.save_state(
                    osp.join(args.project_top_dir, args.project_name, "checkpoints", "last_model")
                )

                # TODO - Fix Log
                # if best_test_acc < eval_acc:
                #     logger.info(colored("Test accuracy, improved @epoch {}".format(epoch), "green"))
                #     best_test_acc = eval_acc
                #     best_test_epoch = epoch

                #     save_state_dicts(
                #         osp.join(args.checkpoint_dir, "best_model.pth"),
                #         epoch,
                #         model=mvt3dvg,
                #         optimizer=optimizer,
                #         lr_scheduler=lr_scheduler,
                #     )
                # else:
                #     logger.info(
                #         colored("Test accuracy, did not improve @epoch {}".format(epoch), "red")
                #     )

                # log_train_test_information()
                # train_meters.update(test_meters)
                bar.refresh()

        accelerator.end_training()
        logger.info("Finished training successfully.", main_process_only=True)

    elif args.mode == "test":
        # TODO: change evaluation metrics
        raise NotImplementedError
        meters = evaluate_on_dataset(
            mvt3dvg, data_loaders["test"], criteria, device, pad_idx, args=args, tokenizer=tokenizer
        )
        print("Reference-Accuracy: {:.4f}".format(meters["test_referential_acc"]))
        print("Object-Clf-Accuracy: {:.4f}".format(meters["test_object_cls_acc"]))
        print("Text-Clf-Accuracy {:.4f}:".format(meters["test_txt_cls_acc"]))

        out_file = osp.join(args.checkpoint_dir, "test_result.txt")


if __name__ == "__main__":
    main()
