#!/usr/bin/env python
# coding: utf-8

import json
import logging
import operator
import os.path as osp
import pprint
import shutil
import sys
from datetime import datetime

from termcolor import colored
from torch import optim
from tqdm import tqdm

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
import torch
import torch._dynamo
import train_utils
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from metrics import LOCAL_METRIC_PATHS
from torch import optim

##################################
#                                #
#    Visual grounding related    #
#                                #
##################################
# isort: split
from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer
from transformers import BatchEncoding, BertTokenizer

###########################
#                         #
#    Diffusion related    #
#                         #
###########################
# isort: split

from models.point_e_model.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e_model.diffusion.sampler import PointCloudSampler
from models.point_e_model.models.configs import MODEL_CONFIGS, model_from_config
from models.point_e_model.models.download import load_checkpoint
from models.point_e_model.util.common import get_linear_scheduler

#######################
#                     #
#    Miscellaneous    #
#                     #
#######################
# isort: split
import torchinfo
from utils.arguments import parse_arguments
from utils.logger import get_logger, init_logger

DATE_FMT = "%Y-%m-%d_%H-%M-%S"


def main():
    # Parse arguments
    args = parse_arguments()
    project_time = datetime.now()
    project_time_str = project_time.strftime(DATE_FMT)
    acc_config = ProjectConfiguration(
        project_dir=osp.join(args.project_top_dir, args.project_name),
        logging_dir=osp.join(args.project_top_dir, "tf_logs"),
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=acc_config,
        kwargs_handlers=[ddp_kwargs],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    # tracker setup
    accelerator.init_trackers(f"{args.project_name}_{project_time_str}", config=vars(args))
    # torchdynamo setting
    torch._dynamo.config.log_level = logging.ERROR
    # save log file only to main process
    init_logger(
        accelerator,
        log_file=osp.join(
            args.project_top_dir,
            args.project_name,
            "logs",
            project_time_str + ".log",
        ),
    )
    logger = get_logger(__name__)
    logger.info(
        f"Project {args.project_name} start at {project_time_str}.",
        main_process_only=True,
    )
    logger.info(f"Args: {vars(args)}", main_process_only=True)
    if accelerator.is_main_process and accelerator.num_processes > 1:
        logger.info(
            f"Distributed training with {accelerator.num_processes} processes.",
            main_process_only=True,
        )
    if args.gradient_accumulation_steps > 1:
        logger.info(
            f"Gradient accumulation steps: {args.gradient_accumulation_steps}.",
            main_process_only=True,
        )
    device = accelerator.device
    set_seed(args.random_seed)
    logger.info(f"Random seed: {args.random_seed}.", main_process_only=True)
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
    data_loaders = make_data_loaders(
        args=args,
        accelerator=accelerator,
        referit_data=referit_data,
        class_to_idx=class_to_idx,
        scans=all_scans_in_dict,
        mean_rgb=mean_rgb,
        tokenizer=tokenizer,
    )

    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx["pad"]
    # Object-type classification
    class_name_list = list(class_to_idx.keys())

    class_name_tokens: BatchEncoding = tokenizer(class_name_list, return_tensors="pt", padding=True)
    class_name_tokens = class_name_tokens.to(device)

    ###################
    #                 #
    #    mvt model    #
    #                 #
    ###################
    if args.mvt_model == "referIt3DNet_transformer":
        mvt3dvg = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    else:
        assert False
    logger.info(
        f"Model {args.mvt_model} architecture: {torchinfo.summary(mvt3dvg, verbose=0)}",
        main_process_only=True,
    )

    # mvt model params
    param_list = list()
    special_lr_dict = {
        "language_encoder": args.init_lr * 0.1,
        "refer_encoder": args.init_lr * 0.1,
    }
    special_mod_names = list(special_lr_dict.keys())
    for name, mod in mvt3dvg.named_children():
        param_list.append(
            {"params": mod.parameters(), "lr": special_lr_dict.get(name, args.init_lr)}
        )
        if name in special_mod_names:
            special_mod_names.remove(name)
    assert len(special_mod_names) == 0, f"Special modules not found: {special_mod_names}"

    if args.mode == "train":
        mvt3dvg = torch.compile(mvt3dvg)
    mvt3dvg = mvt3dvg.to(device)

    #######################
    #                     #
    #    point-e model    #
    #                     #
    #######################
    point_e_config = MODEL_CONFIGS[args.point_e_model]
    point_e_config["cache_dir"] = osp.join(args.project_top_dir, "cache", "point_e_model")
    point_e_config["n_ctx"] = args.points_per_object
    with accelerator.local_main_process_first():
        point_e = model_from_config(point_e_config, device)
        # load pretrained model
        if args.pretrained_point_e:
            logger.info(
                f"Loading pretrained {args.point_e_model} model.",
                main_process_only=True,
            )
            ckpt_state_dict = load_checkpoint(
                args.point_e_model, device=device, cache_dir=point_e_config["cache_dir"]
            )
            point_e.load_state_dict(state_dict=ckpt_state_dict, strict=False)
    point_e.to(device)
    logger.info(
        f"Model {args.point_e_model} architecture: {torchinfo.summary(point_e, verbose=0)}",
        main_process_only=True,
    )

    # construct diffusion
    point_e_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[args.point_e_model])

    # point-e model params
    param_list.append({"params": point_e.parameters(), "lr": args.init_lr * 0.2})
    optimizer = optim.AdamW(param_list, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)

    lr_scheduler = get_linear_scheduler(
        optimizer,
        start_step=args.warmup_steps * accelerator.num_processes,
        end_step=args.global_training_steps * accelerator.num_processes,
        start_lr=args.init_lr,
        end_lr=args.min_lr,
    )

    # adapt with `accelerate`
    (
        mvt3dvg,
        point_e,
        data_loaders["train"],
        data_loaders["test"],
        data_loaders["test_small"],
        optimizer,
        lr_scheduler,
    ) = accelerator.prepare(
        mvt3dvg,
        point_e,
        data_loaders["train"],
        data_loaders["test"],
        data_loaders["test_small"],
        optimizer,
        lr_scheduler,
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
    )

    misc_states = {
        "global_training_steps": 0,
        "training_metric_steps": 0,
        "evaluation_steps": 0,
        "checkpoint_steps": 0,
        "best_test_metric": None,
        "best_test_step": -1,
        "best_test_metric_name": "test_rf3d_loc_estimate_with_top_k_dist"
        if args.axis_norm
        else "test_rf3d_loc_estimate_dist",
        "best_test_metric_comp": operator.lt,
    }

    # Resume training
    if args.resume_path:
        accelerator.load_state(args.resume_path)
        misc_states = torch.load(osp.join(args.resume_path, "misc_states.pkl"))
        # adjust schduler's epoch state
        lr_scheduler.last_epoch = misc_states["global_training_steps"]
        logger.info(f"Resuming training from {args.resume_path}.", main_process_only=True)
        logger.info(
            f"Starting from @step {misc_states['global_training_steps']}.", main_process_only=True
        )

    else:
        if args.mode == "test":
            logger.warning(
                "No resume path provided. Starting evaluation from scratch.", main_process_only=True
            )

    # metrics for evaluation
    with accelerator.local_main_process_first():
        metrics_dict = dict()
        metrics_dict["test"] = {
            "test_rf3d_loss": evaluate.load(
                LOCAL_METRIC_PATHS["average"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_loss",
            ),
            "test_rf3d_loc_estimate": evaluate.load(
                LOCAL_METRIC_PATHS["loc_estimate"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_loc_estimate",
            ),
            "test_rf3d_loc_estimate_with_top_k": evaluate.load(
                LOCAL_METRIC_PATHS["loc_estimate_with_top_k"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_loc_estimate_with_top_k",
            )
            if args.axis_norm
            else None,
            "test_rf3d_cls": evaluate.load(
                LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_cls",
                ignore_label=pad_idx,
            ),
            "test_rf3d_txt": evaluate.load(
                LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_txt",
                ignore_label=pad_idx,
            ),
            "test_point_e_pc_cd": evaluate.load(
                LOCAL_METRIC_PATHS["pairwise_cd"],
                n_features=6,  # xyz, rgb
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_point_e_pc_cd",
            ),
            "test_point_e_pc_cls": evaluate.load(
                LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_point_e_pc_cls",
                # NOTE - no ignored label here
            ),
        }

    ##################
    #                #
    #    Training    #
    #                #
    ##################
    if args.mode == "train":
        logger.info("Starting the training. Good luck!", main_process_only=True)
        with accelerator.local_main_process_first():
            # load metrics
            metrics_dict["train"] = {
                "train_rf3d_loss": evaluate.load(
                    LOCAL_METRIC_PATHS["average"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_loss",
                ),
                "train_point_e_loss": evaluate.load(
                    LOCAL_METRIC_PATHS["average"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_point_e_loss",
                ),
                "train_total_loss": evaluate.load(
                    LOCAL_METRIC_PATHS["average"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_total_loss",
                ),
                "train_rf3d_loc_estimate": evaluate.load(
                    LOCAL_METRIC_PATHS["loc_estimate"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_loc_estimate",
                ),
                "train_rf3d_cls": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_cls",
                    ignore_label=pad_idx,
                ),
                "train_rf3d_txt": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_txt",
                    ignore_label=pad_idx,
                ),
            }

        with tqdm(
            desc="global_steps",
            initial=misc_states["global_training_steps"],
            total=args.global_training_steps,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        ) as bar:
            ckpt_save_dir = osp.join(
                args.project_top_dir,
                args.project_name,
                "checkpoints",
                project_time_str,
            )
            for current_global_steps in train_utils.start_training_loop_steps(
                accelerator=accelerator,
                MVT3DVG=mvt3dvg,
                point_e=point_e,
                sampler=sampler,
                data_loader=data_loaders["train"],
                optimizer=optimizer,
                scheduler=lr_scheduler,
                device=device,
                args=args,
                metrics_=metrics_dict["train"],
                start_global_steps=misc_states["global_training_steps"],
            ):
                # update misc states
                misc_states["global_training_steps"] = current_global_steps
                # training metrics log
                if (
                    current_global_steps
                    >= misc_states["training_metric_steps"] + args.training_metric_interval
                ):
                    logger.debug(
                        f"Logging training metrics @step {current_global_steps}...",
                        main_process_only=True,
                    )
                    train_meters = train_utils.compute_metrics(metrics_dict["train"])
                    # add learning rate to the metrics
                    train_meters["lr"] = max(lr_scheduler.get_last_lr())
                    accelerator.log(train_meters, step=current_global_steps)
                    misc_states["training_metric_steps"] = current_global_steps

                if (
                    current_global_steps
                    >= misc_states["evaluation_steps"] + args.evaluation_interval
                ):
                    # evaluation in training phase
                    logger.info(
                        f"Evaluating @step {current_global_steps}...", main_process_only=True
                    )
                    train_utils.evaluate_on_dataset(
                        accelerator=accelerator,
                        MVT3DVG=mvt3dvg,
                        point_e=point_e,
                        sampler=sampler,
                        data_loader=data_loaders["test_small"],
                        device=device,
                        args=args,
                        metrics_=metrics_dict["test"],
                    )
                    evaluate_meters = train_utils.compute_metrics(metrics_dict["test"])
                    accelerator.log(evaluate_meters, step=current_global_steps)
                    misc_states["evaluation_steps"] = current_global_steps
                    # checkpointing for best model
                    if accelerator.is_local_main_process:
                        best_test_metric_name: str = misc_states["best_test_metric_name"]
                        best_test_metric_comp = misc_states["best_test_metric_comp"]
                        test_metric: float = evaluate_meters[best_test_metric_name]
                        # save best states
                        if misc_states["best_test_metric"] is None or best_test_metric_comp(
                            test_metric, misc_states["best_test_metric"]
                        ):
                            logger.info(
                                colored(
                                    f"Training metric `{best_test_metric_name}`: {test_metric: .4f}, "
                                    f"improved @step {current_global_steps}",
                                    "green",
                                ),
                                main_process_only=True,
                            )
                            old_best_path = (
                                osp.join(
                                    ckpt_save_dir,
                                    f"best-{best_test_metric_name}-"
                                    f"{misc_states['best_test_metric']:.4f}-"
                                    f"step-{misc_states['best_test_step']}",
                                )
                                if misc_states["best_test_metric"] is not None
                                else None
                            )
                            # remove old best
                            if old_best_path is not None and osp.exists(old_best_path):
                                shutil.rmtree(old_best_path)
                            # save new best
                            misc_states["best_test_metric"] = test_metric
                            misc_states["best_test_step"] = current_global_steps
                            ckpt_name = f"best-{best_test_metric_name}-{test_metric:.4f}-step-{current_global_steps}"
                            accelerator.save_state(osp.join(ckpt_save_dir, ckpt_name))
                            accelerator.save(
                                misc_states, osp.join(ckpt_save_dir, ckpt_name, "misc_states.pkl")
                            )
                        else:
                            logger.info(
                                colored(
                                    f"Training metric `{best_test_metric_name}`: {test_metric: .4f}, "
                                    f"did not improve @step {current_global_steps} "
                                    f"since @step {misc_states['best_test_step']}",
                                    "red",
                                ),
                                main_process_only=True,
                            )

                # Checkpoints
                if accelerator.is_local_main_process:
                    if (
                        current_global_steps
                        >= misc_states["checkpoint_steps"] + args.checkpoint_interval
                    ):
                        logger.info(
                            f"Checkpointing @step {current_global_steps}...", main_process_only=True
                        )
                        # save last states
                        ckpt_name = f"ckpt_{current_global_steps}"
                        accelerator.save_state(osp.join(ckpt_save_dir, ckpt_name))
                        misc_states["checkpoint_steps"] = current_global_steps
                        # save misc states
                        accelerator.save(
                            misc_states, osp.join(ckpt_save_dir, ckpt_name, "misc_states.pkl")
                        )

                # bar update
                bar.update(current_global_steps - bar.n)
                bar.refresh()
                if current_global_steps >= args.global_training_steps:
                    break

        accelerator.end_training()
        logger.info("Finished training successfully.", main_process_only=True)

    ##################
    #                #
    #    evaluate    #
    #                #
    ##################
    elif args.mode == "test":
        logger.info("Starting the evaluation. Good luck!", main_process_only=True)
        train_utils.evaluate_on_dataset(
            accelerator=accelerator,
            MVT3DVG=mvt3dvg,
            point_e=point_e,
            sampler=sampler,
            data_loader=data_loaders["test"],
            device=device,
            args=args,
            metrics_=metrics_dict["test"],
        )
        evaluate_meters = train_utils.compute_metrics(metrics_dict["test"])

        if accelerator.is_main_process:
            logger.info(f"Test metrics: ", main_process_only=True)
            logger.info(pprint.pformat(evaluate_meters), main_process_only=True)
            out_path = osp.join(
                args.project_top_dir,
                args.project_name,
                f"test_metrics_{project_time_str}.json",
            )
            logger.info(f"Saving test metrics to {out_path}", main_process_only=True)
            with open(out_path, "w") as f:
                json.dump(evaluate_meters, f, indent=4)


if __name__ == "__main__":
    main()
