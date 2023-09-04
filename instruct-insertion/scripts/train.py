#!/usr/bin/env python
# coding: utf-8

import json
import os.path as osp
import pprint
import sys
import time
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, set_seed
from metrics import LOCAL_METRIC_PATHS
from torch import optim
from train_utils import evaluate_on_dataset, single_epoch_train

##################################
#                                #
#    Visual grounding related    #
#                                #
##################################
# isort: split
from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer
from transformers import BatchEncoding, BertTokenizer

# from models.referit3d_model.utils.tf_visualizer import Visualizer

###########################
#                         #
#    Diffusion related    #
#                         #
###########################
# isort: split

from models.point_e_model.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e_model.diffusion.sampler import PointCloudSampler
from models.point_e_model.models.configs import MODEL_CONFIGS, model_from_config
from models.point_e_model.util.common import get_linear_scheduler
from models.point_e_model.util.plotting import plot_point_cloud
from models.point_e_model.util.point_cloud import PointCloud

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

    acc_config = ProjectConfiguration(
        project_dir=osp.join(args.project_top_dir, args.project_name),
        logging_dir=osp.join(args.project_top_dir, "tf_logs"),
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="tensorboard", project_config=acc_config, kwargs_handlers=[ddp_kwargs]
    )
    # tracker setup
    accelerator.init_trackers(args.project_name, config=vars(args))
    # save log file only to main process
    project_start_time = datetime.now()
    init_logger(
        accelerator,
        log_file=osp.join(
            args.project_top_dir,
            args.project_name,
            "logs",
            project_start_time.strftime(DATE_FMT),
        ),
    )
    logger = get_logger(__name__)
    logger.info(
        f"Project {args.project_name} start at {project_start_time.strftime(DATE_FMT)}",
        main_process_only=True,
    )
    if accelerator.is_main_process and accelerator.num_processes > 1:
        logger.info(
            f"Distributed training with {accelerator.num_processes} processes.",
            main_process_only=True,
        )
    device = accelerator.device
    set_seed(args.random_seed)
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
    mvt3dvg = mvt3dvg.to(device)
    logger.info(
        f"Model {args.mvt_model} architecture: {torchinfo.summary(mvt3dvg)}", main_process_only=True
    )

    # model params
    param_list = [
        {"params": mvt3dvg.obj_encoder.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.obj_encoder_agg_proj.parameters(), "lr": args.init_lr},
        #
        {"params": mvt3dvg.language_encoder.parameters(), "lr": args.init_lr * 0.1},
        #
        {"params": mvt3dvg.refer_encoder.parameters(), "lr": args.init_lr * 0.1},
        #
        {"params": mvt3dvg.box_feature_mapping.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.box_layers.parameters(), "lr": args.init_lr},
        #
        {"params": mvt3dvg.locate_token.parameters(), "lr": args.init_lr},
        #
        {"params": mvt3dvg.language_clf.parameters(), "lr": args.init_lr},
        {"params": mvt3dvg.object_language_clf.parameters(), "lr": args.init_lr},
    ]
    if not args.label_lang_sup:
        param_list.append({"params": mvt3dvg.obj_clf.parameters(), "lr": args.init_lr})

    #######################
    #                     #
    #    point-e model    #
    #                     #
    #######################
    point_e_config = MODEL_CONFIGS[args.point_e_model]
    point_e_config["cache_dir"] = osp.join(args.project_top_dir, "cache", "point_e_model")
    with accelerator.local_main_process_first():
        point_e = model_from_config(point_e_config, device)
    point_e.to(device)
    logger.info(
        f"Model {args.point_e_model} architecture: {torchinfo.summary(point_e)}",
        main_process_only=True,
    )

    # construct diffusion
    point_e_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[args.point_e_model])

    param_list.append({"params": point_e.parameters(), "lr": args.init_lr})
    optimizer = optim.AdamW(param_list)

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
    )

    best_test_cd_dist = float("inf")
    best_test_epoch = -1

    # Resume training
    if args.resume_path:
        accelerator.load_state(args.resume_path)
        logger.info(f"Resuming training from {args.resume_path}.", main_process_only=True)
        start_training_epoch = lr_scheduler.last_epoch + 1
        logger.info(f"Starting from epoch {start_training_epoch}.", main_process_only=True)
        # TODO - search in the same folder for the best model metrics so far
        # get checkpoint name
        checkpoint_name: str = osp.basename(args.resume_path)
        # format check
        if checkpoint_name.startswith("best_"):
            splits = checkpoint_name.split("_")
            if splits[1] == "cd" and splits[3] == "epoch":
                best_test_cd_dist = float(splits[2])
                best_test_epoch = int(splits[4])
            else:
                logger.warning(
                    "Incompatible checkpoint name. Cannot resume best test metrics.",
                    main_process_only=True,
                )

    else:
        start_training_epoch = 0
        if args.mode == "test":
            logger.warning(
                "No resume path provided. Starting evaluation from scratch.", main_process_only=True
            )

    # metrics for evaluation
    with accelerator.local_main_process_first():
        test_metrics = {
            "test_rf3d_loc_estimate": evaluate.load(
                LOCAL_METRIC_PATHS["loc_estimate"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_loc_estimate",
            ),
            "test_rf3d_cls_acc": evaluate.load(
                LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_cls_acc",
            ),
            "test_rf3d_txt_acc": evaluate.load(
                LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_rf3d_txt_acc",
            ),
            "test_point_e_pc_cd": evaluate.load(
                LOCAL_METRIC_PATHS["pairwise_cd"],
                n_features=6,  # xyz, rgb
                process_id=accelerator.process_index,
                num_process=accelerator.num_processes,
                experiment_id="test_point_e_pc_cd",
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
            metrics = {
                "train_rf3d_loc_estimate": evaluate.load(
                    LOCAL_METRIC_PATHS["loc_estimate"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_loc_estimate",
                ),
                "train_rf3d_cls_acc": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_cls_acc",
                ),
                "train_rf3d_txt_acc": evaluate.load(
                    LOCAL_METRIC_PATHS["accuracy_with_ignore_label"],
                    process_id=accelerator.process_index,
                    num_process=accelerator.num_processes,
                    experiment_id="train_rf3d_txt_acc",
                ),
            }
            metrics.update(test_metrics)

        with tqdm(
            range(start_training_epoch, args.max_train_epochs),
            desc="epochs",
            disable=not accelerator.is_main_process,
        ) as bar:
            for epoch in bar:
                logger.info(f"Current LR: {lr_scheduler.get_last_lr()}", main_process_only=True)
                # Train:
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
                # add learning rate to the metrics
                train_meters["lr"] = max(lr_scheduler.get_last_lr())

                accelerator.log(train_meters, step=epoch)

                evaluate_meters = evaluate_on_dataset(
                    accelerator=accelerator,
                    MVT3DVG=mvt3dvg,
                    point_e=point_e,
                    sampler=sampler,
                    data_loader=data_loaders["test_small"],
                    device=device,
                    pad_idx=pad_idx,
                    args=args,
                    metrics=metrics,
                )

                test_cd_dist: float = evaluate_meters["test_point_e_pc_cd"]

                lr_scheduler.step()

                # Checkpoints
                if accelerator.is_main_process:
                    # save last states
                    accelerator.save_state(
                        osp.join(args.project_top_dir, args.project_name, "checkpoints", "last")
                    )

                    # save best states
                    if best_test_cd_dist > test_cd_dist:
                        logger.info(
                            colored(
                                f"Training test CD distance: {best_test_cd_dist: .4f}, improved @epoch {epoch}",
                                "green",
                            ),
                            main_process_only=True,
                        )
                        best_test_cd_dist = test_cd_dist
                        best_test_epoch = epoch
                        accelerator.save_state(
                            osp.join(
                                args.project_top_dir,
                                args.project_name,
                                "checkpoints",
                                f"best_cd_{test_cd_dist:.4f}_epoch_{epoch}",
                            )
                        )
                    else:
                        logger.info(
                            colored(
                                f"Training test CD distance: {best_test_cd_dist: .4f}, did not improve @epoch {epoch} since @epoch {best_test_epoch}",
                                "red",
                            ),
                            main_process_only=True,
                        )

                bar.refresh()

        accelerator.end_training()
        logger.info("Finished training successfully.", main_process_only=True)

    ##################
    #                #
    #    evaluate    #
    #                #
    ##################
    elif args.mode == "test":
        logger.info("Starting the evaluation. Good luck!", main_process_only=True)
        metrics = test_metrics
        evaluate_meters = evaluate_on_dataset(
            accelerator=accelerator,
            MVT3DVG=mvt3dvg,
            point_e=point_e,
            sampler=sampler,
            data_loader=data_loaders["test_small"],
            device=device,
            pad_idx=pad_idx,
            args=args,
            metrics=metrics,
        )

        if accelerator.is_main_process:
            logger.info(f"Test metrics: ", main_process_only=True)
            logger.info(pprint.pformat(evaluate_meters), main_process_only=True)
            out_path = osp.join(
                args.project_top_dir,
                args.project_name,
                f"test_metrics_{project_start_time.strftime(DATE_FMT)}.json",
            )
            logger.info(f"Saving test metrics to {out_path}", main_process_only=True)
            with open(out_path, "w") as f:
                json.dump(evaluate_meters, f, indent=4)


if __name__ == "__main__":
    main()
