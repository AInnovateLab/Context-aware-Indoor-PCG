from collections import defaultdict
from typing import Any, Dict

import accelerate
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from models.point_e_model.diffusion.sampler import PointCloudSampler
from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer
from transformers import BatchEncoding


def move_batch_to_device_(batch: Dict[str, Any], device):
    for k in batch:
        if isinstance(batch[k], list):
            continue
        elif isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        elif isinstance(batch[k], BatchEncoding):
            batch[k] = batch[k].to(device)
        else:
            pass

    return batch


def single_epoch_train(
    accelerator: accelerate.Accelerator,
    MVT3DVG: ReferIt3DNet_transformer,
    point_e: nn.Module,
    sampler: PointCloudSampler,
    data_loader,
    optimizer,
    device: torch.device,
    pad_idx: int,
    args,
    metrics: Dict[str, evaluate.EvaluationModule],
    epoch=None,
):
    rf3d_loss_list = list()
    point_e_loss_list = list()
    total_loss_list = list()

    # Set the model in training mode
    MVT3DVG.train()
    point_e.train()

    for batch in tqdm.tqdm(data_loader, disable=not accelerator.is_main_process):
        with accelerator.accumulate(MVT3DVG, point_e):
            move_batch_to_device_(batch, device)

            # Forward pass
            ctx_embeds, RF3D_LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS = MVT3DVG(batch)

            # NOTE - This is the point_e part
            # train diffusion
            reals = batch["tgt_pc"][:, :, :6]  # (B, P, 6 or 7)
            cond = batch["text"]  # List of str

            # Here we add the tensor from MVT3DVG to point_e
            losses = sampler.loss_texts(ctx_embeds, reals, cond, reals.shape[0])

            LOSS: torch.Tensor = RF3D_LOSS.mean() + losses.mean()

            # Backward
            optimizer.zero_grad()
            accelerator.backward(LOSS)
            optimizer.step()

            # Update the loss and accuracy meters
            locate_tgt = torch.concat(
                (batch["tgt_box_center"], batch["tgt_box_max_dist"][:, None]), dim=-1
            )

            # gather for multi-gpu
            (
                LOSS,
                LOCATE_PREDS,
                CLASS_LOGITS,
                LANG_LOGITS,
                RF3D_LOSS,
                losses,
                locate_tgt,
                batch["ctx_class"],
                batch["tgt_class"],
            ) = accelerator.gather_for_metrics(
                (
                    LOSS,
                    LOCATE_PREDS,
                    CLASS_LOGITS,
                    LANG_LOGITS,
                    RF3D_LOSS,
                    losses,
                    locate_tgt,
                    batch["ctx_class"],
                    batch["tgt_class"],
                )
            )

            rf3d_loss_list.append(float(RF3D_LOSS.mean()))
            point_e_loss_list.append(float(losses.mean()))
            total_loss_list.append(float(LOSS.mean()))

            metrics["train_rf3d_loc_estimate"].add_batch(
                predictions=LOCATE_PREDS.float(),
                references=locate_tgt.float(),
            )

            if args.obj_cls_alpha > 0:
                metrics["train_rf3d_cls_acc"].add_batch(
                    predictions=CLASS_LOGITS.argmax(-1).flatten(),
                    references=batch["ctx_class"].flatten(),
                )

            if args.lang_cls_alpha > 0:
                metrics["train_rf3d_txt_acc"].add_batch(
                    predictions=LANG_LOGITS.argmax(-1),
                    references=batch["tgt_class"],
                )

    #############################
    #                           #
    #    metrics computation    #
    #                           #
    #############################
    if accelerator.is_main_process:
        loc_estimate = metrics["train_rf3d_loc_estimate"].compute()
        rf3d_cls_acc = metrics["train_rf3d_cls_acc"].compute(ignore_label=pad_idx)
        rf3d_txt_acc = metrics["train_rf3d_txt_acc"].compute(ignore_label=pad_idx)
    else:
        _ = metrics["train_rf3d_loc_estimate"].compute()
        _ = metrics["train_rf3d_cls_acc"].compute(ignore_label=pad_idx)
        _ = metrics["train_rf3d_txt_acc"].compute(ignore_label=pad_idx)
        loc_estimate = defaultdict(float)
        rf3d_cls_acc = defaultdict(float)
        rf3d_txt_acc = defaultdict(float)

    ret = {
        "train_total_loss": np.mean(total_loss_list),
        "train_rf3d_loss": np.mean(rf3d_loss_list),
        "train_point_e_loss": np.mean(point_e_loss_list),
        "train_rf3d_loc_dist": loc_estimate["dist"],
        "train_rf3d_loc_radius_diff": loc_estimate["radius_diff"],
        "train_rf3d_cls_acc": rf3d_cls_acc["accuracy"],
        "train_rf3d_txt_acc": rf3d_txt_acc["accuracy"],
    }

    return ret


@torch.no_grad()
def evaluate_on_dataset(
    accelerator: accelerate.Accelerator,
    MVT3DVG: ReferIt3DNet_transformer,
    point_e: nn.Module,
    sampler: PointCloudSampler,
    data_loader,
    device: torch.device,
    pad_idx: int,
    args,
    metrics: Dict[str, evaluate.EvaluationModule],
    epoch=None,
):
    MVT3DVG.eval()
    point_e.eval()

    rf3d_loss_list = list()

    for batch in tqdm.tqdm(data_loader, disable=not accelerator.is_main_process):
        move_batch_to_device_(batch, device)
        ctx_embeds, LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS = MVT3DVG(batch)

        ######################
        #                    #
        #    evaluate mvt    #
        #                    #
        ######################
        # Update the loss and accuracy meters
        locate_tgt = torch.concat(
            (batch["tgt_box_center"], batch["tgt_box_max_dist"][:, None]), dim=-1
        )

        #########################
        #                       #
        #    evalute point-e    #
        #                       #
        #########################
        prompts = batch["text"]
        # stack twice for guided scale
        ctx_embeds = torch.cat((ctx_embeds, ctx_embeds), dim=0)
        samples_it = sampler.sample_batch_progressive(
            batch_size=len(prompts),
            ctx_embeds=ctx_embeds,
            model_kwargs=dict(texts=prompts),
            accelerator=accelerator,
        )
        # get the last timestep prediction
        for last_pcs in samples_it:
            pass
        # last_pcs: (B, C, P), requires colors scales
        pos = last_pcs[:, :3, :]
        aux = last_pcs[:, 3:, :]
        aux = aux.clamp_(0, 255).round_().div_(255.0)
        diff_pcs = torch.cat((pos, aux), dim=1)  # (B, C, P)
        diff_pcs = diff_pcs.transpose(1, 2)  # (B, P, C)

        ########################
        #                      #
        #    metrics update    #
        #                      #
        ########################
        # gather for multi-gpu
        (
            LOSS,
            LOCATE_PREDS,
            CLASS_LOGITS,
            LANG_LOGITS,
            locate_tgt,
            diff_pcs,
            batch["ctx_class"],
            batch["tgt_class"],
            batch["tgt_pc"],
        ) = accelerator.gather_for_metrics(
            (
                LOSS,
                LOCATE_PREDS,
                CLASS_LOGITS,
                LANG_LOGITS,
                locate_tgt,
                diff_pcs,
                batch["ctx_class"],
                batch["tgt_class"],
                batch["tgt_pc"],
            )
        )

        rf3d_loss_list.append(float(LOSS.mean()))

        metrics["test_rf3d_loc_estimate"].add_batch(
            predictions=LOCATE_PREDS.float(),
            references=locate_tgt.float(),
        )

        if args.obj_cls_alpha > 0:
            metrics["test_rf3d_cls_acc"].add_batch(
                predictions=CLASS_LOGITS.argmax(-1).flatten(),
                references=batch["ctx_class"].flatten(),
            )

        if args.lang_cls_alpha > 0:
            metrics["test_rf3d_txt_acc"].add_batch(
                predictions=LANG_LOGITS.argmax(-1),
                references=batch["tgt_class"],
            )

        metrics["test_point_e_pc_cd"].add_batch(
            predictions=diff_pcs[..., :6].float(),
            references=batch["tgt_pc"][..., :6].float(),
        )

    #############################
    #                           #
    #    metrics computation    #
    #                           #
    #############################
    if accelerator.is_main_process:
        loc_estimate = metrics["test_rf3d_loc_estimate"].compute()
        poine_e_pc_cd = metrics["test_point_e_pc_cd"].compute()
        rf3d_cls_acc = metrics["test_rf3d_cls_acc"].compute(ignore_label=pad_idx)
        rf3d_txt_acc = metrics["test_rf3d_txt_acc"].compute(ignore_label=pad_idx)
    else:
        _ = metrics["test_rf3d_loc_estimate"].compute()
        _ = metrics["test_point_e_pc_cd"].compute()
        _ = metrics["test_rf3d_cls_acc"].compute(ignore_label=pad_idx)
        _ = metrics["test_rf3d_txt_acc"].compute(ignore_label=pad_idx)
        loc_estimate = defaultdict(float)
        poine_e_pc_cd = defaultdict(float)
        rf3d_cls_acc = defaultdict(float)
        rf3d_txt_acc = defaultdict(float)

    ret = {
        "test_rf3d_loss": np.mean(rf3d_loss_list),
        "test_rf3d_loc_dist": loc_estimate["dist"],
        "test_rf3d_loc_radius_diff": loc_estimate["radius_diff"],
        "test_rf3d_cls_acc": rf3d_cls_acc["accuracy"],
        "test_rf3d_txt_acc": rf3d_txt_acc["accuracy"],
        "test_point_e_pc_cd_dist": poine_e_pc_cd["distance"],
        "test_point_e_pc_cd_feat_diff": poine_e_pc_cd["feat_diff"],
    }

    return ret
