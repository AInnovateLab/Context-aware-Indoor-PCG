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
    diff_loss_list = list()
    total_loss_list = list()

    # Set the model in training mode
    MVT3DVG.train()
    point_e.train()

    for batch in tqdm.tqdm(data_loader, disable=not accelerator.is_main_process):
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

        # TODO: change target, and be careful of the pad item, and gather more
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

        rf3d_loss_list.append(RF3D_LOSS.mean())
        diff_loss_list.append(losses.mean())
        total_loss_list.append(LOSS.mean())

        metrics["train_rf3d_loc_estimate"].add_batch(
            predictions=LOCATE_PREDS,
            references=locate_tgt,
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
    loc_estimate = metrics["train_rf3d_loc_estimate"].compute()

    ret = {
        "train_total_loss": np.mean(total_loss_list),
        "train_rf3d_loss": np.mean(rf3d_loss_list),
        "train_diff_loss": np.mean(diff_loss_list),
        "train_rf3d_loc_dist": loc_estimate["dist"],
        "train_rf3d_loc_radius_diff": loc_estimate["radius_diff"],
        "train_rf3d_cls_acc": metrics["train_rf3d_cls_acc"].compute(ignore=pad_idx),
        "train_rf3d_txt_acc": metrics["train_rf3d_txt_acc"].compute(ignore=pad_idx),
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

        # gather for multi-gpu
        (
            LOSS,
            LOCATE_PREDS,
            CLASS_LOGITS,
            LANG_LOGITS,
            locate_tgt,
            batch["ctx_class"],
            batch["tgt_class"],
        ) = accelerator.gather_for_metrics(
            (
                LOSS,
                LOCATE_PREDS,
                CLASS_LOGITS,
                LANG_LOGITS,
                locate_tgt,
                batch["ctx_class"],
                batch["tgt_class"],
            )
        )

        rf3d_loss_list.append(LOSS.mean())

        metrics["test_rf3d_loc_estimate"].add_batch(
            predictions=LOCATE_PREDS,
            references=locate_tgt,
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

        prompts = batch["text"]
        samples = sampler.sample_batch_progressive(
            batch_size=len(batch["text"]),
            ctx_embeds=ctx_embeds,
            model_kwargs=dict(texts=[prompts]),
        )
        pc = sampler.output_to_point_clouds(samples)[0]
        # prompt = prompt.replace(" ", "_")
        # file_out = f"{folder_vis}/{prompt}_seed-{seed}_64steps.ply"
        # file_out = f"test.ply"
        # with open(file_out, "wb") as writer:
        #     pc.write_ply(writer)

        # metrics here
        ...

    #############################
    #                           #
    #    metrics computation    #
    #                           #
    #############################
    loc_estimate = metrics["test_rf3d_loc_estimate"].compute()

    ret = {
        "test_rf3d_loss": np.mean(rf3d_loss_list),
        "test_rf3d_loc_dist": loc_estimate["dist"],
        "test_rf3d_loc_radius_diff": loc_estimate["radius_diff"],
        "test_rf3d_cls_acc": metrics["test_rf3d_cls_acc"].compute(ignore=pad_idx),
        "test_rf3d_txt_acc": metrics["test_rf3d_txt_acc"].compute(ignore=pad_idx),
    }

    return ret
