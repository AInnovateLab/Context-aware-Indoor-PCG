import copy
from typing import Any, Dict

import accelerate
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from model.referit3d_model.referit3d_net import ReferIt3DNet_transformer
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
    sampler,
    data_loader,
    optimizer,
    device: torch.device,
    pad_idx: int,
    args,
    metrics: Dict[str, evaluate.EvaluationModule],
    epoch=None,
):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """

    total_loss_list = list()
    total_loss_weights = list()

    # Set the model in training mode
    MVT3DVG.train()
    point_e.train()

    for batch in tqdm.tqdm(data_loader, disable=not accelerator.is_main_process):
        move_batch_to_device_(batch, device)

        # Forward pass
        out_feats, CLASS_LOGITS, LANG_LOGITS = MVT3DVG.first_stage_forward(batch)

        # NOTE - This is the point_e part
        # train diffusion
        step = 0
        reals = batch["pointcloud"]
        reals = reals.to(device)
        cond = batch["desc"]

        # TODO - Here we need to reshape the tensor from MVT3DVG
        mvt_feats = out_feats

        # TODO - Here we add the tensor from MVT3DVG to point_e
        losses = sampler.loss_texts(mvt_feats, reals, cond, reals.shape[0])

        # NOTE - logger and model saving, this need to be reconsider
        # if env.is_master() and step % config["echo_every"] == 0:
        #     logger.info(
        #         f"Epoch: {epoch}, step: {step}, lr:{cur_lr:.6f}, losses: {losses.item():g}"
        #     )
        #     writer.add_scalar("losses", losses.item(), global_step=step)

        # if config["evaluate_every"] > 0 and step > 0 and step % config["evaluate_every"] == 0:
        #     test(step)
        # if env.is_master() and step > 0 and step % config["save_every"] == 0:
        #     save()
        step += 1

        # TODO - Redesign the loss function, should we put them together?
        # continue training MVT3DVG
        LOSS, LOGITS = MVT3DVG.second_stage_forward(out_feats, batch, CLASS_LOGITS, LANG_LOGITS)
        LOSS: torch.Tensor = LOSS.mean() + losses.mean()

        res = {}
        res["logits"] = LOGITS
        res["class_logits"] = CLASS_LOGITS
        res["lang_logits"] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        accelerator.backward(LOSS)
        optimizer.step()
        total_loss_list.append(LOSS.item())

        # Update the loss and accuracy meters
        target = batch["target_pos"]

        predictions = torch.argmax(res["logits"], dim=1)

        # TODO: change target, and be careful of the pad item, and gather more
        predictions, target = accelerator.gather_for_metrics((predictions, target))

        metrics["referit3d_loc_acc"].add_batch(
            predictions=predictions,
            references=target,
        )

        if args.obj_cls_alpha > 0:
            metrics["referit3d_cls_acc"].add_batch(
                predictions=res["class_logits"].argmax(-1).flatten(),
                references=batch["class_labels"].flatten(),
            )

        if args.lang_cls_alpha > 0:
            metrics["referit3d_txt_acc"].add_batch(
                predictions=res["lang_logits"].argmax(-1),
                references=batch["target_class"],
            )

    # metrics["train_total_loss"] = total_loss_mtr.avg
    # metrics["train_referential_acc"] = ref_acc_mtr.avg
    # metrics["train_object_cls_acc"] = cls_acc_mtr.avg
    # metrics["train_txt_cls_acc"] = txt_acc_mtr.avg
    return {
        "referit3d_loss": np.mean(total_loss_list),
        "referit3d_loc_acc": metrics["referit3d_loc_acc"].compute(ignore=pad_idx),
        "referit3d_cls_acc": metrics["referit3d_cls_acc"].compute(ignore=pad_idx),
        "referit3d_txt_acc": metrics["referit3d_txt_acc"].compute(ignore=pad_idx),
    }
