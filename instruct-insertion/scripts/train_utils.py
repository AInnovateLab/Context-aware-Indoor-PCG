import copy
from typing import Any, Dict

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from model.referit3d_model.referit3d_net import ReferIt3DNet_transformer


def move_batch_to_device_(batch: Dict[str, Any], device):
    for k in batch:
        if isinstance(batch[k], list):
            continue
        elif isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
        else:
            pass

    return batch


def single_epoch_train(
    MVT3DVG: ReferIt3DNet_transformer,
    point_e,
    sampler,
    config,
    data_loader,
    criteria,
    optimizer,
    device,
    pad_idx,
    args,
    tokenizer=None,
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

    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    MVT3DVG.train()
    point_e.train()

    for batch in tqdm.tqdm(data_loader):
        move_batch_to_device_(batch, device)

        lang_tokens = tokenizer(batch["tokens"], return_tensors="pt", padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch["lang_tokens"] = lang_tokens

        # Forward pass
        out_feats, CLASS_LOGITS, LANG_LOGITS = MVT3DVG.first_stage_forward(batch, epoch)

        # NOTE - This is the point_e part
        # train diffusion
        step = 0
        reals = batch["pointcloud"]
        reals = reals.to(device)
        cond = batch["desc"]

        # TODO - Here we need to reshape the tensor from MVT3DVG
        mvt_feats = copy.deepcopy(out_feats)
        mvt_feats = mvt_feats.to(device)

        # TODO - Here we add the tensor from MVT3DVG to point_e
        losses = sampler.loss_texts(mvt_feats, reals, cond, reals.shape[0])
        losses.backward()

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
        LOSS = LOSS.mean()

        res = {}
        res["logits"] = LOGITS
        res["class_logits"] = CLASS_LOGITS
        res["lang_logits"] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch["target_pos"]
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res["logits"], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(
                res["class_logits"], batch["class_labels"], ignore_label=pad_idx
            )
            cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res["lang_logits"], -1)
            cls_b_acc = torch.mean((batch_guess == batch["target_class"]).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics["train_total_loss"] = total_loss_mtr.avg
    metrics["train_referential_acc"] = ref_acc_mtr.avg
    metrics["train_object_cls_acc"] = cls_acc_mtr.avg
    metrics["train_txt_cls_acc"] = txt_acc_mtr.avg
    return metrics
