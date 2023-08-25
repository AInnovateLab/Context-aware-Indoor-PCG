import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from model.point_e.configs.config import load_config
from model.point_e.diffusion.sampler import PointCloudSampler
from model.point_e.models.configs import MODEL_CONFIGS, model_from_config
from model.point_e.util.common import get_linear_scheduler
from model.referit3d.utils.evaluation import AverageMeter
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ["objects", "tokens", "target_pos"]  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append("class_labels")

    if args.lang_cls_alpha > 0:
        batch_keys.append("target_class")

    return batch_keys


def single_epoch_train(
    MVT3DVG,
    point_e,
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
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)
    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(device)

        # if args.object_encoder == 'pnet':
        #     batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        lang_tokens = tokenizer(batch["tokens"], return_tensors="pt", padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].cuda()
        batch["lang_tokens"] = lang_tokens

        # Forward pass
        out_feats, CLASS_LOGITS, LANG_LOGITS = MVT3DVG.first_stage_forward(batch, epoch)

        # NOTE - This is the point_e part
        model_config = config["model"]
        dataset_config = config["dataset"]
        opt_config = config["optimizer"]
        sched_config = config["lr_sched"]
        max_epoches = config["max_epoches"]
        name = model_config["name"]

        # TODO - Fix sample
        # construct sampler
        train_sampler = DistributedSampler(train_set)
        train_dl = torch.utils.data.DataLoader(
            train_set,
            batch_size=model_config["batch_size"],
            num_workers=args.num_workers,
            persistent_workers=True,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=True,
        )

        # construct learning rate scheduler
        max_steps = max_epoches * len(train_dl)
        if sched_config["type"] == "linear":
            sched = get_linear_scheduler(
                opt,
                start_epoch=0,
                end_epoch=max_steps,
                start_lr=opt_config["lr"],
                end_lr=sched_config["min_lr"],
            )
        else:
            assert False, "sched type not support"

        # TODO - Do we need aux_channels?
        aux_channels = [] if "3channel" in model_config["name"] else ["R", "G", "B"]
        sampler = PointCloudSampler(
            device=device,
            models=[base_model],
            diffusions=[base_diffusion],
            num_points=[model_config["num_points"]],
            aux_channels=aux_channels,
            guidance_scale=[3.0],
            use_karras=[True],
            karras_steps=[64],
            sigma_min=[model_config["sigma_min"]],
            sigma_max=[model_config["sigma_max"]],
            s_churn=[3],
            model_kwargs_key_filter=[args.cond],
        )

        # train diffusion
        step = 0
        train_sampler.set_epoch(epoch)
        for batch in train_dl:
            cur_lr = sched.get_last_lr()[0]
            point_e.train()  # train mode
            opt.zero_grad()
            reals = batch["pointcloud"]
            reals = reals.to(device)

            cond = batch["desc"]
            # TODO - Here we add the tensor from MVT3DVG to point_e
            losses = sampler.loss_texts(reals, cond, reals.shape[0])
            losses.backward()
            opt.step()
            sched.step()

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


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert type(correct_guessed) == torch.Tensor

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
