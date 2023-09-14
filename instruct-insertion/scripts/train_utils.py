import os
from typing import Any, Dict, Generator, Tuple

import accelerate
import evaluate
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from models.point_e_model.diffusion.sampler import PointCloudSampler
from models.point_e_model.util.plotting import plot_point_cloud
from models.point_e_model.util.point_cloud import PointCloud
from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer
from plyfile import PlyData, PlyElement
from transformers import BatchEncoding
from utils import PathLike, create_dir


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


def compute_metrics(metrics: Dict[str, evaluate.EvaluationModule]) -> Dict[str, Any]:
    ret = dict()
    for metric_name, metric in metrics.items():
        metric_ret = metric.compute()
        if isinstance(metric_ret, dict):
            for k, v in metric_ret.items():
                ret[f"{metric_name}_{k}"] = v
        else:
            ret[metric_name] = metric_ret

    return ret


def write_ply(points, save_path: PathLike):
    """
    points: numpy array in shape (N, 6) or (N, 7)
    save_name: str end with ".ply"
    """
    assert points.shape[1] == 6 or points.shape[1] == 7, "points.shape[1] should be 6 or 7"
    save_path = str(save_path)
    assert save_path.endswith(".ply"), "save_name should end with '.ply'"
    points = [
        (points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5])
        for i in range(points.shape[0])
    ]
    vertex = np.array(
        points,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
        ],
    )
    data = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    PlyData([data]).write(save_path)


def start_training_loop_steps(
    accelerator: accelerate.Accelerator,
    MVT3DVG: ReferIt3DNet_transformer,
    point_e: nn.Module,
    sampler: PointCloudSampler,
    data_loader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args,
    metrics_: Dict[str, evaluate.EvaluationModule],
    start_global_steps: int = 0,
) -> Generator[Tuple[int, int], None, None]:
    # Set the model in training mode
    MVT3DVG.train()
    point_e.train()
    current_global_steps = start_global_steps
    gradient_acc_counter = 0

    while True:
        for batch in tqdm.tqdm(
            data_loader,
            desc="local_steps_in_epoch",
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        ):
            with accelerator.accumulate(MVT3DVG, point_e):
                move_batch_to_device_(batch, device)

                # Forward pass
                ctx_embeds, RF3D_LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS = MVT3DVG(batch)

                # NOTE - This is the point_e part
                # train diffusion
                reals = batch["tgt_pc"][:, :, :6]  # (B, P, 6)
                ## rgb rescale to [0, 255]
                reals[:, :, 3:] = reals[:, :, 3:].mul(255.0)
                cond = batch["text"]  # List of str

                # Here we add the tensor from MVT3DVG to point_e
                losses = sampler.loss_texts(ctx_embeds, reals, cond, reals.shape[0])

                LOSS: torch.Tensor = RF3D_LOSS.mean() + losses.mean()

                # Backward
                optimizer.zero_grad()
                accelerator.backward(LOSS)
                optimizer.step()

                gradient_acc_counter += 1
                if gradient_acc_counter % accelerator.gradient_accumulation_steps == 0:
                    gradient_acc_counter = 0
                    current_global_steps += 1
                    # scheduler
                    scheduler.step()

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

                metrics_["train_rf3d_loss"].add(predictions=float(RF3D_LOSS.mean()))
                metrics_["train_point_e_loss"].add(predictions=float(losses.mean()))
                metrics_["train_total_loss"].add(predictions=float(LOSS.mean()))

                metrics_["train_rf3d_loc_estimate"].add_batch(
                    predictions=LOCATE_PREDS.float(),
                    references=locate_tgt.float(),
                )

                if args.obj_cls_alpha > 0:
                    metrics_["train_rf3d_cls"].add_batch(
                        predictions=CLASS_LOGITS.argmax(-1).flatten(),
                        references=batch["ctx_class"].flatten(),
                    )

                if args.lang_cls_alpha > 0:
                    metrics_["train_rf3d_txt"].add_batch(
                        predictions=LANG_LOGITS.argmax(-1),
                        references=batch["tgt_class"],
                    )

            yield current_global_steps


@torch.no_grad()
def evaluate_on_dataset(
    accelerator: accelerate.Accelerator,
    MVT3DVG: ReferIt3DNet_transformer,
    point_e: nn.Module,
    sampler: PointCloudSampler,
    data_loader,
    device: torch.device,
    args,
    metrics_: Dict[str, evaluate.EvaluationModule],
):
    MVT3DVG.eval()
    point_e.eval()
    MVT3DVG = accelerator.unwrap_model(MVT3DVG)

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

        #######################
        #                     #
        #    Visualization    #
        #                     #
        #######################
        # Visualization of the point cloud
        aux_input = {}
        for i, name in enumerate(sampler.aux_channels):
            v = aux[:, i]
            aux_input[name] = v

        for batch_idx in range(last_pcs.shape[0]):
            demo_dir = os.path.join(
                args.project_top_dir,
                args.project_name,
                "vis",
            )
            create_dir(demo_dir)
            stimulus_id = batch["stimulus_id"][batch_idx]

            pc = PointCloud(
                coords=pos[batch_idx].t().cpu().numpy(),
                channels={k: v[batch_idx].cpu().numpy() for k, v in aux_input.items()},
            )
            plot_point_cloud(
                pc,
                color=True,
                grid_size=1,
                fixed_bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                area=1,
                save_path=os.path.join(demo_dir, f"{stimulus_id}.png"),
            )

            tgt_pc = batch["tgt_pc"][batch_idx]  # (P, 6 or 7)
            raw_pc = PointCloud(
                coords=tgt_pc[:, :3].cpu().numpy(),
                channels={
                    "R": tgt_pc[:, 3].cpu().numpy(),
                    "G": tgt_pc[:, 4].cpu().numpy(),
                    "B": tgt_pc[:, 5].cpu().numpy(),
                },
            )
            plot_point_cloud(
                raw_pc,
                color=True,
                grid_size=1,
                fixed_bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                area=1,
                save_path=os.path.join(demo_dir, f"raw_{stimulus_id}.png"),
            )

            # coords = pos[batch_idx].permute(1, 0)  # (P, 3)
            # coords = (
            #     coords * batch["tgt_box_max_dist"][batch_idx] + batch["tgt_box_center"][batch_idx]
            # )
            # colors = aux[batch_idx].permute(1, 0).mul_(255.0)  # (P, 3 or 4)
            # vis_pc = torch.cat((coords, colors), dim=-1)  # (P, 6 or 7)

            # TODO - Need test
            # write_ply(vis_pc, output_file=os.path.join(demo_dir, f"{stimulus_id}.ply"))

            # NOTE - break here since only save the first img each batch
            break

        diff_pcs = torch.cat((pos, aux), dim=1)  # (B, D=6, P)
        diff_pcs = diff_pcs.transpose(1, 2)  # (B, P, D=6)
        # NOTE - produce the `height append`
        if args.height_append:
            tgt_pc_height = batch["tgt_pc"][:, :, -1:]  # (B, P, 1)
            # avg height
            tgt_pc_height = tgt_pc_height.mean(dim=1, keepdim=True).repeat(
                1, diff_pcs.shape[1], 1
            )  # (B, P, 1)
            diff_pcs = torch.cat((diff_pcs, tgt_pc_height), dim=-1)  # (B, P, D=7)
        _, TGT_CLASS_LOGITS = MVT3DVG.forward_obj_cls(
            diff_pcs[:, None, :, :]
        )  # (B, 1, # of classes)

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
            TGT_CLASS_LOGITS,
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
                TGT_CLASS_LOGITS,
                batch["ctx_class"],
                batch["tgt_class"],
                batch["tgt_pc"],
            )
        )

        metrics_["test_rf3d_loss"].add(predictions=float(LOSS.mean()))

        metrics_["test_rf3d_loc_estimate"].add_batch(
            predictions=LOCATE_PREDS.float(),
            references=locate_tgt.float(),
        )

        if args.obj_cls_alpha > 0:
            metrics_["test_rf3d_cls"].add_batch(
                predictions=CLASS_LOGITS.argmax(-1).flatten(),
                references=batch["ctx_class"].flatten(),
            )

        if args.lang_cls_alpha > 0:
            metrics_["test_rf3d_txt"].add_batch(
                predictions=LANG_LOGITS.argmax(-1),
                references=batch["tgt_class"],
            )

        metrics_["test_point_e_pc_cd"].add_batch(
            predictions=diff_pcs[..., :6].float(),
            references=batch["tgt_pc"][..., :6].float(),
        )

        metrics_["test_point_e_pc_cls"].add_batch(
            predictions=TGT_CLASS_LOGITS.argmax(-1).flatten(),
            references=batch["tgt_class"],
        )
