import json
import os
import os.path as osp
import sys

import accelerate
import numpy as np
import pyvista as pv
import torch
from easydict import EasyDict as edict
from termcolor import colored

sys.path.append(os.path.join(os.getcwd(), ".."))

accelerator = accelerate.Accelerator()
device = accelerator.device

# load existing args
PROJECT_TOP_DIR = "../../tmp_link_saves"
PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_rr4_sr3d")
with open(osp.join(PROJECT_DIR, "config.json.txt"), "r") as f:
    args = edict(json.load(f))

from data.referit3d.in_out.neural_net_oriented import (
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
    trim_scans_per_referit3d_data_,
)

# load data
SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
REFERIT_CSV_FILE = "../../datasets/nr3d/nr3d_generative_20230825_final.csv"
all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)
referit_data = load_referential_data(args, args.referit3D_file, scans_split)
# Prepare data & compute auxiliary meta-information.
all_scans_in_dict = trim_scans_per_referit3d_data_(referit_data, all_scans_in_dict)
mean_rgb = compute_auxiliary_data(referit_data, all_scans_in_dict)

from transformers import BertTokenizer

# prepare tokenizer
tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
# Prepare the Listener
n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
pad_idx = class_to_idx["pad"]
# Object-type classification
class_name_list = list(class_to_idx.keys())

class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
class_name_tokens = class_name_tokens.to(device)

from data.referit3d.datasets import make_data_loaders

data_loaders = make_data_loaders(
    args=args,
    accelerator=accelerator,
    referit_data=referit_data,
    class_to_idx=class_to_idx,
    scans=all_scans_in_dict,
    mean_rgb=mean_rgb,
    tokenizer=tokenizer,
)

from models.point_e_model.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from models.point_e_model.diffusion.sampler import PointCloudSampler
from models.point_e_model.models.configs import MODEL_CONFIGS, model_from_config

# Load models
from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer

# referit3d model
mvt3dvg = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
# point-e model
point_e_config = MODEL_CONFIGS[args.point_e_model]
point_e_config["cache_dir"] = osp.join(PROJECT_TOP_DIR, "cache", "point_e_model")
point_e_config["n_ctx"] = args.points_per_object
point_e = model_from_config(point_e_config, device)
point_e_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[args.point_e_model])
# move models to gpu
mvt3dvg = mvt3dvg.to(device).eval()
point_e = point_e.to(device).eval()

# load model and checkpoints
if args.mode == "train":
    mvt3dvg = torch.compile(mvt3dvg)
mvt3dvg, point_e = accelerator.prepare(mvt3dvg, point_e)
CHECKPOINT_DIR = osp.join(PROJECT_DIR, "checkpoints", "2023-09-21_18-18-07", "ckpt_240000")
accelerator.load_state(CHECKPOINT_DIR)

from models.point_e_model.diffusion.sampler import PointCloudSampler

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

import pickle

from EMD_evaluation.emd_module import emd_eval

# TODO - Download the object_dict to local disk

# Read object_dict from local disk
with open("object_dict_testset.pkl", "rb") as f:
    object_dict = pickle.load(f)

idx_has_been_used = []
avg_emd_of_classes = {}
num_of_obj_in_classes = {}

# for i in range(2000):
while len(idx_has_been_used) < 1000:
    from scripts.train_utils import move_batch_to_device_

    # get random data
    test_dataset = data_loaders["test"].dataset
    rand_idx = np.random.randint(0, len(test_dataset))
    if rand_idx not in idx_has_been_used:
        idx_has_been_used.append(rand_idx)
    else:
        continue
    rand_data = test_dataset[rand_idx]
    rand_data_scan, rand_data_target_objs = test_dataset.get_reference_data(rand_idx)[:2]
    rand_data_3d_objs = rand_data_scan.three_d_objects.copy()
    rand_data_3d_objs.remove(rand_data_target_objs)
    # rand_data["text"] = "Create a chair on the ground in the corner."
    # rand_data["tokens"] = test_dataset.tokenizer(rand_data["text"], max_length=test_dataset.max_seq_len, truncation=True, padding=False)
    collate_fn = data_loaders["test"].collate_fn
    # get batch
    batch = collate_fn([rand_data])
    batch = move_batch_to_device_(batch, device)

    with torch.no_grad():
        ctx_embeds, LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS, pred_xyz = mvt3dvg(batch)

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
        last_pcs = last_pcs.permute(0, 2, 1)

        # For axis_norm model
        pred_xy, pred_z, pred_radius = pred_xyz
        pred_xy_topk_bins = pred_xy.topk(5, dim=-1)[1]  # (B, 5)
        pred_z_topk_bins = pred_z.topk(5, dim=-1)[1]  # (B, 5)
        pred_x_topk_bins = pred_xy_topk_bins % args.axis_norm_bins  # (B, 5)
        pred_y_topk_bins = pred_xy_topk_bins // args.axis_norm_bins  # (B, 5)
        pred_bins = torch.stack(
            (pred_x_topk_bins, pred_y_topk_bins, pred_z_topk_bins), dim=-1
        )  # (B, 5, 3)
        pred_bins = (pred_bins.float() + 0.5) / args.axis_norm_bins  # (B, 5, 3)
        (
            min_box_center_axis_norm,  # (B, 3)
            max_box_center_axis_norm,  # (B, 3)
        ) = (
            batch["min_box_center_before_axis_norm"],
            batch["max_box_center_before_axis_norm"],
        )  # all range from [-1, 1]
        pred_topk_xyz = (
            min_box_center_axis_norm[:, None]
            + (max_box_center_axis_norm - min_box_center_axis_norm)[:, None] * pred_bins
        )  # (B, 5, 3)
        pred_radius = pred_radius.unsqueeze(-1).permute(0, 2, 1).repeat(1, 5, 1)  # (B, 5, 1)
        # pred_topk_xyz = torch.cat([pred_topk_xyz, pred_radius], dim=-1)  # (B, 5, 4)

        # Choose this or the next block
        # Choose which object position to visualize
        # The object_idx should between 0 - 4
        object_idx = 0

        vis_pc = last_pcs.squeeze(0)  # (P, 6)

        pos = vis_pc[:, :3]
        aux = vis_pc[:, 3:]

        pred_box_center, pred_box_max_dist = (
            pred_topk_xyz[:, object_idx, :],
            pred_radius[:, object_idx, :],
        )

        # Process the generated point cloud
        coords = pos * pred_box_max_dist + pred_box_center
        colors = aux.clamp(0, 255).round()  # (P, 3 or 4)
        vis_pc = torch.cat((coords, colors), dim=-1)  # (P, 6)
        vis_pc = vis_pc.unsqueeze(0)  # (1, P, 6)
        vis_pc = vis_pc.cpu().numpy()

        # Compute the EMD
        if batch["tgt_class"].item() not in avg_emd_of_classes:
            avg_emd_of_classes[batch["tgt_class"].item()] = 0
            num_of_obj_in_classes[batch["tgt_class"].item()] = 0
        avg_emd_of_classes[batch["tgt_class"].item()] += emd_eval(
            coords, object_dict[batch["tgt_class"].item()]
        )
        num_of_obj_in_classes[batch["tgt_class"].item()] += 1

for key, value in avg_emd_of_classes.items():
    avg_emd_of_classes[key] = value / num_of_obj_in_classes[key]

with open("avg_emd_of_classes.pkl", "wb") as f:
    pickle.dump(avg_emd_of_classes, f)
with open("num_of_obj_in_classes.pkl", "wb") as f:
    pickle.dump(num_of_obj_in_classes, f)
