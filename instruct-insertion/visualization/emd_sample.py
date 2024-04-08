import json
import os
import os.path as osp
import pickle
import sys
from typing import Literal

import accelerate
import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict
from transformers import BertTokenizer

sys.path.append(os.path.join(os.getcwd(), ".."))
from data.referit3d.in_out.neural_net_oriented import (
    compute_auxiliary_data,
    load_referential_data,
    load_scan_related_data,
    trim_scans_per_referit3d_data_,
)
from openpoints.cpp.emd.emd import earth_mover_distance
from scripts.train_utils import move_batch_to_device_

accelerator = accelerate.Accelerator()
device = accelerator.device

################################
#                              #
#    Data & Hyperparameters    #
#                              #
################################
# load existing args
PROJECT_TOP_DIR = "../../tmp_link_saves"
# NOTE: Modify here.
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_rr4_sr3d")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-09-21_18-18-07",
#     "ckpt_800000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_rr4")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-09-18_14-52-06",
#     "ckpt_160000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-09-16_16-54-54",
#     "ckpt_160000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_16bin")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-10-22_15-38-56",
#     "ckpt_160000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-10-12_15-42-52",
#     "ckpt_160000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "baseline")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-10-21_14-18-59",
#     "ckpt_160000",
# )
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "point_e_only")
# CHECKPOINT_DIR = osp.join(
#     PROJECT_DIR,
#     "checkpoints",
#     "2023-10-25_16-29-07",
# "ckpt_160000",
# )

with open(osp.join(PROJECT_DIR, "config.json.txt"), "r") as f:
    args = edict(json.load(f))

# load data
# NOTE: Modify here.
SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global.pkl"
# SCANNET_PKL_FILE = "../../datasets/scannet/instruct/global_small.pkl"
REFERIT_CSV_FILE = "../../datasets/nr3d/nr3d_generative_20230825_final.csv"
all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)
referit_data = load_referential_data(args, args.referit3D_file, scans_split)
# Prepare data & compute auxiliary meta-information.
all_scans_in_dict = trim_scans_per_referit3d_data_(referit_data, all_scans_in_dict)
mean_rgb = compute_auxiliary_data(referit_data, all_scans_in_dict)

MAX_SAMPLE_LEN = 4000  # Max number of samples to generate.
DATASET_TYPE: Literal["train", "test"] = "test"  # "train" or "test"
TOPK = 5  # Top-k for axis_norm model

###############
#             #
#    Model    #
#             #
###############
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
# if args.mode == "train":
mvt3dvg = torch.compile(mvt3dvg)
mvt3dvg, point_e = accelerator.prepare(mvt3dvg, point_e)
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


####################
#                  #
#    Statistics    #
#                  #
####################
idx_has_been_used = []
one_nn = {}
one_nna = {}
emd_dis = {}
num_of_obj_in_classes = {}
cls_top1_correct_for_each_class = {}
cls_top5_correct_for_each_class = {}

emd = earth_mover_distance()

max_len = min(len(data_loaders[DATASET_TYPE]), MAX_SAMPLE_LEN)
it = iter(data_loaders[DATASET_TYPE])
# Reverse to get idx_to_class
idx_to_class = {v: k for k, v in class_to_idx.items()}

objs = []
for _ in tqdm.tqdm(range(max_len)):
    batch = next(it)

    # get batch
    batch = move_batch_to_device_(batch, device)
    B, N, C = len(batch["text"]), batch["ctx_pc"].shape[1], args.inner_dim
    batch4mvt = batch.copy()
    batch4mvt.pop("scan_id")
    batch4mvt.pop("stimulus_id")
    batch4mvt.pop("text")
    with torch.no_grad():
        if not args.point_e_only:
            ctx_embeds, LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS, pred_xyz = mvt3dvg(batch4mvt)
        else:
            ctx_embeds = torch.zeros((B, C), device=device)
            RF3D_LOSS = torch.zeros(1, device=device)
            CLASS_LOGITS = torch.zeros((B, N, n_classes), device=device)
            LANG_LOGITS = torch.zeros((B, C), device=device)
            LOCATE_PREDS = torch.zeros((B, 4), device=device)
            pred_xyz = None
        # stack twice for guided scale
        ctx_embeds = torch.cat((ctx_embeds, ctx_embeds), dim=0)

        prompts = batch["text"]

        # obj = {"prompt":"",
        #        "objs": [],
        #        "ref": numpy,
        #        "stimulus_id": str,
        #        "class": int,
        #        "class_str": str,
        #        "pred_xyz_n": numpy,
        #        "pred_xyz": numpy,
        #        "radius": numpy}

        generated_objs = []
        for i in range(1):
            samples_it = sampler.sample_batch_progressive(
                batch_size=B,
                ctx_embeds=ctx_embeds,
                model_kwargs=dict(texts=prompts),
                accelerator=accelerator,
            )
            # get the last timestep prediction
            for last_pcs in samples_it:
                pass
            pos = last_pcs[:, :3, :]
            aux = last_pcs[:, 3:, :]
            aux = aux.clamp_(0, 255).round_().div_(255.0)

            last_pcs = torch.cat((pos, aux), dim=1)
            last_pcs = last_pcs.permute(0, 2, 1)  # (B, P, 6)
            # coords = last_pcs[:, :, :3]
            generated_objs.append(last_pcs.cpu().numpy())
        P = last_pcs.shape[1]

        # For axis_norm model
        if args.axis_norm:
            pred_xy, pred_z, pred_radius = pred_xyz
            pred_xy_topk_bins = pred_xy.topk(TOPK, dim=-1)[1]  # (B, TOPK)
            # pred_z_topk_bins = pred_z.topk(TOPK, dim=-1)[1]  # (B, TOPK)
            pred_z_topk_bins = pred_z.argmax(dim=-1, keepdim=True).repeat(1, TOPK)  # (B, TOPK)
            pred_x_topk_bins = pred_xy_topk_bins % args.axis_norm_bins  # (B, TOPK)
            pred_y_topk_bins = pred_xy_topk_bins // args.axis_norm_bins  # (B, TOPK)
            pred_bins = torch.stack(
                (pred_x_topk_bins, pred_y_topk_bins, pred_z_topk_bins), dim=-1
            )  # (B, TOPK, 3)
            pred_bins = (pred_bins.float() + 0.5) / args.axis_norm_bins  # (B, TOPK, 3)
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
            )  # (B, TOPK, 3)

            # print(P)
            # pred_radius = pred_radius.unsqueeze(-1).permute(0, 2, 1).repeat(1, 5, 1)  # (B, 5, 1)
            pred_radius = pred_radius[:, None, :].repeat(1, P, 1)  # (B, P, 1)
            # pred_topk_xyz = torch.cat([pred_topk_xyz, pred_radius], dim=-1)  # (B, 5, 4)

            pred_xyz_real = last_pcs[:, :, :3] * pred_radius  # (B, P, 3)
            pred_xyz_real = pred_xyz_real[:, None, :, :].repeat(1, TOPK, 1, 1) + pred_topk_xyz[
                :, :, None, :
            ].repeat(
                1, 1, P, 1
            )  # (B, 5, P, 3)
            # print(pred_xyz_real.shape)
            # print(len(generated_objs))
        else:
            # FIXME: DONT USE THIS. Already deprecated.
            raise RuntimeError("Deprecated code.")
            # replace last_pcs with the real point cloud
            pred_box_center, pred_box_max_dist = LOCATE_PREDS[:, :3], LOCATE_PREDS[:, 3:4]
            # print(pred_box_max_dist.shape)
            pred_radius = pred_box_max_dist[:, None, :].repeat(1, P, 1)  # (B, P, 1)

            # Process the generated point cloud
            coords = last_pcs[:, :, :3] * pred_radius
            coords = coords + pred_box_center[:, None, :].repeat(1, P, 1)  # (B, P, 3)
            pred_topk_xyz = coords
            pred_xyz_real = coords

        # put data into obj
        # objs_tmp = []
        for j in range(batch["tgt_class"].shape[0]):
            obj = {}
            obj["prompt"] = batch["text"][j]
            obj["objs"] = []
            for generated_obj in generated_objs:
                # objs_tmp.append(generated_obj[j])
                obj["objs"].append(generated_obj[j])
            obj["ref"] = batch["tgt_pc"][j].cpu().numpy()
            obj["stimulus_id"] = batch["stimulus_id"][j]
            obj["class"] = batch["tgt_class"][j].item()
            obj["class_str"] = idx_to_class[obj["class"]]
            obj["pred_xyz_raw"] = pred_topk_xyz[j].cpu().numpy()
            obj["radius"] = pred_radius[j].cpu().numpy()
            obj["pred_xyz"] = pred_xyz_real[j].cpu().numpy()
            objs.append(obj)
print("Done!")

################
#              #
#    Saving    #
#              #
################
# create new folder to store the results
with open(osp.join(PROJECT_DIR, "objs.pkl"), "wb") as f:
    pickle.dump(objs, f)
print("Save success!")

try:
    import im_remind

    im_remind.send_tg_msg("Done! Please check the results.")
except ImportError:
    pass
