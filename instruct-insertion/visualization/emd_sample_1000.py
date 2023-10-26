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
from openpoints.cpp.emd.emd import earth_mover_distance

accelerator = accelerate.Accelerator()
device = accelerator.device

import pickle

with open("object_dict_testset.pkl", "rb") as f:
    object_dict = pickle.load(f)
# {key: (B, P, 3)}
all_obj_cls = []
for key, value in object_dict.items():
    for _ in range(value.shape[0]):
        all_obj_cls.append(key)
all_objects = torch.cat(list(object_dict.values()), dim=0)

# load existing args
PROJECT_TOP_DIR = "../../tmp_link_saves"
PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_rr4")
CHECKPOINT_DIR = osp.join(
    PROJECT_DIR,
    "checkpoints",
    "2023-09-21_18-18-07",
    "best-test_rf3d_loc_estimate_with_top_k_dist-1.1188-step-120000",
)
# PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps")
# CHECKPOINT_DIR = osp.join(PROJECT_DIR, "checkpoints", "2023-10-12_15-42-52", "best-test_rf3d_loc_estimate_dist-1.6379-step-96000")
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

import pickle

from EMD_evaluation.emd_module import emd_eval

idx_has_been_used = []
one_nn = {}
one_nna = {}
emd = {}
num_of_obj_in_classes = {}
cls_top1_correct_for_each_class = {}
cls_top5_correct_for_each_class = {}

emd = earth_mover_distance()

max_len = min(len(data_loaders["test"]), 4000)
it = iter(data_loaders["test"])
import tqdm

for _ in tqdm.tqdm(range(max_len)):
    batch = next(it)
    from scripts.train_utils import move_batch_to_device_

    # get batch
    batch = move_batch_to_device_(batch, device)
    batch4mvt = batch.copy()
    batch4mvt.pop("scan_id")
    batch4mvt.pop("stimulus_id")
    batch4mvt.pop("text")
    with torch.no_grad():
        ctx_embeds, LOSS, CLASS_LOGITS, LANG_LOGITS, LOCATE_PREDS, pred_xyz = mvt3dvg(batch4mvt)

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
        pos = last_pcs[:, :3, :]
        aux = last_pcs[:, 3:, :]
        aux = aux.clamp_(0, 255).round_().div_(255.0)

        last_pcs = torch.cat((pos, aux), dim=1)
        last_pcs = last_pcs.permute(0, 2, 1)  # (B, P, 6)
        coords = last_pcs[:, :, :3]

        # Compute the EMD
        for i in range(batch["tgt_class"].shape[0]):
            ele = batch["tgt_class"][i].item()
            if ele not in one_nn:
                one_nn[ele] = []
                one_nna[ele] = []
                emd[ele] = []
                num_of_obj_in_classes[ele] = 0

            # Compute the 1-nn between the object and all objects of its class
            one_nn_tmp = emd.forward(
                coords[i : i + 1].repeat(object_dict[ele].shape[0], 1, 1).contiguous(),
                object_dict[ele],
            )  # (1, len(object_dict))
            one_nn_tmp_v = one_nn_tmp.min()
            one_nn[ele].append(one_nn_tmp_v)

            # Compute the emd between the object and all objects of its class
            emd[ele].append(one_nn_tmp)
            num_of_obj_in_classes[ele] += len(object_dict[ele])

            # one_nna_tmp = torch.zeros(all_objects.shape[0], device=device)
            # for j in range(all_objects.shape[0]):
            #     one_nna_tmp[j] = (emd.forward(coords[i:i+1].contiguous(), all_objects[j:j+1]))
            # cloest_class = all_obj_cls[one_nna_tmp.argmin()]
            # one_nna[ele].append(cloest_class)

        # # Compute the cls
        # # NOTE - produce the `height append`
        # diff_pcs = last_pcs  # (B, P, 6)
        # if args.height_append:
        #     tgt_pc_height = batch["tgt_pc"][:, :, -1:]  # (B, P, 1)
        #     # avg height
        #     tgt_pc_height = tgt_pc_height.mean(dim=1, keepdim=True).repeat(
        #         1, diff_pcs.shape[1], 1
        #     )  # (B, P, 1)
        #     diff_pcs = torch.cat((diff_pcs, tgt_pc_height), dim=-1)  # (B, P, D=7)
        # _, TGT_CLASS_LOGITS = mvt3dvg.forward_obj_cls(diff_pcs[:, None, :, :])
        # # record the accuaracy
        # for i in range(batch["tgt_class"].shape[0]):
        #     predictions = np.array(TGT_CLASS_LOGITS[i].topk(5)[1].flatten().cpu())
        #     ele = batch["tgt_class"][i].item()
        #     if ele not in cls_top1_correct_for_each_class:
        #         cls_top1_correct_for_each_class[ele] = 0
        #         cls_top5_correct_for_each_class[ele] = 0
        #     if predictions[0] == ele:
        #         cls_top1_correct_for_each_class[ele] += 1
        #     if ele in predictions:
        #         cls_top5_correct_for_each_class[ele] += 1
print("Done!")

# create new folder to store the results
tgt_folder = PROJECT_DIR.split("/")[-1]
if not os.path.exists(PROJECT_DIR.split("/")[-1]):
    os.mkdir(PROJECT_DIR.split("/")[-1])

# with open(tgt_folder + "/avg_emd_of_classes.pkl", "wb") as f:
#     pickle.dump(one_nn, f)
# with open(tgt_folder + "/num_of_obj_in_classes.pkl", "wb") as f:
#     pickle.dump(num_of_obj_in_classes, f)
# with open(tgt_folder + "/cls_top1_correct.pkl", "wb") as f:
#     pickle.dump(cls_top1_correct_for_each_class, f)
# with open(tgt_folder + "/cls_top5_correct.pkl", "wb") as f:
#     pickle.dump(cls_top5_correct_for_each_class, f)
all_data = {}
all_data["one_nn"] = one_nn
# all_data["one_nna"] = one_nna
all_data["mmd_emd"] = emd
all_data["num_of_obj_in_classes"] = num_of_obj_in_classes
with open("all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)

import im_remind

im_remind.send_tg_msg("Done! Please check")
