import argparse
import os
import os.path as osp
import pickle
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

sys.path.append(f"{osp.dirname(__file__)}/..")

from openpoints.cpp.emd.emd import EarthMoverDistanceFunction

# obj_pkl_data := [{
#   "prompt":"",
#   "objs": list of numpy of shape [P, 6],
#   "ref": numpy of shape [P, 6],
#   "stimulus_id": str,
#   "class": int,
#   "class_str": str,
# }]
PROJECT_TOP_DIR = f"{osp.dirname(__file__)}/../../tmp_link_saves"
PROJECT_DIR = osp.join(PROJECT_TOP_DIR, "fps_axisnorm_rr4_sr3d")
CHECKPOINT_DIR = osp.join(
    PROJECT_DIR,
    "checkpoints",
    "2023-09-21_18-18-07",
    "ckpt_800000",
)
SCANNET_PKL_FILE = f"{osp.dirname(__file__)}/../../datasets/scannet/instruct/global.pkl"


@torch.no_grad()
def one2many_emd(one: torch.Tensor, many: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Args:
        one: (P, 6)
        many: (N, P, 6)

    Returns:
        (N,)
    """
    one = one.unsqueeze(0)  # (1, P, 6)
    ret = []
    for sidx in range(0, many.shape[0], batch_size):
        eidx = min(sidx + batch_size, many.shape[0])
        many_batch = many[sidx:eidx].contiguous()
        one_batch = one.repeat(eidx - sidx, 1, 1).contiguous()
        emd = EarthMoverDistanceFunction.apply(many_batch, one_batch)  # (batch_size,)
        emd /= one.size(1)
        ret.append(emd)
    ret = torch.cat(ret, dim=0)
    return ret


@torch.no_grad()
def precompute_emds(
    obj_pkl_data: dict[str, object],
    class2idx: dict[str, int],
    sample_idx: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Precompute the EMDs for each object in the dataset.

    Returns:
        dict[class_str, torch.Tensor]: (objs_len + refs_len, refs_len + objs_len)
    """
    results: dict[str, float] = {}
    if not torch.cuda.is_available():
        raise RuntimeError("EMD computation requires CUDA.")
    device = torch.device("cuda")
    with tqdm(total=len(obj_pkl_data) * 2, desc="PRE-EMDS") as pbar:
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, refs = [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    refs.append(datum["ref"])
            objs: np.ndarray = np.stack(objs, axis=0)[..., :3]  # (*, P, 3)
            refs: np.ndarray = np.stack(refs, axis=0)[..., :3]  # (*, P, 3)
            objs_len = len(objs)
            refs_len = len(refs)
            # move to torch
            objs: torch.Tensor = torch.from_numpy(objs).to(device=device)
            refs: torch.Tensor = torch.from_numpy(refs).to(device=device)
            objs_refs = torch.cat((objs, refs), dim=0).contiguous()  # (objs_len + refs_len, P, 3)
            refs_objs = torch.cat((refs, objs), dim=0).contiguous()  # (refs_len + objs_len, P, 3)

            t = torch.zeros((objs_len + refs_len, refs_len + objs_len), device=device)
            for obj_ref_idx, item in enumerate(objs_refs):
                # obj: (P, 6)
                emd = one2many_emd(item, refs_objs, batch_size)
                t[obj_ref_idx] = emd
                pbar.update()

            results[class_str] = t
    return results


@torch.no_grad()
def compute_mmd_emd(
    obj_pkl_data: dict[str, object],
    class2idx: dict[str, int],
    sample_idx: int,
    batch_size: int,
    precomputed_emds: dict[str, torch.Tensor] = None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not torch.cuda.is_available():
        raise RuntimeError("EMD computation requires CUDA.")
    device = torch.device("cuda")
    with tqdm(total=len(obj_pkl_data), desc="MMD-EMD") as pbar:
        # iterate each class
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, refs = [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    refs.append(datum["ref"])
            objs: np.ndarray = np.stack(objs, axis=0)[..., :3]  # (*, P, 3)
            refs: np.ndarray = np.stack(refs, axis=0)[..., :3]  # (*, P, 3)

            # move to torch
            objs: torch.Tensor = torch.from_numpy(objs).to(device=device).contiguous()
            refs: torch.Tensor = torch.from_numpy(refs).to(device=device).contiguous()

            # compute the MMD-EMD
            mmd_sum = torch.zeros((), device=device)
            for ref_idx, ref in enumerate(refs):
                # ref: (P, 6)
                if precomputed_emds is None:
                    mmd = one2many_emd(ref, objs, batch_size)  # (N,)
                else:
                    mmd = precomputed_emds[class_str][: len(objs), ref_idx]  # (N,)
                mmd_sum += mmd.min()
                pbar.update()

            mmd = mmd_sum / len(refs)
            results[class_str] = mmd.item()

    return results


@torch.no_grad()
def compute_cov_emd(
    obj_pkl_data: dict[str, object],
    class2idx: dict[str, int],
    sample_idx: int,
    batch_size: int,
    precomputed_emds: dict[str, torch.Tensor] = None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not torch.cuda.is_available():
        raise RuntimeError("EMD computation requires CUDA.")
    device = torch.device("cuda")
    total = sum(len(datum["objs"]) for datum in obj_pkl_data)
    with tqdm(total=total, desc="COV-EMD") as pbar:
        # iterate each class
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, refs = [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    refs.append(datum["ref"])
            objs: np.ndarray = np.stack(objs, axis=0)[..., :3]  # (*, P, 3)
            refs: np.ndarray = np.stack(refs, axis=0)[..., :3]  # (*, P, 3)

            # move to torch
            objs: torch.Tensor = torch.from_numpy(objs).to(device=device).contiguous()
            refs: torch.Tensor = torch.from_numpy(refs).to(device=device).contiguous()

            # compute the MMD-EMD
            cov_idx_set = set()
            for obj_idx, obj in enumerate(objs):
                # obj: (P, 6)
                if precomputed_emds is None:
                    cov = one2many_emd(obj, refs, batch_size)  # (N,)
                else:
                    cov = precomputed_emds[class_str][obj_idx, len(refs) :]  # (N,)
                cov_idx_set.add(cov.argmin().item())
                pbar.update()

            results[class_str] = len(cov_idx_set) / len(refs)

    return results


@torch.no_grad()
def compute_1nna_emd(
    obj_pkl_data: dict[str, object],
    class2idx: dict[str, int],
    batch_size: int,
    sample_idx: int,
    precomputed_emds: dict[str, torch.Tensor] = None,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if not torch.cuda.is_available():
        raise RuntimeError("EMD computation requires CUDA.")
    device = torch.device("cuda")
    total = len(obj_pkl_data) * 2
    with tqdm(total=total, desc="1-NNA-EMD") as pbar:
        # iterate each class
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, refs = [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    refs.append(datum["ref"])
            assert len(objs) == len(refs)
            objs: np.ndarray = np.stack(objs, axis=0)[..., :3]  # (*, P, 3)
            refs: np.ndarray = np.stack(refs, axis=0)[..., :3]  # (*, P, 3)

            # move to torch
            objs: torch.Tensor = torch.from_numpy(objs).to(device=device).contiguous()
            refs: torch.Tensor = torch.from_numpy(refs).to(device=device).contiguous()
            alls = torch.cat((objs, refs), dim=0)  # (2*, P, 3)
            half_num = len(objs)

            # compute the 1NNA-EMD
            hit_count = 0
            for all_idx, item in enumerate(alls):
                # item: (P, 6)
                if precomputed_emds is None:
                    emd = one2many_emd(item, alls, batch_size)  # (2*,)
                else:
                    emd = precomputed_emds[class_str][all_idx]  # (2*,)
                nn_idx = emd.topk(2, dim=0, largest=False)[1][1].item()
                if all_idx < half_num and nn_idx < half_num:
                    hit_count += 1
                elif all_idx >= half_num and nn_idx >= half_num:
                    hit_count += 1
                pbar.update()

            results[class_str] = hit_count / len(alls)

    return results


def voxelize_density(pcs: np.ndarray, grid: int = 28) -> np.ndarray:
    """
    Args:
        pcs: (N, P, 3): range from [-1, 1]
        grid (int): granularity of the voxelization

    Returns:
        (grid, grid, grid): density map in XYZ order
    """
    ret = np.zeros((grid, grid, grid), dtype=np.int64)
    for pc in pcs:
        # pc: (P, 3)
        pc = (pc + 1) / 2  # [0, 1]
        pc_q = np.round(pc * grid).astype(np.int64)
        pc_q = np.clip(pc_q, 0, grid - 1)
        for point in pc_q:
            ret[point[0], point[1], point[2]] += 1

    # norm
    ret = ret.astype(np.float64) / ret.sum()
    return ret.astype(np.float32)


def compute_jsd(
    obj_pkl_data: dict[str, object],
    class2idx: dict[str, int],
    sample_idx: int,
) -> dict[str, float]:
    results: dict[str, float] = {}
    with tqdm(total=len(class2idx), desc="JSD") as pbar:
        # iterate each class
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, refs = [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    refs.append(datum["ref"])
            objs: np.ndarray = np.stack(objs, axis=0)[..., :3]  # (*, P, 3)
            refs: np.ndarray = np.stack(refs, axis=0)[..., :3]  # (*, P, 3)

            objs_density = voxelize_density(objs)
            refs_density = voxelize_density(refs)
            jsd_sqrt = jensenshannon(objs_density.flatten(), refs_density.flatten())

            results[class_str] = jsd_sqrt**2
            pbar.update()

    return results


@torch.no_grad()
def compute_acc(
    obj_pkl_data: dict[str, object], class2idx: dict[str, int], sample_idx: int
) -> tuple[dict[str, float], dict[str, float]]:
    import json

    import accelerate
    from data.referit3d.in_out.neural_net_oriented import load_scan_related_data
    from easydict import EasyDict as edict
    from models.point_e_model.models.configs import MODEL_CONFIGS, model_from_config
    from models.referit3d_model.referit3d_net import ReferIt3DNet_transformer
    from transformers import BertTokenizer

    with open(osp.join(PROJECT_DIR, "config.json.txt"), "r") as f:
        args = edict(json.load(f))

    accelerator = accelerate.Accelerator()
    device = accelerator.device

    _, _, class_to_idx = load_scan_related_data(SCANNET_PKL_FILE)
    # prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx["pad"]
    # Object-type classification
    class_name_list = list(class_to_idx.keys())

    class_name_tokens = tokenizer(class_name_list, return_tensors="pt", padding=True)
    class_name_tokens = class_name_tokens.to(device=device)

    # referit3d model
    mvt3dvg = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    # point-e model
    point_e_config = MODEL_CONFIGS[args.point_e_model]
    point_e_config["cache_dir"] = osp.join(PROJECT_TOP_DIR, "cache", "point_e_model")
    point_e_config["n_ctx"] = args.points_per_object
    point_e = model_from_config(point_e_config, device)
    # move models to gpu
    mvt3dvg = mvt3dvg.to(device=device).eval()
    point_e = point_e.to(device=device).eval()
    # load model and checkpoints
    # if args.mode == "train":
    mvt3dvg = torch.compile(mvt3dvg)
    mvt3dvg, point_e = accelerator.prepare(mvt3dvg, point_e)
    accelerator.load_state(CHECKPOINT_DIR)

    acc1_results, acc5_results = {}, {}
    with tqdm(total=len(obj_pkl_data), desc="ACC") as pbar:
        # iterate each class
        for class_str, class_int in class2idx.items():
            # find all objects of this class
            objs, tgt_class, refs = [], [], []
            for datum in obj_pkl_data:
                if datum["class"] == class_int:
                    objs.append(datum["objs"][sample_idx])
                    tgt_class.append(datum["class"])
                    refs.append(datum["ref"])
            objs: np.ndarray = np.stack(objs, axis=0)  # (*, P, 6)
            refs: np.ndarray = np.stack(refs, axis=0)  # (*, P, 7)
            assert objs.shape[-1] == 6
            if args.height_append:
                assert refs.shape[-1] == 7
            else:
                assert refs.shape[-1] == 6
            assert len(objs) == len(tgt_class)

            # move to torch
            objs: torch.Tensor = torch.from_numpy(objs).to(device=device)
            refs: torch.Tensor = torch.from_numpy(refs).to(device=device)
            tgt_class: torch.Tensor = torch.tensor(tgt_class, dtype=torch.long, device=device)

            # compute the ACC
            hit1_count = torch.zeros((), device=device)
            hit5_count = torch.zeros((), device=device)
            for sidx in range(0, objs.shape[0], args.batch_size):
                eidx = min(sidx + args.batch_size, objs.shape[0])
                objs_batch = objs[sidx:eidx]  # (batch_size, P, 6)
                refs_batch = refs[sidx:eidx]  # (batch_size, P, 6 or 7)
                tgt_class_batch = tgt_class[sidx:eidx]  # (batch_size,)
                # height append
                if args.height_append:
                    tgt_pc_height = refs_batch[:, :, -1:]  # (B, P, 1)
                    # avg height
                    tgt_pc_height = tgt_pc_height.mean(dim=1, keepdim=True).repeat(
                        1, objs_batch.shape[1], 1
                    )  # (B, P, 1)
                    objs_batch = torch.cat((objs_batch, tgt_pc_height), dim=-1)  # (B, P, D=7)
                _, preds = mvt3dvg.forward_obj_cls(
                    objs_batch[:, None]
                )  # (batch_size, 1, n_classes)
                preds = preds.squeeze(1)  # (batch_size, n_classes)
                preds_top5 = preds.topk(5, dim=1, largest=True)[1]  # (batch_size, 5)
                # compute Top-1
                hit1_count += (preds_top5[:, 0] == tgt_class_batch).sum()
                # compute Top-5
                hit5_count += (preds_top5 == tgt_class_batch[:, None]).sum()
                pbar.update(eidx - sidx)

            acc1_results[class_str] = hit1_count.item() / objs.shape[0]
            acc5_results[class_str] = hit5_count.item() / objs.shape[0]

    return acc1_results, acc5_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pkl-file", type=str, help="input pickle file")
    parser.add_argument(
        "-s",
        "--sample-idx",
        type=int,
        default=0,
        help="index of samples in generated objs to compute. [Default: 0]",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="batch size for computing metrics. [Default: 32]",
    )
    parser.add_argument("-o", "--output", type=str, default=osp.join(osp.curdir, "results.csv"))
    parser.add_argument(
        "-m", "--metrics", choices=["mmd-emd", "cov-emd", "1-nna-emd", "jsd", "acc"], nargs="+"
    )
    parser.add_argument("--force", action="store_true", help="overwrite the output file")

    args = parser.parse_args()

    if osp.exists(args.output):
        if args.force:
            warnings.warn(f"File {args.output} already exists. Overwriting.")
        else:
            raise RuntimeError(
                f"File {args.output} already exists. Aborting unless add '--force' option."
            )

    if not args.metrics:
        raise argparse.ArgumentError(None, "Please specify at least one metric to '-m'.")

    # read the pkl file
    with open(args.pkl_file, "rb") as f:
        obj_pkl_data = pickle.load(f)
        # obj_pkl_data = obj_pkl_data[:32]

    # statistics on the class types
    class2idx: dict[str, int] = {}
    for obj in obj_pkl_data:
        class_str, class_int = obj["class_str"], obj["class"]
        class2idx[class_str] = class_int
    idx2class = {v: k for k, v in class2idx.items()}

    out: dict[str, dict[str, float]] = {}

    if "1-nna-emd" in args.metrics:
        precomputed_emds = precompute_emds(
            obj_pkl_data,
            class2idx=class2idx,
            sample_idx=args.sample_idx,
            batch_size=args.batch_size,
        )
    else:
        precomputed_emds = None

    for metric in args.metrics:
        try:
            if metric == "mmd-emd":
                out["mmd_emd"] = compute_mmd_emd(
                    obj_pkl_data,
                    class2idx=class2idx,
                    sample_idx=args.sample_idx,
                    batch_size=args.batch_size,
                    precomputed_emds=precomputed_emds,
                )
            elif metric == "cov-emd":
                out["cov_emd"] = compute_cov_emd(
                    obj_pkl_data,
                    class2idx=class2idx,
                    sample_idx=args.sample_idx,
                    batch_size=args.batch_size,
                    precomputed_emds=precomputed_emds,
                )
            elif metric == "1-nna-emd":
                out["one_nna"] = compute_1nna_emd(
                    obj_pkl_data,
                    class2idx=class2idx,
                    sample_idx=args.sample_idx,
                    batch_size=args.batch_size,
                    precomputed_emds=precomputed_emds,
                )
            elif metric == "jsd":
                out["jsd"] = compute_jsd(
                    obj_pkl_data,
                    class2idx=class2idx,
                    sample_idx=args.sample_idx,
                )
            elif metric == "acc":
                out["acc1"], out["acc5"] = compute_acc(obj_pkl_data, sample_idx=args.sample_idx)
        except Exception as e:
            warnings.warn(f"Failed to compute {metric}.")
            traceback.print_exc()

    # write to csv
    os.makedirs(osp.dirname(args.output), exist_ok=True)
    out_pd_dict = {"object class": list(class2idx.keys())}
    for metric, results in out.items():
        assert list(results.keys()) == list(class2idx.keys())
        out_pd_dict[metric] = list(results.values())
    out_pd = pd.DataFrame.from_dict(out_pd_dict)
    out_pd.to_csv(args.output, index=False)
