from functools import partial
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from accelerate import Accelerator, DistributedType
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import BatchEncoding, PreTrainedTokenizer

from .in_out.scannet_scan import ScannetScan, ThreeDObject

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import (
    check_segmented_object_order,
    infer_floor_z_coord,
    instance_labels_of_context,
    normalize_pc,
    sample_scan_object,
)


class ReferIt3DDataset(Dataset):
    def __init__(
        self,
        references: pd.DataFrame,
        scans: Dict[str, ScannetScan],
        max_seq_len: int,
        points_per_object: int,
        max_context_objects: int,
        class_to_idx: Dict[str, int],
        tokenizer: PreTrainedTokenizer,
        object_transformation=None,
        height_append: bool = True,
        fps: bool = False,
        max_fps_candidates: Optional[int] = None,
        random_rotation: bool = False,
        axis_norm: bool = False,
    ):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_context_objects = max_context_objects
        self.class_to_idx = class_to_idx
        self.tokenizer = tokenizer
        self.height_append = height_append
        self.fps = fps
        self.max_fps_candidates = max_fps_candidates
        self.random_rotation = random_rotation
        self.axis_norm = axis_norm
        self.object_transformation = object_transformation
        assert check_segmented_object_order(scans), "Objects are not ordered by object id"

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index: int):
        ref = self.references.iloc[index]
        scan_id: str = ref["scan_id"]
        stimulus_id: str = ref["stimulus_id"]
        scan = self.scans[ref["scan_id"]]
        target: ThreeDObject = scan.three_d_objects[ref["target_id"]]
        text: str = ref["utterance_generative"]
        is_nr3d: bool = ref["dataset"] == "nr3d"

        return scan, target, text, is_nr3d, scan_id, stimulus_id

    def prepare_context_objects(self, scan: ScannetScan, target: ThreeDObject):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        context_objects = [
            o for o in scan.three_d_objects if (o.instance_label == target_label and (o != target))
        ]

        # Then all more objects up to max-number of context objects
        clutter = [o for o in scan.three_d_objects if o.instance_label != target_label]

        context_objects.extend(clutter)
        context_objects = context_objects[: self.max_context_objects]
        np.random.shuffle(context_objects)

        return context_objects

    def __getitem__(self, index: int):
        res = dict()
        scan, target, text, is_nr3d, scan_id, stimulus_id = self.get_reference_data(index)
        # Make a context of background objects
        context = self.prepare_context_objects(scan, target)
        assert len(context) <= self.max_context_objects
        context_w_tgt = [target] + context

        # sample point/color for them
        samples_w_tgt = np.array(
            [
                sample_scan_object(
                    o,
                    self.points_per_object,
                    use_fps=self.fps,
                    max_fps_candidates=self.max_fps_candidates,
                )
                for o in context_w_tgt
            ]
        )  # (# of objects, # of points, 6)
        # 7 when we append height to the point cloud
        if self.height_append:
            floor_z = infer_floor_z_coord(scan)
            height = samples_w_tgt[:, :, 2] - floor_z
            samples_w_tgt = np.concatenate(
                [samples_w_tgt, height[:, :, None]], axis=2
            )  # (# of objects, # of points, 7)

        # the center point of bbox in the scene coord system
        tgt_box_center_w_tgt = np.array(
            [o.get_bbox().center() for o in context_w_tgt]
        )  # (# of objects, 3)

        # random ratation
        if self.random_rotation:
            deg = 90 * np.random.randint(4)
            rot_mat = (R.from_euler("z", deg, degrees=True).as_matrix().transpose()).astype(
                np.float32
            )
            samples_w_tgt[:, :, :3] = samples_w_tgt[:, :, :3] @ rot_mat
            tgt_box_center_w_tgt = tgt_box_center_w_tgt @ rot_mat
        else:
            rot_mat = np.eye(3, dtype=np.float32)

        if self.axis_norm:
            # x,y,z axis norm for box center
            # find each min/max of x,y,z axis first
            min_xyz = samples_w_tgt[..., :3].reshape(-1, 3).min(axis=0, keepdims=True)
            max_xyz = samples_w_tgt[..., :3].reshape(-1, 3).max(axis=0, keepdims=True)
            # min_xyz = tgt_box_center_w_tgt.min(axis=0, keepdims=True)  # (1, 3)
            # max_xyz = tgt_box_center_w_tgt.max(axis=0, keepdims=True)  # (1, 3)
            edge_len_xyz = max_xyz - min_xyz  # (1, 3)
            tgt_box_center_w_tgt_axis_norm = (
                tgt_box_center_w_tgt - min_xyz
            ) / edge_len_xyz  # (# of objects, 3)
            # scale from [0, 1] to [-1, 1]
            tgt_box_center_w_tgt_axis_norm = tgt_box_center_w_tgt_axis_norm * 2 - 1

        # the max dist from the center point to the farthest point in the bbox
        tgt_box_max_dist_w_tgt = np.empty((len(context_w_tgt),))  # (# of objects,)
        for i, o in enumerate(context_w_tgt):
            tmp = o.pc @ rot_mat - tgt_box_center_w_tgt[i][None, :]  # (# of points, 3)
            tgt_box_max_dist_w_tgt[i] = np.linalg.norm(tmp, axis=1, ord=2).max()

        if self.object_transformation is not None:
            samples_w_tgt = self.object_transformation(
                samples_w_tgt, box_center=tgt_box_center_w_tgt, box_max_dist=tgt_box_max_dist_w_tgt
            )

        res["scan_id"] = scan_id
        res["stimulus_id"] = stimulus_id
        # text
        res["text"] = text
        res["tokens"]: BatchEncoding = self.tokenizer(
            text, max_length=self.max_seq_len, truncation=True, padding=False
        )

        # context
        # NOTE: take care of padding, so that a batch has same # of objects across scans.
        res["ctx_key_padding_mask"] = np.array(
            [False] * len(context) + [True] * (self.max_context_objects - len(context)), dtype=bool
        )
        res["ctx_class"] = instance_labels_of_context(
            context,
            self.max_context_objects,
            self.class_to_idx,
            add_padding=True,
        )  # (# of objects,)
        if len(context) < self.max_context_objects:
            # padding
            res["ctx_pc"] = np.pad(
                samples_w_tgt[1:],
                ((0, self.max_context_objects - len(context)), (0, 0), (0, 0)),
                constant_values=0,
            )  # (# of objects, # of points, 6 or 7)
            res["ctx_box_center"] = np.pad(
                tgt_box_center_w_tgt[1:],
                ((0, self.max_context_objects - len(context)), (0, 0)),
                constant_values=0,
            )  # (# of objects, 3)
            res["ctx_box_max_dist"] = np.pad(
                tgt_box_max_dist_w_tgt[1:],
                (0, self.max_context_objects - len(context)),
                constant_values=0,
            )  # (# of objects,)
        else:
            res["ctx_pc"] = samples_w_tgt[1:]
            res["ctx_box_center"] = tgt_box_center_w_tgt[1:]
            res["ctx_box_max_dist"] = tgt_box_max_dist_w_tgt[1:]

        # target
        res["tgt_pc"] = samples_w_tgt[0]  # (# of points, 6 or 7)
        res["tgt_class"] = self.class_to_idx[target.instance_label]  # scalar
        res["tgt_box_center"] = tgt_box_center_w_tgt[0]  # (3,)
        res["tgt_box_max_dist"] = tgt_box_max_dist_w_tgt[0]  # scalar

        if self.axis_norm:
            res["min_box_center_before_axis_norm"] = min_xyz[0]  # (3,)
            res["max_box_center_before_axis_norm"] = max_xyz[0]  # (3,)
            res["ctx_box_center_axis_norm"] = np.pad(
                tgt_box_center_w_tgt_axis_norm[1:],
                ((0, self.max_context_objects - len(context)), (0, 0)),
                constant_values=0,
            )  # (# of objects, # of points, 6 or 7)
            res["tgt_box_center_axis_norm"] = tgt_box_center_w_tgt_axis_norm[0]  # (3,)

        """
        Data format in batch:
        {
            "scan_id": List[str],
            "stimulus_id": List[str],
            ---
            "text": List[str],
            "tokens": BatchEncoding,    # see `custom_collate_fn` below for details
            ---
            "ctx_key_padding_mask": BoolTensor, (B, # of context),
            "ctx_class": LongTensor, (B, # of context),
            "ctx_pc": FloatTensor, (B, # of context, P, 6 or 7), # color range as [0, 1]
            "ctx_box_center": FloatTensor, (B, # of context, 3),
            "ctx_box_max_dist": FloatTensor, (B, # of context),
            ---
            "tgt_pc": FloatTensor, (B, P, 6 or 7),
            "tgt_class": LongTensor, (B,),
            "tgt_box_center": FloatTensor, (B, 3),
            "tgt_box_max_dist": FloatTensor, (B,),
            --- optional(axis-norm) ---
            "min_box_center_before_axis_norm": FloatTensor, (B, 3),
            "max_box_center_before_axis_norm": FloatTensor, (B, 3),
            "ctx_box_center_axis_norm": FloatTensor, (B, # of context, 3),
            "tgt_box_center_axis_norm": FloatTensor, (B, 3),
        }
        """

        return res


def make_data_loaders(
    args,
    accelerator: Accelerator,
    referit_data: pd.DataFrame,
    class_to_idx: Dict[str, int],
    scans: Dict[str, ScannetScan],
    mean_rgb: np.ndarray,
    tokenizer: PreTrainedTokenizer,
):
    data_loaders: Dict[Literal["train", "test", "test_small"], DataLoader] = dict()

    def custom_collate_fn(batch):
        """
        Hook for customizing the way datasets are merged.
        """
        # leave "tokens" alone
        tokens = [b.pop("tokens") for b in batch]
        batch = default_collate(batch)

        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        else:
            pad_to_multiple_of = 8

        batch["tokens"] = tokenizer.pad(
            tokens,
            padding="max_length",
            max_length=args.max_seq_len,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return batch

    for split in ("train", "test", "test_small"):
        mask = referit_data[f"is_{split}"]

        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        is_training = split == "train"

        object_transformation = partial(
            normalize_pc, mean_rgb=None, unit_norm=args.unit_sphere_norm
        )

        dataset = ReferIt3DDataset(
            references=d_set,
            scans=scans,
            max_seq_len=args.max_seq_len,
            points_per_object=args.points_per_object,
            max_context_objects=args.max_context_objects,
            class_to_idx=class_to_idx,
            tokenizer=tokenizer,
            object_transformation=object_transformation,
            height_append=args.height_append,
            fps=args.fps,
            max_fps_candidates=args.max_fps_candidates,
            random_rotation=is_training and args.random_rotation,
            axis_norm=args.axis_norm,
        )

        data_loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            shuffle=is_training,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
        )

    return data_loaders
