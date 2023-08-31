from functools import partial
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .in_out.scannet_scan import ScannetScan, ThreeDObject

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import (
    check_segmented_object_order,
    dataset_to_dataloader,
    decode_stimulus_string,
    infer_floor_z_coord,
    instance_labels_of_context,
    max_io_workers,
    mean_rgb_unit_norm_transform,
    pad_samples,
    sample_scan_object,
)


class ReferIt3DDataset(Dataset):
    def __init__(
        self,
        references: pd.DataFrame,
        scans: Dict[str, ScannetScan],
        max_seq_len: int,
        points_per_object: int,
        max_distractors: int,
        class_to_idx: Dict[str, int],
        object_transformation=None,
        height_append: bool = True,
        visualization: bool = False,
    ):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.height_append = height_append
        self.visualization = visualization
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index: int):
        ref = self.references.iloc[index]
        scan_id: str = ref["scan_id"]
        scan = self.scans[ref["scan_id"]]
        target: ThreeDObject = scan.three_d_objects[ref["target_id"]]
        text: str = ref["utterance_generative"]
        is_nr3d: bool = ref["dataset"] == "nr3d"

        return scan, target, text, is_nr3d, scan_id

    def prepare_distractors(self, scan: ScannetScan, target: ThreeDObject):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [
            o for o in scan.three_d_objects if (o.instance_label == target_label and (o != target))
        ]

        # Then all more objects up to max-number of distractors
        clutter = [o for o in scan.three_d_objects if o.instance_label != target_label]

        distractors.extend(clutter)
        distractors = distractors[: self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index: int):
        res = dict()
        scan, target, text, is_nr3d, scan_id = self.get_reference_data(index)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array(
            [sample_scan_object(o, self.points_per_object) for o in context]
        )  # (# of objects, # of points, 6)
        # 7 when we append height to the point cloud
        if self.height_append:
            floor_z = infer_floor_z_coord(scan)
            height = samples[:, :, 2] - floor_z
            samples = np.concatenate(
                [samples, height[:, :, None]], axis=2
            )  # (# of objects, # of points, 7)

        # mark their classes
        res["class_labels"] = instance_labels_of_context(
            context, self.max_context_size, self.class_to_idx
        )
        res["scan_id"] = scan_id
        # the center point of bbox in the scene coord system
        box_center = np.array([o.get_bbox().center() for o in context])
        box_z_len = np.array([o.get_bbox().lz for o in context])

        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res["context_size"] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res["objects"] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=bool)
        target_class_mask[: len(context)] = [
            target.instance_label == o.instance_label for o in context
        ]

        res["target_class"] = self.class_to_idx[target.instance_label]
        res["target_pos"] = target_pos
        res["target_class_mask"] = target_class_mask
        res["text"] = text
        res["is_nr3d"] = is_nr3d
        res["box_center"] = box_center
        res["box_z_len"] = box_z_len

        if self.visualization:
            # 6 is the maximum context size we used in dataset collection
            distrators_pos = np.zeros((6,))
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res["utterance"] = self.references.iloc[index]["utterance"]
            res["stimulus_id"] = self.references.iloc[index]["stimulus_id"]
            res["distrators_pos"] = distrators_pos
            res["object_ids"] = object_ids
            res["target_object_id"] = target.object_id

        return res


def make_data_loaders(
    args,
    referit_data: pd.DataFrame,
    class_to_idx: Dict[str, int],
    scans: Dict[str, ScannetScan],
    mean_rgb: np.ndarray,
):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders: Dict[Literal["train", "test"], DataLoader] = dict()
    is_train = referit_data["is_train"]
    splits = ("train", "test")

    object_transformation = partial(
        mean_rgb_unit_norm_transform, mean_rgb=mean_rgb, unit_norm=args.unit_sphere_norm
    )

    for split in splits:
        mask = is_train if split == "train" else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == "train" else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == "test":

            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print(
                "length of dataset before removing non multiple test utterances {}".format(
                    len(d_set)
                )
            )
            print(
                "removed {} utterances from the test set that don't have multiple distractors".format(
                    np.sum(~multiple_targets_mask)
                )
            )
            print(
                "length of dataset after removing non multiple test utterances {}".format(
                    len(d_set)
                )
            )

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ReferIt3DDataset(
            references=d_set,
            scans=scans,
            max_seq_len=args.max_seq_len,
            points_per_object=args.points_per_object,
            max_distractors=max_distractors,
            class_to_idx=class_to_idx,
            object_transformation=object_transformation,
            visualization=args.mode == "evaluate",
        )

        seed = None
        if split == "test":
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(
            dataset, split, args.batch_size, n_workers, seed=seed
        )

    return data_loaders
