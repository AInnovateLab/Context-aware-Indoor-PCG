import pickle
from functools import partial
from typing import Literal, Optional

import numpy as np
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer

from ...data_generation.nr3d import decode_stimulus_string
from .converter import Converter

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import (
    check_segmented_object_order,
    dataset_to_dataloader,
    instance_labels_of_context,
    max_io_workers,
    mean_rgb_unit_norm_transform,
    objects_bboxes,
    pad_samples,
    sample_scan_object,
)


class ListeningDataset(Dataset):
    def __init__(
        self,
        references,
        scans,
        vocab,
        max_seq_len,
        points_per_object,
        max_distractors,
        class_to_idx=None,
        object_transformation=None,
        visualization=False,
        hook_type: Literal[False, "sa", "po", "train"] = False,
        hook_type_fn_name: Optional[str] = None,
        hook_data_path="./objs.pkl",
    ):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.hook_data_path = hook_data_path
        self.hook_type = hook_type
        self.hook_type_fn_name = hook_type_fn_name
        # TODO - add necessary args

        # Hook datasets
        self.converter = Converter(
            self.hook_data_path,
            self.hook_type,
            self.hook_type_fn_name,
        )
        if self.converter.need_hook_dataset():
            self.references, self.scans = self.converter.modify_dataset(self.references, self.scans)

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.iloc[index]
        scan_id = ref["scan_id"]
        scan = self.scans[ref["scan_id"]]
        target = scan.three_d_objects[ref["target_id"]]
        # sega_update: 使用原始的token
        # tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)

        # ori_tokens = ref["tokens"]
        # tokens = " ".join(ori_tokens)

        tokens = ref["utterance"]

        # tokens = self.vocab(sen).input_ids
        # print(len(tokens))
        # tokens = np.array(tokens)
        # tokens = np.array([102]*(self.max_seq_len + 2 + self.max_context_size * 2))
        # tokens[:min(self.max_seq_len + 2, len(emb))] = emb[:min(self.max_seq_len + 2, len(emb))]
        is_nr3d = ref["dataset"] == "nr3d"
        stimulus_id: str = ref["stimulus_id"]
        gtext: str = ref["utterance_generative"]

        return scan, target, tokens, is_nr3d, scan_id, stimulus_id, gtext

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [
            o for o in scan.three_d_objects if (o.instance_label == target_label and (o != target))
        ]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[: self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        (
            scan,
            target,
            tokens,
            is_nr3d,
            scan_id,
            stimulus_id,
            gtext,
        ) = self.get_reference_data(index)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        # mark their classes
        # res['ori_labels'],
        res["class_labels"] = instance_labels_of_context(
            context, self.max_context_size, self.class_to_idx
        )
        res["scan_id"] = scan_id
        box_info = np.zeros((self.max_context_size, 4))
        box_info[: len(context), 0] = [o.get_bbox().cx for o in context]
        box_info[: len(context), 1] = [o.get_bbox().cy for o in context]
        box_info[: len(context), 2] = [o.get_bbox().cz for o in context]
        box_info[: len(context), 3] = [o.get_bbox().volume() for o in context]
        # box_corners = np.zeros((self.max_context_size, 8, 3))
        # box_corners[: len(context)] = [o.get_bbox().corners for o in context]

        # NOTE: HOOK here
        if self.converter.need_hook_getitem():
            hook_results = self.converter.hook_getitem(stimulus_id, box_info, samples, target_pos)
            # `hook_xyz` should be Point-clouds in the original coords.
            hook_xyz, hook_rgb = hook_results["hook_xyz"], hook_results["hook_rgb"]
            assert hook_xyz.shape[0] == hook_rgb.shape[0]
            hook_target_pc = np.concatenate([hook_xyz, hook_rgb], axis=-1)  # [P, 6]
            samples[target_pos] = hook_target_pc
            # hook bbox
            box_centroid = (hook_xyz.max(axis=0) + hook_xyz.min(axis=0)) / 2.0  # [3,]
            box_info[target_pos, :3] = box_centroid
            box_info[target_pos, 3] = (hook_xyz.max(axis=0) - hook_xyz.min(axis=0)).prod()

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
        res["tokens"] = tokens
        res["is_nr3d"] = is_nr3d
        res["box_info"] = box_info
        # res["box_corners"] = box_corners

        if self.visualization:
            distrators_pos = np.zeros(
                (6)
            )  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res["utterance"] = self.references.loc[index]["utterance"]
            res["stimulus_id"] = self.references.loc[index]["stimulus_id"]
            res["distrators_pos"] = distrators_pos
            res["object_ids"] = object_ids
            res["target_object_id"] = target.object_id

        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data["is_train"]
    splits = [
        ("train", args.hook_type if args.hook_type == "train" else False),
        ("test", args.hook_type if args.hook_type in ("sa", "po") else False),
    ]

    object_transformation = partial(
        mean_rgb_unit_norm_transform, mean_rgb=mean_rgb, unit_norm=args.unit_sphere_norm
    )

    for split, hook_type in splits:
        mask = is_train if split == "train" else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == "train" else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # # if split == test remove the utterances of unique targets
        # if split == "test":

        #     def multiple_targets_utterance(x):
        #         _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
        #         return len(distractors_ids) > 0

        #     multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
        #     d_set = d_set[multiple_targets_mask]
        #     d_set.reset_index(drop=True, inplace=True)
        #     print(
        #         "length of dataset before removing non multiple test utterances {}".format(
        #             len(d_set)
        #         )
        #     )
        #     print(
        #         "removed {} utterances from the test set that don't have multiple distractors".format(
        #             np.sum(~multiple_targets_mask)
        #         )
        #     )
        #     print(
        #         "length of dataset after removing non multiple test utterances {}".format(
        #             len(d_set)
        #         )
        #     )

        #     assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(
            references=d_set,
            scans=scans,
            vocab=vocab,
            max_seq_len=args.max_seq_len,
            points_per_object=args.points_per_object,
            max_distractors=max_distractors,
            class_to_idx=class_to_idx,
            object_transformation=object_transformation,
            visualization=args.mode == "evaluate",
            hook_data_path=args.hook_data_path,
            hook_type=hook_type,
            hook_type_fn_name=args.hook_type_fn_name,
        )

        seed = None
        if split == "test":
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(
            dataset, split, args.batch_size, n_workers, seed=seed
        )

    return data_loaders
