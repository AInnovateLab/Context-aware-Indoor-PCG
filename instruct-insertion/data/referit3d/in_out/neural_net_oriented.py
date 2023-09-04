import os.path as osp
import pathlib
from ast import literal_eval
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
from data.utils import PathLike, read_lines, unpickle_data
from utils.logger import get_logger

from ..utils import decode_stimulus_string
from .scannet_scan import ScannetScan


def scannet_official_train_val(
    valid_views=None, verbose=True
) -> Dict[Literal["train", "test"], Set[str]]:
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    logger = get_logger(__name__)
    path_prefix = pathlib.Path(__file__).parent.parent.absolute()
    splits_desc = {
        "train": "meta/scannet/splits/official/v2/scannetv2_train.txt",
        "test": "meta/scannet/splits/official/v2/scannetv2_val.txt",
        "test_small": "meta/scannet/splits/official/v2/scannetv2_val_small.txt",
    }
    splits = dict()
    for k, v in splits_desc.items():
        path = osp.join(path_prefix, v)
        splits[k] = read_lines(path)

    if valid_views is not None:
        for k in splits:
            splits[k] = [sc for sc in splits[k] if sc[-2:] in valid_views]

    if verbose:
        for k in splits:
            logger.debug(f"{k} split has {len(splits[k])} scans.")

    scans_split = dict()
    for k in splits:
        scans_split[k] = set(splits[k])
    return scans_split


def objects_counter_percentile(scan_ids: List[str], all_scans: Dict[str, ScannetScan], prc: float):
    all_obs_len = list()
    for scan_id in all_scans:
        if scan_id in scan_ids:
            all_obs_len.append(len(all_scans[scan_id].three_d_objects))
    return np.percentile(all_obs_len, prc)


def mean_color(scan_ids: List[str], all_scans: Dict[str, ScannetScan]) -> np.ndarray:
    """
    Returns:
        np.ndarray: shape (3,). The mean RGB color of the points in the specified scans.
    """
    mean_rgb = np.zeros((3,), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points
    return mean_rgb


def load_referential_data(
    args, referit_csv: PathLike, scans_split: Dict[str, Set[str]]
) -> pd.DataFrame:
    """
    :param args:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    logger = get_logger(__name__)
    referit_data = pd.read_csv(referit_csv)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data["mentions_target_class"]]
        referit_data.reset_index(drop=True, inplace=True)
        logger.info(
            "Dropping utterances without explicit "
            "mention to the target class {}->{}".format(n_original, len(referit_data))
        )

    referit_data = referit_data[
        [
            "instance_type",
            "scan_id",
            "dataset",
            "target_id",
            "utterance",
            "stimulus_id",
            "utterance_generative",
        ]
    ]

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    for split_name, split_scan_ids in scans_split.items():
        cond = referit_data.scan_id.apply(lambda x: x in split_scan_ids)
        referit_data[f"is_{split_name}"] = cond

    # do this last, so that all the previous actions remain unchanged
    if args.augment_with_sr3d is not None:
        logger.info("Adding Sr3D as augmentation.")
        sr3d = pd.read_csv(args.augment_with_sr3d)
        # NOTE: sr3d only takes training data
        is_train = sr3d.scan_id.apply(lambda x: x in scans_split["train"])
        sr3d["is_train"] = is_train
        sr3d = sr3d[is_train]
        sr3d = sr3d[referit_data.columns]
        logger.info(f"Dataset-size before augmentation: {len(referit_data)}")
        referit_data = pd.concat([referit_data, sr3d], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        logger.info(f"Dataset-size after augmentation: {len(referit_data)}")

    return referit_data


def load_scan_related_data(preprocessed_scannet_file: PathLike, verbose=True, add_pad=True):
    logger = get_logger(__name__)
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    all_scans: List[ScannetScan]
    if verbose:
        logger.debug("Loaded in RAM {} scans".format(len(all_scans)))

    instance_labels = set()
    for scan in all_scans:
        idx = np.array([o.object_id for o in scan.three_d_objects])
        instance_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(
            idx == np.arange(len(idx))
        )  # assert the list of objects-ids -is- the range(n_objects).
        # because we use this ordering when we sample objects from a scan.
    all_scans_dict = {scan.scan_id: scan for scan in all_scans}  # place scans in dictionary

    class_to_idx: Dict[str, int] = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    if verbose:
        logger.debug("{} instance classes exist in these scans".format(len(class_to_idx)))

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx["pad"] = len(class_to_idx)

    scans_split = scannet_official_train_val()

    return all_scans_dict, scans_split, class_to_idx


def compute_auxiliary_data(
    referit_data: pd.DataFrame, all_scans: Dict[str, ScannetScan]
) -> np.ndarray:
    """
    Given a train-split compute useful quantities like mean-rgb.

    Args:
        referit_data (pd.DataFrame): pandas Dataframe, as returned from load_referential_data()
        all_scans (Dict[str, ScannetScan]): a dictionary holding ScannetScan objects

    Returns:
        np.ndarray: shape (1, 3). The mean RGB color of the points in the specified scans.
    """
    assert all_scans is not None
    logger = get_logger(__name__)

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data["is_train"]]["scan_id"])
    logger.info("{} training scans will be used.".format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    logger.info(
        "{}-th percentile of number of objects in the (training) scans"
        " is: {:.2f}".format(prc, obj_cnt)
    )

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data["is_train"]]["scan_id"])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    logger.info(
        "{}-th percentile of number of objects in the (testing) scans"
        " is: {:.2f}".format(prc, obj_cnt)
    )
    return mean_rgb


def trim_scans_per_referit3d_data_(referit_data: pd.DataFrame, scans: Dict[str, ScannetScan]):
    """
    Remove scans not in referit_data inplace.
    """
    logger = get_logger(__name__)
    in_r3d = referit_data["scan_id"].unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    logger.info("Dropped {} scans to reduce mem-foot-print.".format(len(to_drop)))
    return scans


##
## Are below necessary? Refactor. Maybe send to a _future_ package
## I think I wrote them to extract the classes of only the training data, but we rejected this idea.
##

# def object_classes_of_scans(scan_ids, all_scans, verbose=False):
#     """ get the object classes (e.g., chair, table...) that the specified scans contain.
#     :param scan_ids: a list of strings
#     :param all_scans: a dictionary holding ScannetScan objects
#     :return: a dictionary mapping all present object classes (string) to a unique int
#     """
#     object_classes = set()
#     for scan_id, scan in all_scans.items():
#         if scan_id in scan_ids:
#             object_classes.update([s.instance_label for s in scan.three_d_objects])
#
#     if verbose:
#         print('{} object classes were found.'.format(len(object_classes)))
#     return object_classes
#
#
# def object_class_to_idx_dictionary(object_classes, add_pad=False):
#     class_to_idx = {m: i for i, m in enumerate(sorted(list(object_classes)))}
#     if add_pad:
#         class_to_idx['pad'] = len(class_to_idx)
#     return class_to_idx
