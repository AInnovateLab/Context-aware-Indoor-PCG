import os.path as osp
import pathlib
from ast import literal_eval
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import pandas as pd
from data.utils import PathLike, read_lines, unpickle_data

from ..utils import decode_stimulus_string
from .scannet_scan import ScannetScan


def scannet_official_train_val(
    valid_views=None, verbose=True
) -> Dict[Literal["train", "test"], Set[str]]:
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    path_prefix = pathlib.Path(__file__).parent.parent.absolute()
    train_split = osp.join(path_prefix, "meta/scannet/splits/official/v2/scannetv2_train.txt")
    train_split = read_lines(train_split)
    test_split = osp.join(path_prefix, "meta/scannet/splits/official/v2/scannetv2_val.txt")
    test_split = read_lines(test_split)

    if valid_views is not None:
        train_split = [sc for sc in train_split if sc[-2:] in valid_views]
        test_split = [sc for sc in test_split if sc[-2:] in valid_views]

    if verbose:
        print("#train/test scans:", len(train_split), "/", len(test_split))

    scans_split = dict()
    scans_split["train"] = set(train_split)
    scans_split["test"] = set(test_split)
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
    referit_data = pd.read_csv(referit_csv)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data["mentions_target_class"]]
        referit_data.reset_index(drop=True, inplace=True)
        print(
            "Dropping utterances without explicit "
            "mention to the target class {}->{}".format(n_original, len(referit_data))
        )

    referit_data = referit_data[
        ["tokens", "instance_type", "scan_id", "dataset", "target_id", "utterance", "stimulus_id"]
    ]
    referit_data.loc["tokens"] = referit_data["tokens"].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split["train"])
    referit_data["is_train"] = is_train

    # Trim data based on token length
    train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    print(
        "{}-th percentile of token length for remaining (training) data"
        " is: {:.1f}".format(95, np.percentile(train_token_lens, 95))
    )
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
    referit_data.reset_index(drop=True, inplace=True)
    print(
        "Dropping utterances with more than {} tokens, {}->{}".format(
            args.max_seq_len, n_original, len(referit_data)
        )
    )

    # do this last, so that all the previous actions remain unchanged
    if args.augment_with_sr3d is not None:
        print("Adding Sr3D as augmentation.")
        sr3d = pd.read_csv(args.augment_with_sr3d)
        sr3d.tokens = sr3d["tokens"].apply(literal_eval)
        is_train = sr3d.scan_id.apply(lambda x: x in scans_split["train"])
        sr3d["is_train"] = is_train
        sr3d = sr3d[is_train]
        sr3d = sr3d[referit_data.columns]
        print("Dataset-size before augmentation:", len(referit_data))
        referit_data = pd.concat([referit_data, sr3d], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        print("Dataset-size after augmentation:", len(referit_data))

    context_size = referit_data[~referit_data.is_train].stimulus_id.apply(
        lambda x: decode_stimulus_string(x)[2]
    )
    print(
        "(mean) Random guessing among target-class test objects {:.4f}".format(
            (1 / context_size).mean()
        )
    )

    return referit_data


def load_scan_related_data(preprocessed_scannet_file: PathLike, verbose=True, add_pad=True):
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    all_scans: List[ScannetScan]
    if verbose:
        print("Loaded in RAM {} scans".format(len(all_scans)))

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
        print("{} instance classes exist in these scans".format(len(class_to_idx)))

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

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data["is_train"]]["scan_id"])
    print("{} training scans will be used.".format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    print(
        "{}-th percentile of number of objects in the (training) scans"
        " is: {:.2f}".format(prc, obj_cnt)
    )

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data["is_train"]]["scan_id"])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    print(
        "{}-th percentile of number of objects in the (testing) scans"
        " is: {:.2f}".format(prc, obj_cnt)
    )
    return mean_rgb


def trim_scans_per_referit3d_data_(referit_data: pd.DataFrame, scans: Dict[str, ScannetScan]):
    """
    Remove scans not in referit_data inplace.
    """
    in_r3d = referit_data["scan_id"].unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print("Dropped {} scans to reduce mem-foot-print.".format(len(to_drop)))
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
