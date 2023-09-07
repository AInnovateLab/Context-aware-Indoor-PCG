import multiprocessing as mp
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
from utils.logger import get_logger

if TYPE_CHECKING:
    from ..in_out.scannet_scan import ScannetScan
    from ..in_out.three_d_object import ThreeDObject

############################
#                          #
#    Dataset IO related    #
#                          #
############################


def dataset_to_dataloader(
    dataset, split, batch_size, n_workers, pin_memory=True, collate_fn=None
) -> DataLoader:
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    shuffle = split == "train"

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return data_loader


######################################
#                                    #
#    data (pre)precessing related    #
#                                    #
######################################


def sample_scan_object(object: "ThreeDObject", n_points: int, use_fps=False) -> np.ndarray:
    sample = object.sample(n_samples=n_points, use_fps=use_fps)
    return np.concatenate([sample["xyz"], sample["color"]], axis=1)


def check_segmented_object_order(scans: Dict[str, "ScannetScan"]):
    """
    Check all scan objects have the three_d_objects sorted by id.

    Args:
        scans (dict):
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                logger = get_logger()
                logger.critical(f"Objects are not ordered by object id: {scan_id}")
                return False
            idx += 1
    return True


def instance_labels_of_context(
    context: List["ThreeDObject"],
    max_context_size: int,
    label_to_idx: Dict[str, int],
    add_padding=True,
):
    """
    :param context: a list of the objects
    :return:
    """
    ori_instance_labels = [o.instance_label for o in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        ori_instance_labels.extend(["pad"] * n_pad)

    instance_labels = np.array([label_to_idx[x] for x in ori_instance_labels])

    return instance_labels


def normalize_pc(
    segmented_objects: np.ndarray,
    unit_norm: bool,
    mean_rgb: Optional[np.ndarray] = None,
    box_center: Optional[np.ndarray] = None,
    box_max_dist: Optional[np.ndarray] = None,
    inplace=True,
    **kwargs,
) -> np.ndarray:
    """
    Normalize the segmented objects.

    Args:
        segmented_objects (np.ndarray): (# of objects, # of points, 6 or 7), point-clouds with color (and height).
        mean_rgb (np.ndarray): (3,), the mean RGB color of the points of all scans.
        unit_norm (bool): If True, the xyz coordinates are normalized to the unit sphere.
        box_center (np.ndarray): (# of objects, 3), the center of the bounding box of each object. If not provided,
            the center of the samples is used.
        box_max_dist (np.ndarray): (# of objects,), the maximum distance of the points from the center
            of the bounding box. If not provided, the maximum distance of the points from the center of samples is used.
        inplace (bool): If False, the transformation is applied in a copy of the segmented_objects.
    """
    if len(kwargs):
        warnings.warn(f"Unused arguments: {kwargs}")
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    if mean_rgb is not None:
        segmented_objects[:, :, 3:6] -= mean_rgb[None, None, :]

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        if box_center is None:
            mean_center = xyz.mean(axis=1)
        else:
            mean_center = box_center
        xyz -= mean_center[:, None, :]
        if box_max_dist is None:
            max_dist = np.linalg.norm(xyz, axis=-1, ord=2).max(axis=-1)
        else:
            max_dist = box_max_dist
        xyz /= max_dist[:, None, None]
        segmented_objects[:, :, :3] = xyz

    return segmented_objects


def infer_floor_z_coord(scan: "ScannetScan", floor_label: str = "floor") -> float:
    """
    Infer the z-coordinate of the floor. If the "floor" exists, the maximum
    z-coordinate of the floor is returned. Otherwise, the minimum z-coordinate
    of all the points in the scan is returned.

    Args:
        scan (ScannetScan): The scan to infer the floor z-coordinate from.

    Returns:
        float: The z-coordinate of the upper surface of the floor.
    """
    floor_zs = [
        obj.get_bbox().extrema[5]
        for obj in scan.three_d_objects
        if obj.instance_label == floor_label
    ]
    if len(floor_zs) > 0:
        return sum(floor_zs) / len(floor_zs)
    else:
        # no floor found
        return scan.pc[:, 2].min()


def check_numpy_to_torch(x):
    import torch

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    import torch

    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def decode_stimulus_string(s: str) -> Tuple[str, str, int, int, List[int]]:
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    Args:
        s: stimulus string

    Example:
        A typical stimulus string is like "scene0525_00-plant-5-9-10-11-12-62".
        The scene_id is "scene0525_00", the instance_label is "plant",
        the number of objects is 5, the target object id is 9, and the
        distractors object ids are 10, 11, 12, and 62.
    """
    splits = s.split("-")
    assert len(splits) >= 4, f"Invalid stimulus string: {s}"
    scene_id, instance_label, n_objects, target_id = splits[:4]

    # type conversion
    instance_label = instance_label.replace("_", " ")
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in splits[4:] if i != ""]
    assert len(distractors_ids) == n_objects - 1, f"Invalid stimulus string: {s}"

    return scene_id, instance_label, n_objects, target_id, distractors_ids
