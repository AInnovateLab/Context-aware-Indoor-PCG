import multiprocessing as mp
import warnings
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ..in_out.scannet_scan import ScannetScan

############################
#                          #
#    Dataset IO related    #
#                          #
############################


def max_io_workers():
    """
    Returns:
        int: number of available cores - 1.
    """
    return max(mp.cpu_count() - 1, 1)


def dataset_to_dataloader(dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
    batch_size_multiplier = 1 if split == "train" else 2
    b_size = int(batch_size_multiplier * batch_size)

    drop_last = False
    if split == "train" and len(dataset) % b_size == 1:
        print("dropping last batch during training")
        drop_last = True

    shuffle = split == "train"

    worker_init_fn = lambda x: np.random.seed(seed)
    if split == "test":
        if type(seed) is not int:
            warnings.warn("Test split is not seeded in a deterministic manner.")

    data_loader = DataLoader(
        dataset,
        batch_size=b_size,
        num_workers=n_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    return data_loader


######################################
#                                    #
#    data (pre)precessing related    #
#                                    #
######################################


def sample_scan_object(object, n_points, training=False, use_fps=False, rank=0):
    sample = object.sample(n_samples=n_points, training=training, use_fps=use_fps, rank=rank)
    return np.concatenate([sample["xyz"], sample["color"]], axis=1)


def pad_samples(samples, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.zeros(shape, dtype=samples.dtype) * padding_value
        temp[: samples.shape[0], : samples.shape[1]] = samples
        samples = temp

    return samples


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
                print("Check failed for {}".format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    ori_instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        ori_instance_labels.extend(["pad"] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in ori_instance_labels])

    # ori_labels=[]
    # for ori_label in ori_instance_labels:
    #     ori_labels.append('[CLS] '+ori_label+' [SEP]')
    # ori_instance_labels = ' '.join(ori_labels)

    return instance_labels


def mean_rgb_unit_norm_transform(
    segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True
):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz

    return segmented_objects


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