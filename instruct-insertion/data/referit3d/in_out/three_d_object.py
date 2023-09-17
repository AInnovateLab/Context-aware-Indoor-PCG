from typing import TYPE_CHECKING, List, Optional

import fpsample
import numpy as np
import torch
from shapely.geometry import LineString, Polygon
from sklearn.neighbors import NearestNeighbors

from ..utils import rotate_points_along_z
from ..utils.plotting import plot_pointcloud
from .cuboid import OrientedCuboid

if TYPE_CHECKING:
    from .scannet_scan import ScannetScan


class ThreeDObject(object):
    """
    Representing a ScanNet 3D Object
    """

    def __init__(self, scan: "ScannetScan", object_id: str, points: List[int], instance_label: str):
        """
        Args:
            scan (ScannetScan): The scan that this object belongs to
            object_id (str): The object id
            points (List[int]): The indices of the points that belong to this object
            instance_label (str): The instance label of this object
        """
        self.scan = scan
        self.object_id = object_id
        self.points = points
        self._instance_label = instance_label

        self.axis_aligned_bbox = None
        self.is_axis_aligned_bbox_set = False

        self.object_aligned_bbox = None
        self.has_object_aligned_bbox = False

        self.front_direction = None
        self.has_front_direction = False
        self._use_true_instance = True

        self.pc = None  # The point cloud (xyz)

    @property
    def instance_label(self):
        if self._use_true_instance:
            return self._instance_label
        else:
            return self.semantic_label()

    @instance_label.setter
    def instance_label(self, instance_label):
        self._instance_label = instance_label

    @property
    def color(self):
        """
        The color (RGB) of the object's point cloud.
        """
        return self.scan.color[self.points]

    def plot(self, with_color=True):
        pc = self.get_pc()
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        color = None
        if with_color:
            color = self.color
        return plot_pointcloud(x, y, z, color=color)

    def z_min(self):
        bbox = self.get_axis_align_bbox()
        return bbox.extrema[2]

    def z_max(self):
        bbox = self.get_axis_align_bbox()
        return bbox.extrema[5]

    def set_axis_align_bbox(self):
        pc = self.get_pc()

        cx, cy, cz = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2.0
        lx, ly, lz = np.max(pc, axis=0) - np.min(pc, axis=0)
        rot = np.eye(N=3)
        assert lx > 0 and ly > 0 and lz > 0

        self.axis_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, rot)
        self.is_axis_aligned_bbox_set = True

    def get_axis_align_bbox(self):
        if self.is_axis_aligned_bbox_set:
            pass
        else:
            self.set_axis_align_bbox()
        return self.axis_aligned_bbox

    def set_pc(self):
        if self.pc is None:
            self.pc = self.scan.pc[self.points]

    def get_pc(self):
        # Set the pc if not previously initialized
        self.set_pc()

        return self.pc

    def set_object_aligned_bbox(self, cx, cy, cz, lx, ly, lz, rot):
        self.object_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, rot)
        self.has_object_aligned_bbox = True

    def get_bbox(self, axis_aligned=False):
        """if you have object-align return this, else compute/return axis-aligned"""
        if not axis_aligned and self.has_object_aligned_bbox:
            return self.object_aligned_bbox
        else:
            return self.get_axis_align_bbox()

    def iou_2d(self, other):
        a = self.get_bbox(axis_aligned=True).corners
        b = other.get_bbox(axis_aligned=True).corners

        a_xmin, a_xmax = np.min(a[:, 0]), np.max(a[:, 0])
        a_ymin, a_ymax = np.min(a[:, 1]), np.max(a[:, 1])

        b_xmin, b_xmax = np.min(b[:, 0]), np.max(b[:, 0])
        b_ymin, b_ymax = np.min(b[:, 1]), np.max(b[:, 1])

        box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
        box_b = [b_xmin, b_ymin, b_xmax, b_ymax]

        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

        return iou, i_ratios, a_ratios

    def visualize_axis_align_bbox(self, axis=None):
        bbox = self.get_axis_align_bbox()
        return bbox.plot(axis=axis)

    def intersection(self, other, axis=2):
        bbox = self.get_bbox(axis_aligned=True)
        l_min, l_max = bbox.extrema[axis], bbox.extrema[axis + 3]

        other_bbox = other.get_bbox(axis_aligned=True)
        other_l_min, other_l_max = other_bbox.extrema[axis], other_bbox.extrema[axis + 3]

        a = max(l_min, other_l_min)
        b = min(l_max, other_l_max)
        i = b - a

        return i, i / (l_max - l_min), i / (other_l_max - other_l_min)

    def semantic_label(self):
        one_point = self.scan.semantic_label[self.points[0]]
        return self.scan.dataset.idx_to_semantic_cls(one_point)

    def distance_from_other_object(self, other, optimized=False):
        if optimized:
            z_face = self.get_bbox().z_faces()[0]  # Top face
            points = tuple(map(tuple, z_face[:, :2]))  # x, y coordinates
            center = (self.get_bbox().cx, self.get_bbox().cy)

            other_z_face = other.get_bbox().z_faces()[0]
            other_points = tuple(map(tuple, other_z_face[:, :2]))
            other_center = (other.get_bbox().cx, other.get_bbox().cy)

            cent_line = LineString([center, other_center])
            return cent_line.intersection(Polygon(points + other_points).convex_hull).length
        else:
            nn = NearestNeighbors(n_neighbors=1).fit(self.get_pc())
            distances, _ = nn.kneighbors(other.get_pc())
            res = np.min(distances)
        return res

    @torch.no_grad()
    def sample(self, n_samples, use_fps=False, max_fps_candidates: Optional[int] = None):
        """sub-sample its pointcloud and color"""
        xyz = self.get_pc()
        color = self.color

        n_points = len(self.points)
        assert xyz.shape[0] == len(self.points)

        # fps - furthest point sampling
        if use_fps:
            if n_points < n_samples:
                idx = np.random.choice(n_points, n_samples - n_points, replace=True)
                idx = np.concatenate((np.arange(n_points), idx))
                np.random.shuffle(idx)
            elif n_points == n_samples:
                idx = np.arange(n_points)
                np.random.shuffle(idx)
            else:
                max_fps_candidates = max_fps_candidates or 4 * n_samples
                assert max_fps_candidates >= n_samples, "max_fps_candidates must be >= n_samples"
                idx = fpsample.bucket_fps_kdline_sampling(xyz, n_samples, h=5)

            object_samples = {
                "xyz": xyz[idx].copy(),
                "color": color[idx].copy(),
            }
        else:
            idx = np.random.choice(n_points, n_samples, replace=n_points < n_samples)
            object_samples = {
                "xyz": xyz[idx].copy(),
                "color": color[idx].copy(),
            }

        return object_samples
