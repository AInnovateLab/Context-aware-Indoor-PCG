import numpy as np
from numba import float32, int32
from numba.experimental import jitclass

spec = [
    ("n_samples", int32),
    ("selected_pts_expanded", float32[:, :, :]),
    ("selected_pts_idx", int32[:]),
    ("pcd_xyz", float32[:, :]),
    ("n_selected_pts", int32),
    ("dist_pts_to_selected_min", float32[:]),
    ("res_selected_idx", int32),
]


@jitclass(spec)
class FPS:
    def __init__(self, pcd_xyz, n_samples):
        assert n_samples >= 1, "n_samples should be >= 1"
        self.n_samples = n_samples
        n_pts, dim = pcd_xyz.shape
        self.pcd_xyz = pcd_xyz.astype(np.float32)
        # Random pick a start
        start_idx = np.random.randint(low=0, high=n_pts)
        self.n_selected_pts = 1
        self.dist_pts_to_selected_min = np.zeros((n_pts,), dtype=np.float32)
        self.res_selected_idx = -1

        self.selected_pts_expanded = np.zeros((n_samples, 1, dim), dtype=np.float32)
        self.selected_pts_expanded[0] = self.pcd_xyz[start_idx]
        self.selected_pts_idx = np.zeros((n_samples,), dtype=np.int32)
        self.selected_pts_idx[0] = start_idx

    def step(self):
        if self.n_selected_pts == 1:
            dist_pts_to_selected = np.sum(
                (self.pcd_xyz - self.selected_pts_expanded[:1]) ** 2, axis=2
            ).T  # (n_pts, 1)
            # write in numba way
            self.dist_pts_to_selected_min = dist_pts_to_selected[:, 0]
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.pcd_xyz[self.res_selected_idx]
            self.selected_pts_idx[self.n_selected_pts] = self.res_selected_idx
            self.n_selected_pts += 1

        elif self.n_selected_pts < self.n_samples:
            dist_pts_to_selected = self.distance(
                self.pcd_xyz, self.pcd_xyz[self.res_selected_idx][None, None]
            ).T  # (n_pts, 1)
            dist_pts_to_selected = dist_pts_to_selected[:, 0]
            self.dist_pts_to_selected_min = np.minimum(
                self.dist_pts_to_selected_min, dist_pts_to_selected
            )
            self.res_selected_idx = np.argmax(self.dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.pcd_xyz[self.res_selected_idx]
            self.selected_pts_idx[self.n_selected_pts] = self.res_selected_idx
            self.n_selected_pts += 1
        else:
            pass

    def fit(self):
        """
        Returns:
            selected_pts_idx: (n_samples,), 1d int array of the indices of selected points,
        """
        assert (
            self.n_samples <= self.pcd_xyz.shape[0]
        ), "n_samples should be less than the number of points"
        for _ in range(1, self.n_samples):
            self.step()
        return self.selected_pts_idx

    def distance(self, a, b):
        return np.sum((a - b) ** 2, axis=2)
