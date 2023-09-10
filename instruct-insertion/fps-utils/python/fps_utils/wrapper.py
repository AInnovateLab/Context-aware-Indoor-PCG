import numpy as np

from .fps_utils import _fps_sampling


def fps_sampling(pcd_xyz: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Args:
        pcd_xyz: (n_pts, 3)
        n_samples: int
    Returns:
        selected_pts_idx: (n_samples,)
    """
    assert n_samples >= 1, "n_samples should be >= 1"
    assert pcd_xyz.ndim == 2
    n_pts, _ = pcd_xyz.shape
    assert n_pts >= n_samples, "n_pts should be >= n_samples"
    pcd_xyz = pcd_xyz.astype(np.float32)
    pcd_xyz = np.asfortranarray(pcd_xyz)
    # Random pick a start
    start_idx = np.random.randint(low=0, high=n_pts)
    return _fps_sampling(pcd_xyz, n_samples, start_idx)
