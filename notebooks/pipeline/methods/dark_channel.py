import numpy as np
from scipy.ndimage import binary_closing, minimum_filter

from ..base import CloudDetector


def compute_dark_channel(rgb, patch_size, valid_mask=None):
    """Return the RGB minimum in each odd-sized local spatial window."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if not np.isfinite(rgb).all():
        raise ValueError("rgb values must be finite")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int):
        raise TypeError("patch_size must be an odd integer")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")

    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != rgb.shape[:2]:
            raise ValueError("valid_mask must match the RGB grid")
        rgb = np.where(valid[..., None], rgb, 1.0)

    local_min_per_band = minimum_filter(
        rgb,
        size=(patch_size, patch_size, 1),
        mode="nearest",
    )
    return local_min_per_band.min(axis=-1)


class DarkChannelPrior(CloudDetector):
    """DCP-inspired threshold baseline; high local minima indicate cloud."""

    name = "dark_channel_prior"

    def __init__(self, patch_size=15, threshold=0.65, refine_edges=False):
        if isinstance(patch_size, bool) or not isinstance(patch_size, int):
            raise TypeError("patch_size must be an odd integer")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")

        self.patch_size = patch_size
        self.threshold = float(threshold)
        self.refine_edges = bool(refine_edges)
        self.last_thresholds = {}

    def predict(self, features, valid_mask):
        features = np.asarray(features, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if features.ndim != 3 or features.shape[-1] != 7:
            raise ValueError("features must have shape (H, W, 7)")
        if valid.shape != features.shape[:2]:
            raise ValueError("valid_mask must match the feature grid")

        dark_channel = compute_dark_channel(features[..., :3], self.patch_size, valid)
        prediction = dark_channel >= self.threshold
        if self.refine_edges:
            prediction = binary_closing(
                prediction,
                structure=np.ones((3, 3), dtype=bool),
            )
        prediction &= valid
        self.last_thresholds = {
            "patch_size": self.patch_size,
            "dark_channel_threshold": self.threshold,
            "edge_refinement": self.refine_edges,
        }
        return prediction.astype(np.uint8)
