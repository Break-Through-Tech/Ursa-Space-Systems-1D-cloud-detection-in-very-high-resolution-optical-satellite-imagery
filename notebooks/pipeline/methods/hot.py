import numpy as np

from ..base import CloudDetector


class HOTBaseline(CloudDetector):
    """HOT-style RGB blue-minus-red cloud candidate detector."""

    name = "hot_rgb"

    def __init__(self, threshold=0.20, slope=0.5, intercept=0.0):
        parameters = (threshold, slope, intercept)
        if not all(np.isfinite(value) for value in parameters):
            raise ValueError("HOT parameters must be finite")
        if not -1 <= threshold <= 1:
            raise ValueError("threshold must be in [-1, 1]")
        if not 0 <= slope <= 2:
            raise ValueError("slope must be in [0, 2]")
        if not -1 <= intercept <= 1:
            raise ValueError("intercept must be in [-1, 1]")

        self.threshold = float(threshold)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.last_thresholds = self._threshold_metadata()

    def _threshold_metadata(self):
        return {
            "hot_threshold": self.threshold,
            "hot_slope": self.slope,
            "hot_intercept": self.intercept,
            "threshold_source": "fixed",
        }

    def predict(self, features, valid_mask):
        features = np.asarray(features, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        if features.ndim != 3 or features.shape[-1] != 7:
            raise ValueError("features must have shape (H, W, 7)")
        if valid.shape != features.shape[:2]:
            raise ValueError("valid_mask must match the feature grid")

        rgb = features[..., :3]
        if not np.isfinite(rgb).all():
            raise ValueError("RGB values must be finite")
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("RGB values must be in [0, 1]")

        red = rgb[..., 0]
        blue = rgb[..., 2]
        score = blue - (self.slope * red + self.intercept)
        self.last_thresholds = self._threshold_metadata()
        return (valid & (score > self.threshold)).astype(np.uint8)
