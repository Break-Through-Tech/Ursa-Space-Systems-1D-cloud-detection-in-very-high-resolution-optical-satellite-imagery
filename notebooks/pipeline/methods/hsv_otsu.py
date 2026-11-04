
# 2. Add the Otsu plug-in
import numpy as np
from ..base import CloudDetector


def otsu_threshold(values, fallback=0.70):
    """Select a threshold by maximizing between-class variance."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0 or np.ptp(values) == 0:
        return float(fallback), "fixed_fallback"

    counts, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2

    left_n = np.cumsum(counts)[:-1].astype(float)
    right_n = counts.sum() - left_n

    left_sum = np.cumsum(counts * centers)[:-1]
    total_sum = np.sum(counts * centers)

    left_mean = left_sum / np.maximum(left_n, 1)
    right_mean = (total_sum - left_sum) / np.maximum(right_n, 1)

    variance = left_n * right_n * (left_mean - right_mean) ** 2

    return float(centers[np.argmax(variance)]), "otsu"


class HSVOtsu(CloudDetector):
    name = "hsv_otsu"

    def __init__(self, s_max=0.20, v_fallback=0.70):
        if not (0 <= s_max <= 1 and 0 <= v_fallback <= 1):
            raise ValueError("Thresholds must be in [0, 1]")

        self.s_max = float(s_max)
        self.v_fallback = float(v_fallback)
        self.last_thresholds = {}

    def predict(self, features, valid_mask):
        features = np.asarray(features)
        valid = np.asarray(valid_mask, dtype=bool)

        if features.ndim != 3 or features.shape[-1] != 7:
            raise ValueError("features must have shape (H, W, 7)")

        if valid.shape != features.shape[:2]:
            raise ValueError("valid_mask must match the feature grid")

        # Shared feature order:
        # R, G, B, brightness, saturation, NIR, NDVI
        v = features[..., 3]
        s = features[..., 4]

        valid = valid & np.isfinite(v) & np.isfinite(s)

        if valid.any():
            if (
                v[valid].min() < 0 or v[valid].max() > 1
                or s[valid].min() < 0 or s[valid].max() > 1
            ):
                raise ValueError(
                    "Brightness and saturation must be in [0, 1]"
                )

        threshold, source = otsu_threshold(
            v[valid], self.v_fallback
        )

        self.last_thresholds = {
            "v_threshold": threshold,
            "s_threshold": self.s_max,
            "threshold_source": source,
        }

        return (
            valid & (v > threshold) & (s < self.s_max)
        ).astype(np.uint8)
