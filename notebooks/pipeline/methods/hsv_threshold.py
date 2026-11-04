import numpy as np
from ..base import CloudDetector

class HSVThreshold(CloudDetector):
    name = "hsv_threshold"

    def __init__(self, v_min=0.70, s_max=0.20):
        self.v_min = v_min
        self.s_max = s_max

    def predict(self, features, valid_mask):
        v = features[..., 3]  # brightness
        s = features[..., 4]  # saturation
        return ((v > self.v_min) & (s < self.s_max) & valid_mask).astype(np.uint8)
