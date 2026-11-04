from abc import ABC, abstractmethod
import numpy as np

class CloudDetector(ABC):
    name = "base"  # becomes the output folder name

    @abstractmethod
    def predict(self, features: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        features:   (H, W, 7) float32 -> R, G, B, brightness, saturation, NIR, NDVI
        valid_mask: (H, W) bool, True = real pixel
        returns:    (H, W) uint8, 1 = cloud, 0 = not cloud
        """
