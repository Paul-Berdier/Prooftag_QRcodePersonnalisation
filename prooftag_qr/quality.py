import cv2
import numpy as np
from PIL import Image


def image_quality_metrics(image: Image.Image) -> dict[str, float]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(((gray <= 3) | (gray >= 252)).mean())
    return {
        "brightness_mean": float(gray.mean() / 255.0),
        "contrast_std": float(gray.std() / 127.5),
        "entropy_bits": entropy,
        "sharpness_laplacian": laplacian_variance,
        "clipped_pixel_ratio": clipped,
    }
