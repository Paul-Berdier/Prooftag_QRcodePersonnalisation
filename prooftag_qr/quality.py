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


def image_change_metrics(image: Image.Image, reference: Image.Image) -> dict[str, float]:
    candidate = np.asarray(image.convert("RGB"), dtype=np.int16)
    baseline = np.asarray(reference.convert("RGB").resize(image.size), dtype=np.int16)
    absolute_change = np.abs(candidate - baseline)
    return {
        "changed_pixel_ratio": float((absolute_change.max(axis=2) > 10).mean()),
        "mean_absolute_change": float(absolute_change.mean() / 255.0),
    }


def composite_guided_regions(
    reference: Image.Image,
    generated: Image.Image,
    control: Image.Image,
    *,
    dilation_px: int = 4,
    feather_px: int = 4,
) -> tuple[Image.Image, Image.Image]:
    """Project a rediffused image onto the QR-guided neighborhoods only."""
    if dilation_px < 0 or feather_px < 0:
        raise ValueError("dilation_px and feather_px cannot be negative")
    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    generated_array = np.asarray(
        generated.convert("RGB").resize(reference.size), dtype=np.float32
    )
    control_array = np.asarray(control.convert("RGB").resize(reference.size), dtype=np.float32)
    changed = np.max(np.abs(control_array - reference_array), axis=2) > 3
    mask = changed.astype(np.uint8) * 255
    if dilation_px:
        size = dilation_px * 2 + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    if feather_px:
        size = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (size, size), sigmaX=max(1.0, feather_px / 2))
    alpha = mask.astype(np.float32)[..., None] / 255.0
    composite = reference_array * (1.0 - alpha) + generated_array * alpha
    return (
        Image.fromarray(np.rint(composite).clip(0, 255).astype(np.uint8), mode="RGB"),
        Image.fromarray(mask, mode="L"),
    )
