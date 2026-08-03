import hashlib

import cv2
import numpy as np
from PIL import Image


def image_sha256(image: Image.Image) -> str:
    """Hash decoded pixels, mode and dimensions instead of encoder metadata."""
    source = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{source.width}x{source.height}:".encode())
    digest.update(np.asarray(source, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def image_quality_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
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
        "rgb_clipped_channel_ratio": float(
            ((rgb <= 3) | (rgb >= 252)).mean()
        ),
        "saturation_mean": float(saturation.mean()),
        "saturation_p95": float(np.quantile(saturation, 0.95)),
        "high_saturation_pixel_ratio": float((saturation >= 0.90).mean()),
    }


def image_change_metrics(image: Image.Image, reference: Image.Image) -> dict[str, float]:
    candidate_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    baseline_rgb = np.asarray(
        reference.convert("RGB").resize(image.size), dtype=np.uint8
    )
    candidate = candidate_rgb.astype(np.int16)
    baseline = baseline_rgb.astype(np.int16)
    absolute_change = np.abs(candidate - baseline)
    candidate_saturation = (
        cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
        / 255.0
    )
    baseline_saturation = (
        cv2.cvtColor(baseline_rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
        / 255.0
    )
    candidate_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY)
    baseline_gray = cv2.cvtColor(baseline_rgb, cv2.COLOR_RGB2GRAY)
    return {
        "changed_pixel_ratio": float((absolute_change.max(axis=2) > 10).mean()),
        "mean_absolute_change": float(absolute_change.mean() / 255.0),
        "clipped_pixel_ratio_increase": float(
            ((candidate_gray <= 3) | (candidate_gray >= 252)).mean()
            - ((baseline_gray <= 3) | (baseline_gray >= 252)).mean()
        ),
        "rgb_clipped_channel_ratio_increase": float(
            ((candidate_rgb <= 3) | (candidate_rgb >= 252)).mean()
            - ((baseline_rgb <= 3) | (baseline_rgb >= 252)).mean()
        ),
        "saturation_mean_increase": float(
            candidate_saturation.mean() - baseline_saturation.mean()
        ),
        "high_saturation_ratio_increase": float(
            (candidate_saturation >= 0.90).mean()
            - (baseline_saturation >= 0.90).mean()
        ),
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
