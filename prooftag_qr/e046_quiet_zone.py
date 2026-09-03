"""Quiet-zone E046 sans suppression uniforme de l'œuvre.

Les méthodes historiques `white` et `adaptive_light` remplacent toute la bordure
par une couleur uniforme. E046 conserve toujours le raster brut et propose une
variante de livraison *scene-preserving* :

- aucune découpe ;
- cœur QR 580x580 recopié octet pour octet ;
- périphérie issue de l'œuvre elle-même, fortement lissée, désaturée et éclaircie ;
- aucune couleur plate injectée sur toute la bordure ;
- métriques explicites pour comparer le gain de scan et le coût visuel.

Cette transformation n'est pas déclarée « téléphone-validée ».
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_PADDING_PX = 78
LUMA_COEFFICIENTS = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class QuietZoneMetrics:
    padding_px: int
    pixel_count: int
    luminance_mean: float
    luminance_p05: float
    luminance_min: float
    luminance_std: float
    dark_pixel_ratio: float
    edge_energy: float
    unique_color_count_capped: int
    flat_uniform: bool


def core_bounds(image: Image.Image, padding_px: int = DEFAULT_PADDING_PX) -> tuple[int, int, int, int]:
    if padding_px < 0:
        raise ValueError("padding_px must be non-negative")
    width, height = image.size
    if width <= 2 * padding_px or height <= 2 * padding_px:
        raise ValueError("padding removes the complete image")
    return padding_px, padding_px, width - padding_px, height - padding_px


def quiet_zone_mask(image: Image.Image, padding_px: int = DEFAULT_PADDING_PX) -> np.ndarray:
    left, top, right, bottom = core_bounds(image, padding_px)
    mask = np.ones((image.height, image.width), dtype=bool)
    mask[top:bottom, left:right] = False
    return mask


def core_pixel_sha256(image: Image.Image, padding_px: int = DEFAULT_PADDING_PX) -> str:
    left, top, right, bottom = core_bounds(image, padding_px)
    core = image.convert("RGB").crop((left, top, right, bottom))
    digest = hashlib.sha256()
    digest.update(f"RGB:{core.width}x{core.height}:".encode("ascii"))
    digest.update(core.tobytes())
    return digest.hexdigest()


def quiet_zone_metrics(
    image: Image.Image,
    *,
    padding_px: int = DEFAULT_PADDING_PX,
    dark_threshold: float = 0.55,
) -> QuietZoneMetrics:
    if not 0.0 < dark_threshold < 1.0:
        raise ValueError("dark_threshold must be between 0 and 1")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = quiet_zone_mask(image, padding_px)
    pixels = rgb[mask]
    luminance = pixels @ LUMA_COEFFICIENTS

    full_luma = rgb @ LUMA_COEFFICIENTS
    dx = np.abs(np.diff(full_luma, axis=1, prepend=full_luma[:, :1]))
    dy = np.abs(np.diff(full_luma, axis=0, prepend=full_luma[:1, :]))
    edge = ((dx + dy) * 0.5)[mask]

    quantized = np.rint(pixels * 31.0).astype(np.uint8)
    packed = (
        quantized[:, 0].astype(np.uint32) * 1024
        + quantized[:, 1].astype(np.uint32) * 32
        + quantized[:, 2].astype(np.uint32)
    )
    unique = min(4096, int(np.unique(packed).size))
    return QuietZoneMetrics(
        padding_px=padding_px,
        pixel_count=int(mask.sum()),
        luminance_mean=float(luminance.mean()),
        luminance_p05=float(np.quantile(luminance, 0.05)),
        luminance_min=float(luminance.min()),
        luminance_std=float(luminance.std()),
        dark_pixel_ratio=float((luminance < dark_threshold).mean()),
        edge_energy=float(edge.mean()),
        unique_color_count_capped=unique,
        flat_uniform=bool(unique <= 2 and float(luminance.std()) < 1e-4),
    )


def compose_scene_preserving_quiet_zone(
    image: Image.Image,
    *,
    padding_px: int = DEFAULT_PADDING_PX,
    minimum_luminance: float = 0.78,
    blur_radius: float = 12.0,
    saturation_retention: float = 0.30,
) -> tuple[Image.Image, dict[str, Any]]:
    """Éclaircit la bordure à partir de l'œuvre sans toucher au cœur QR."""
    if not 0.0 < minimum_luminance <= 1.0:
        raise ValueError("minimum_luminance must be in (0, 1]")
    if blur_radius <= 0:
        raise ValueError("blur_radius must be positive")
    if not 0.0 <= saturation_retention <= 1.0:
        raise ValueError("saturation_retention must be between 0 and 1")

    source = image.convert("RGB")
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    blurred = np.asarray(
        source.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
        dtype=np.float32,
    ) / 255.0

    blurred_luma = blurred @ LUMA_COEFFICIENTS
    neutral = np.repeat(blurred_luma[..., None], 3, axis=2)
    processed = neutral * (1.0 - saturation_retention) + blurred * saturation_retention

    # Stop below exact white so the composition does not introduce a clipped
    # rectangular frame. 0.985 still leaves enough headroom to reach the E046
    # luminance floor while preserving measurable palette variation.
    light_target = 0.985
    processed_luma = processed @ LUMA_COEFFICIENTS
    denominator = np.maximum(1e-6, light_target - processed_luma)
    whitening = np.clip(
        (minimum_luminance - processed_luma) / denominator,
        0.0,
        1.0,
    )
    processed = (
        processed * (1.0 - whitening[..., None])
        + light_target * whitening[..., None]
    )
    processed = np.minimum(processed, light_target)

    output_array = source_array.copy()
    mask = quiet_zone_mask(source, padding_px)
    output_array[mask] = processed[mask]
    output = Image.fromarray(
        np.rint(np.clip(output_array, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )

    source_core_hash = core_pixel_sha256(source, padding_px)
    output_core_hash = core_pixel_sha256(output, padding_px)
    if source_core_hash != output_core_hash:
        raise RuntimeError("scene-preserving quiet-zone modified QR core pixels")

    before = quiet_zone_metrics(source, padding_px=padding_px)
    after = quiet_zone_metrics(output, padding_px=padding_px)
    if output.size != source.size:
        raise RuntimeError("scene-preserving quiet-zone changed canvas dimensions")

    evidence = {
        "policy": "scene_preserving_blur_desaturate_lighten",
        "no_crop": True,
        "uniform_flat_replacement": False,
        "core_byte_identical": True,
        "source_core_sha256": source_core_hash,
        "output_core_sha256": output_core_hash,
        "parameters": {
            "padding_px": padding_px,
            "minimum_luminance": minimum_luminance,
            "blur_radius": blur_radius,
            "saturation_retention": saturation_retention,
        },
        "before": asdict(before),
        "after": asdict(after),
        "delivery_guard": {
            "core_byte_identical": True,
            "luminance_p05": after.luminance_p05 >= minimum_luminance - (1.5 / 255.0),
            "dark_pixel_ratio": after.dark_pixel_ratio <= 0.02,
            "not_flat_uniform": not after.flat_uniform,
            "no_crop": output.size == source.size,
        },
    }
    evidence["delivery_guard_pass"] = all(evidence["delivery_guard"].values())
    return output, evidence


def compare_core_bytes(
    left: Image.Image,
    right: Image.Image,
    *,
    padding_px: int = DEFAULT_PADDING_PX,
) -> bool:
    return core_pixel_sha256(left, padding_px) == core_pixel_sha256(right, padding_px)
