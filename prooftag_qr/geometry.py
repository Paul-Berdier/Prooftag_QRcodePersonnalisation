from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qrcode
from PIL import Image
from qrcode.exceptions import DataOverflowError

from .qr import ERROR_LEVELS, QRBlueprint, functional_pattern_mask


@dataclass(frozen=True, slots=True)
class AlignedQR:
    """A QR whose core modules are aligned to an integer pixel grid.

    Diffusion dimensions are usually divisible by eight, while a QR including its standard
    four-module border often is not.  The quiet zone is therefore represented as canvas padding,
    rather than by resizing a complete QR and silently changing every module width.
    """

    image: Image.Image
    core_matrix: np.ndarray
    version: int
    error_correction: str
    mask_pattern: int
    module_size: int
    padding_px: int
    canvas_size: int
    payload: str

    @property
    def core_modules(self) -> int:
        return int(self.core_matrix.shape[0])

    @property
    def core_size(self) -> int:
        return self.core_modules * self.module_size

    @property
    def quiet_zone_modules(self) -> float:
        return self.padding_px / self.module_size

    @property
    def core_blueprint(self) -> QRBlueprint:
        core = self.image.crop(
            (
                self.padding_px,
                self.padding_px,
                self.padding_px + self.core_size,
                self.padding_px + self.core_size,
            )
        )
        return QRBlueprint(
            image=core,
            matrix=self.core_matrix.copy(),
            version=self.version,
            border=0,
        )


def generate_aligned_qr(
    payload: str,
    *,
    version: int,
    error_correction: str,
    mask_pattern: int,
    module_size: int,
    canvas_size: int,
) -> AlignedQR:
    """Generate an exact module grid centered on a diffusion-compatible white canvas."""
    if error_correction not in ERROR_LEVELS:
        raise ValueError(f"unsupported error correction: {error_correction}")
    if not 0 <= mask_pattern <= 7:
        raise ValueError("mask_pattern must be between 0 and 7")
    if module_size < 1 or canvas_size < 1:
        raise ValueError("module_size and canvas_size must be positive")
    if canvas_size % 8:
        raise ValueError("canvas_size must be divisible by eight for latent diffusion")

    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_LEVELS[error_correction],
        box_size=module_size,
        border=0,
        mask_pattern=mask_pattern,
    )
    qr.add_data(payload)
    try:
        qr.make(fit=False)
    except DataOverflowError as exc:
        raise ValueError(
            f"payload does not fit QR v{version}/{error_correction}; shorten it explicitly"
        ) from exc

    matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError("QR matrix must be square")
    core_size = int(matrix.shape[0]) * module_size
    remainder = canvas_size - core_size
    if remainder < 0:
        raise ValueError(
            f"canvas {canvas_size}px is smaller than the {core_size}px QR core"
        )
    if remainder % 2:
        raise ValueError(
            "canvas and QR core parities differ; symmetric integer padding is impossible"
        )
    padding = remainder // 2
    if padding < 4 * module_size:
        raise ValueError(
            f"quiet zone is only {padding / module_size:.2f} modules; at least four are required"
        )

    binary = np.where(matrix, 0, 255).astype(np.uint8)
    core = Image.fromarray(binary, mode="L").resize(
        (core_size, core_size), Image.Resampling.NEAREST
    )
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(core.convert("RGB"), (padding, padding))
    return AlignedQR(
        image=canvas,
        core_matrix=matrix,
        version=version,
        error_correction=error_correction,
        mask_pattern=mask_pattern,
        module_size=module_size,
        padding_px=padding,
        canvas_size=canvas_size,
        payload=payload,
    )


def crop_aligned_core(image: Image.Image, aligned: AlignedQR) -> Image.Image:
    """Crop the exact QR core from a generated image without resampling it."""
    if image.size != (aligned.canvas_size, aligned.canvas_size):
        image = image.resize(
            (aligned.canvas_size, aligned.canvas_size), Image.Resampling.LANCZOS
        )
    start = aligned.padding_px
    end = start + aligned.core_size
    return image.crop((start, start, end, end))


def aligned_module_values(
    image: Image.Image,
    aligned: AlignedQR,
    *,
    center_fraction: float = 1.0,
) -> np.ndarray:
    """Return mean grayscale values for exact, optionally centered module regions."""
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0, 1]")
    gray = np.asarray(crop_aligned_core(image, aligned).convert("L"), dtype=np.float32)
    values = np.empty_like(aligned.core_matrix, dtype=np.float32)
    inset = (1 - center_fraction) * aligned.module_size / 2
    for row in range(aligned.core_modules):
        y0 = round(row * aligned.module_size + inset)
        y1 = round((row + 1) * aligned.module_size - inset)
        for col in range(aligned.core_modules):
            x0 = round(col * aligned.module_size + inset)
            x1 = round((col + 1) * aligned.module_size - inset)
            values[row, col] = float(gray[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)].mean())
    return values


def aligned_module_error_rate(
    image: Image.Image,
    aligned: AlignedQR,
    *,
    center_fraction: float = 1.0,
    threshold: float = 128.0,
) -> float:
    values = aligned_module_values(image, aligned, center_fraction=center_fraction)
    predicted = values < threshold
    return float(np.not_equal(predicted, aligned.core_matrix.astype(bool)).mean())


def aligned_module_diagnostics(
    image: Image.Image,
    aligned: AlignedQR,
    *,
    center_fraction: float = 1 / 3,
    dark_threshold: float = 0.45,
    light_threshold: float = 0.65,
) -> dict[str, float]:
    """Measure data and functional errors plus threshold safety margins."""
    values = aligned_module_values(
        image, aligned, center_fraction=center_fraction
    ) / 255.0
    target_dark = aligned.core_matrix.astype(bool)
    predicted_dark = values < 0.5
    errors = predicted_dark != target_dark
    functional = functional_pattern_mask(aligned.core_blueprint)
    safe = np.where(target_dark, values <= dark_threshold, values >= light_threshold)
    signed_margin = np.where(target_dark, dark_threshold - values, values - light_threshold)
    return {
        "module_error_rate": float(errors.mean()),
        "functional_module_error_rate": float(errors[functional].mean()),
        "data_module_error_rate": float(errors[~functional].mean()),
        "threshold_safe_rate": float(safe.mean()),
        "minimum_threshold_margin": float(signed_margin.min()),
        "mean_threshold_margin": float(signed_margin.mean()),
    }
