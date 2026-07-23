from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urldefrag

import numpy as np
from PIL import Image

from .geometry import AlignedQR, aligned_module_values, generate_aligned_qr
from .qr import functional_pattern_mask


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    aligned: AlignedQR
    reference_cost: float
    grid_visibility: float


@dataclass(frozen=True, slots=True)
class AdaptiveBlueprint:
    image: Image.Image
    center_fractions: np.ndarray
    functional_modules: np.ndarray
    reference_cost: float
    grid_visibility: float


def canonical_url_match(decoded: str, expected: str) -> bool:
    """Accept a QArt URL only when removing its fragment yields the exact expected URL."""
    if not decoded:
        return False
    decoded_base, _ = urldefrag(decoded)
    expected_base, expected_fragment = urldefrag(expected)
    return not expected_fragment and decoded_base == expected_base


def reference_cost(image: Image.Image, reference: Image.Image) -> float:
    """Normalized RGB mean absolute error; lower means closer to the reference."""
    target = np.asarray(reference.convert("RGB").resize(image.size), dtype=np.float32)
    candidate = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(np.abs(candidate - target).mean() / 255.0)


def grid_visibility_score(image: Image.Image, aligned: AlignedQR) -> float:
    """Measure periodic edge energy on module borders relative to all local edge energy.

    A binary QR is close to one. A visually integrated blueprint should move toward zero.
    This is an engineering diagnostic, not a perceptual quality or scan metric.
    """
    source = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if source.shape != (aligned.canvas_size, aligned.canvas_size):
        source = np.asarray(
            image.convert("L").resize(
                (aligned.canvas_size, aligned.canvas_size), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0
    start = aligned.padding_px
    end = start + aligned.core_size
    core = source[start:end, start:end]
    horizontal = np.abs(np.diff(core, axis=0))
    vertical = np.abs(np.diff(core, axis=1))
    border_indexes = np.arange(aligned.module_size - 1, aligned.core_size - 1, aligned.module_size)
    periodic = np.concatenate(
        [
            horizontal[border_indexes, :].reshape(-1),
            vertical[:, border_indexes].reshape(-1),
        ]
    )
    all_edges = np.concatenate([horizontal.reshape(-1), vertical.reshape(-1)])
    denominator = float(all_edges.mean())
    return float(periodic.mean() / denominator) if denominator > 1e-8 else 0.0


def exact_mask_candidates(
    payload: str,
    reference: Image.Image,
    *,
    version: int,
    error_correction: str,
    module_size: int,
    canvas_size: int,
) -> list[MaskCandidate]:
    """Return all eight standards-compliant exact-payload QR masks ranked by appearance."""
    candidates: list[MaskCandidate] = []
    for mask_pattern in range(8):
        aligned = generate_aligned_qr(
            payload,
            version=version,
            error_correction=error_correction,
            mask_pattern=mask_pattern,
            module_size=module_size,
            canvas_size=canvas_size,
        )
        candidates.append(
            MaskCandidate(
                aligned=aligned,
                reference_cost=reference_cost(aligned.image, reference),
                grid_visibility=grid_visibility_score(aligned.image, aligned),
            )
        )
    return sorted(candidates, key=lambda item: (item.reference_cost, item.grid_visibility))


def _required_square_fraction(
    value: float,
    *,
    target_dark: bool,
    dark_threshold: float,
    light_threshold: float,
) -> float:
    """Minimum centered square side fraction required by a module-average model."""
    if target_dark:
        if value <= dark_threshold:
            return 0.0
        area_fraction = 1.0 - dark_threshold / max(value, 1e-6)
    else:
        if value >= light_threshold:
            return 0.0
        area_fraction = (light_threshold - value) / max(1.0 - value, 1e-6)
    return math.sqrt(float(np.clip(area_fraction, 0.0, 1.0)))


def build_adaptive_blueprint(
    reference: Image.Image,
    aligned: AlignedQR,
    *,
    dark_threshold: float = 0.35,
    light_threshold: float = 0.72,
    minimum_data_fraction: float = 0.22,
    maximum_data_fraction: float = 0.92,
    functional_fraction: float = 1.0,
    safety_fraction: float = 0.08,
) -> AdaptiveBlueprint:
    """Embed exact module centers into a reference while protecting functional modules.

    The side of each data-module square is derived from its local brightness. Finder, timing,
    alignment, format and version modules remain fully binary. The quiet zone is always white.
    This is a documented Prooftag adaptive blueprint, not the Reed-Solomon QArt algorithm.
    """
    if not 0 < dark_threshold < light_threshold < 1:
        raise ValueError("thresholds must satisfy 0 < dark < light < 1")
    for name, value in {
        "minimum_data_fraction": minimum_data_fraction,
        "maximum_data_fraction": maximum_data_fraction,
        "functional_fraction": functional_fraction,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    if minimum_data_fraction > maximum_data_fraction:
        raise ValueError("minimum_data_fraction cannot exceed maximum_data_fraction")

    canvas = reference.convert("RGB").resize(
        (aligned.canvas_size, aligned.canvas_size), Image.Resampling.LANCZOS
    )
    output = Image.new("RGB", canvas.size, "white")
    start = aligned.padding_px
    output.paste(
        canvas.crop((start, start, start + aligned.core_size, start + aligned.core_size)),
        (start, start),
    )

    source_values = aligned_module_values(canvas, aligned, center_fraction=1.0) / 255.0
    functional = functional_pattern_mask(aligned.core_blueprint)
    fractions = np.zeros_like(source_values, dtype=np.float32)
    pixels = np.asarray(output).copy()

    for row in range(aligned.core_modules):
        for column in range(aligned.core_modules):
            target_dark = bool(aligned.core_matrix[row, column])
            if functional[row, column]:
                fraction = functional_fraction
            else:
                required = _required_square_fraction(
                    float(source_values[row, column]),
                    target_dark=target_dark,
                    dark_threshold=dark_threshold,
                    light_threshold=light_threshold,
                )
                fraction = np.clip(
                    max(minimum_data_fraction, required + safety_fraction),
                    minimum_data_fraction,
                    maximum_data_fraction,
                )
            fractions[row, column] = fraction
            side = max(1, round(aligned.module_size * float(fraction)))
            side = min(side, aligned.module_size)
            inset = (aligned.module_size - side) // 2
            y0 = start + row * aligned.module_size + inset
            x0 = start + column * aligned.module_size + inset
            color = 0 if target_dark else 255
            pixels[y0 : y0 + side, x0 : x0 + side, :] = color

    image = Image.fromarray(pixels, mode="RGB")
    return AdaptiveBlueprint(
        image=image,
        center_fractions=fractions,
        functional_modules=functional,
        reference_cost=reference_cost(image, canvas),
        grid_visibility=grid_visibility_score(image, aligned),
    )


def align_qart_output(
    qart_image: Image.Image,
    *,
    payload: str,
    version: int,
    module_size: int,
    canvas_size: int,
    border_modules: int = 10,
) -> AlignedQR:
    """Crop QArt's ten-module border and place its untouched core on an aligned canvas."""
    core_modules = 17 + 4 * version
    expected = (core_modules + 2 * border_modules) * module_size
    if qart_image.size != (expected, expected):
        raise ValueError(
            f"QArt output is {qart_image.size}, expected {(expected, expected)}; "
            "refuse to infer a resampling"
        )
    border_px = border_modules * module_size
    core_size = core_modules * module_size
    remainder = canvas_size - core_size
    if canvas_size % 8 or remainder < 0 or remainder % 2:
        raise ValueError("canvas cannot hold a symmetric integer-aligned QArt core")
    padding = remainder // 2
    if padding < 4 * module_size:
        raise ValueError("aligned QArt quiet zone must be at least four modules")

    core = qart_image.convert("RGB").crop(
        (border_px, border_px, border_px + core_size, border_px + core_size)
    )
    gray = np.asarray(core.convert("L"), dtype=np.float32)
    matrix = np.zeros((core_modules, core_modules), dtype=np.uint8)
    for row in range(core_modules):
        for column in range(core_modules):
            region = gray[
                row * module_size : (row + 1) * module_size,
                column * module_size : (column + 1) * module_size,
            ]
            matrix[row, column] = int(region.mean() < 128)
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(core, (padding, padding))
    return AlignedQR(
        image=canvas,
        core_matrix=matrix,
        version=version,
        error_correction="L",
        mask_pattern=-1,
        module_size=module_size,
        padding_px=padding,
        canvas_size=canvas_size,
        payload=payload,
    )
