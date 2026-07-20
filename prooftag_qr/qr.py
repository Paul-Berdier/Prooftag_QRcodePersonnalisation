from dataclasses import dataclass

import numpy as np
import qrcode
from PIL import Image
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_M, ERROR_CORRECT_Q
from qrcode.util import pattern_position

ERROR_LEVELS = {"M": ERROR_CORRECT_M, "Q": ERROR_CORRECT_Q, "H": ERROR_CORRECT_H}


@dataclass(slots=True)
class QRBlueprint:
    image: Image.Image
    matrix: np.ndarray
    version: int
    border: int


def generate_qr(
    payload: str, error_correction: str, size: int = 512, border: int = 4
) -> QRBlueprint:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_LEVELS[error_correction],
        box_size=16,
        border=border,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return QRBlueprint(image=image, matrix=matrix, version=qr.version, border=border)


def module_error_rate(candidate: Image.Image, blueprint: QRBlueprint) -> float:
    gray = np.asarray(candidate.convert("L"), dtype=np.float32)
    count = blueprint.matrix.shape[0]
    predicted = np.zeros_like(blueprint.matrix)
    for row in range(count):
        y0 = round(row * gray.shape[0] / count)
        y1 = max(y0 + 1, round((row + 1) * gray.shape[0] / count))
        for col in range(count):
            x0 = round(col * gray.shape[1] / count)
            x1 = max(x0 + 1, round((col + 1) * gray.shape[1] / count))
            predicted[row, col] = gray[y0:y1, x0:x1].mean() < 128
    return float(np.not_equal(predicted, blueprint.matrix).mean())


def functional_pattern_mask(blueprint: QRBlueprint) -> np.ndarray:
    """Return modules that must remain structurally exact, including the quiet zone."""
    count = blueprint.matrix.shape[0]
    border = blueprint.border
    core = count - 2 * border
    mask = np.zeros_like(blueprint.matrix, dtype=bool)

    if border:
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True

    def protect(row_start: int, row_end: int, col_start: int, col_end: int) -> None:
        mask[
            border + row_start : border + row_end,
            border + col_start : border + col_end,
        ] = True

    # Finder patterns and their separators.
    protect(0, 9, 0, 9)
    protect(0, 9, core - 8, core)
    protect(core - 8, core, 0, 9)

    # Timing, format information and the fixed dark module.
    protect(6, 7, 0, core)
    protect(0, core, 6, 7)
    protect(8, 9, 0, core)
    protect(0, core, 8, 9)

    # Alignment patterns that do not overlap a finder pattern.
    for row in pattern_position(blueprint.version):
        for col in pattern_position(blueprint.version):
            overlaps_finder = (row == 6 and col in {6, core - 7}) or (row == core - 7 and col == 6)
            if not overlaps_finder:
                protect(row - 2, row + 3, col - 2, col + 3)

    # Version information appears beside the top-right and bottom-left finders.
    if blueprint.version >= 7:
        protect(0, 6, core - 11, core - 8)
        protect(core - 11, core - 8, 0, 6)
    return mask


def repair_qr_modules(
    candidate: Image.Image,
    blueprint: QRBlueprint,
    *,
    center_scale: float,
    incorrect_only: bool = False,
    preserve_tone: bool = False,
    confidence_margin: float = 0.0,
) -> Image.Image:
    """Lock functional modules and restore data-module centers over the artwork."""
    if not 0.0 <= center_scale <= 1.0:
        raise ValueError("center_scale must be between 0 and 1")
    if not 0.0 <= confidence_margin < 128.0:
        raise ValueError("confidence_margin must be between 0 (inclusive) and 128 (exclusive)")

    source = np.asarray(candidate.convert("RGB").resize(blueprint.image.size)).copy()
    gray = np.asarray(Image.fromarray(source).convert("L"), dtype=np.float32)
    count = blueprint.matrix.shape[0]
    protected = functional_pattern_mask(blueprint)

    for row in range(count):
        y0 = round(row * source.shape[0] / count)
        y1 = max(y0 + 1, round((row + 1) * source.shape[0] / count))
        for col in range(count):
            x0 = round(col * source.shape[1] / count)
            x1 = max(x0 + 1, round((col + 1) * source.shape[1] / count))
            target = 0 if blueprint.matrix[row, col] else 255
            if protected[row, col]:
                source[y0:y1, x0:x1] = target
                continue
            module_mean = float(gray[y0:y1, x0:x1].mean())
            target_is_dark = bool(blueprint.matrix[row, col])
            needs_repair = (
                module_mean >= 128 - confidence_margin
                if target_is_dark
                else module_mean < 128 + confidence_margin
            )
            if center_scale == 0 or (incorrect_only and not needs_repair):
                continue
            width = max(1, round((x1 - x0) * center_scale))
            height = max(1, round((y1 - y0) * center_scale))
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            rx0 = max(x0, cx - width // 2)
            ry0 = max(y0, cy - height // 2)
            rx1 = min(x1, rx0 + width)
            ry1 = min(y1, ry0 + height)
            if preserve_tone:
                region = source[ry0:ry1, rx0:rx1].astype(np.float32)
                if target == 0:
                    region *= 0.25
                else:
                    region = 255 - (255 - region) * 0.25
                source[ry0:ry1, rx0:rx1] = np.rint(region).astype(np.uint8)
            else:
                source[ry0:ry1, rx0:rx1] = target
    return Image.fromarray(source)
