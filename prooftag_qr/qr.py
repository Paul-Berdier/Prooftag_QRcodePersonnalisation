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
    errors = module_error_map(candidate, blueprint)
    return float(errors.mean())


def module_error_map(candidate: Image.Image, blueprint: QRBlueprint) -> np.ndarray:
    """Return the binary module error map on the delivered canvas."""
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
    return np.not_equal(predicted, blueprint.matrix)


def module_error_breakdown(
    candidate: Image.Image,
    blueprint: QRBlueprint,
) -> dict[str, float]:
    """Separate QR-core failures from quiet-zone failures.

    SRPG and SR-MPGD optimize the border-free QR core, but a scanner still needs a
    clear quiet zone around the delivered image. Reporting both regions prevents a
    good core score from hiding a painted or saturated margin.
    """
    errors = module_error_map(candidate, blueprint)
    border = int(blueprint.border)
    if border == 0:
        return {
            "overall": float(errors.mean()),
            "core": float(errors.mean()),
            "quiet_zone": 0.0,
        }
    core = errors[border:-border, border:-border]
    quiet_zone = errors.copy()
    quiet_zone[border:-border, border:-border] = False
    quiet_zone_mask = np.ones_like(errors, dtype=bool)
    quiet_zone_mask[border:-border, border:-border] = False
    return {
        "overall": float(errors.mean()),
        "core": float(core.mean()),
        "quiet_zone": float(quiet_zone[quiet_zone_mask].mean()),
    }


def restore_quiet_zone(
    candidate: Image.Image,
    blueprint: QRBlueprint,
    *,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Restore only the mandatory light margin around an artistic QR core.

    The diffusion model may paint through the quiet zone because the paper losses
    intentionally crop it. The production image must nevertheless present a uniform
    light margin to real decoders. The QR core is copied byte-for-byte.
    """
    source = candidate.convert("RGB")
    border = int(blueprint.border)
    count = int(blueprint.matrix.shape[0])
    if border <= 0:
        return source.copy()
    if 2 * border >= count:
        raise ValueError("invalid QR border")
    left = round(border * source.width / count)
    right = round((count - border) * source.width / count)
    top = round(border * source.height / count)
    bottom = round((count - border) * source.height / count)
    output = Image.new("RGB", source.size, color)
    output.paste(source.crop((left, top, right, bottom)), (left, top))
    return output


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


def _tone_region(region: np.ndarray, target: int, factor: float) -> np.ndarray:
    """Move RGB values toward a QR tone while retaining hue and local texture."""
    source = region.astype(np.float32)
    if target == 0:
        return source * factor
    return 255 - (255 - source) * factor


def _feather_mask(
    height: int, width: int, edge_feather: float, rounded_edges: bool
) -> np.ndarray:
    """Return a smooth mask whose edges blend a repaired center into the artwork."""
    if edge_feather == 0:
        return np.ones((height, width, 1), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    if rounded_edges:
        normalized_x = np.abs((xx + 0.5 - width / 2) / (width / 2))
        normalized_y = np.abs((yy + 0.5 - height / 2) / (height / 2))
        # A fourth-order superellipse looks organic while retaining enough module area.
        radius = (normalized_x**4 + normalized_y**4) ** 0.25
        alpha = np.clip((1.0 - radius) / edge_feather, 0.0, 1.0)
    else:
        distance = np.minimum.reduce(
            (xx + 0.5, width - xx - 0.5, yy + 0.5, height - yy - 0.5)
        )
        feather_pixels = max(1.0, min(height, width) * edge_feather)
        alpha = np.clip(distance / feather_pixels, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha[..., None].astype(np.float32)


def repair_qr_modules(
    candidate: Image.Image,
    blueprint: QRBlueprint,
    *,
    center_scale: float,
    incorrect_only: bool = False,
    preserve_tone: bool = False,
    confidence_margin: float = 0.0,
    tone_factor: float = 0.25,
    edge_feather: float = 0.0,
    preserve_functional_tone: bool = False,
    rounded_edges: bool = False,
) -> Image.Image:
    """Lock functional modules and restore data-module centers over the artwork."""
    if not 0.0 <= center_scale <= 1.0:
        raise ValueError("center_scale must be between 0 and 1")
    if not 0.0 <= confidence_margin < 128.0:
        raise ValueError("confidence_margin must be between 0 (inclusive) and 128 (exclusive)")
    if not 0.0 <= tone_factor <= 1.0:
        raise ValueError("tone_factor must be between 0 and 1")
    if not 0.0 <= edge_feather <= 0.5:
        raise ValueError("edge_feather must be between 0 and 0.5")

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
                if preserve_functional_tone:
                    source[y0:y1, x0:x1] = np.rint(
                        _tone_region(source[y0:y1, x0:x1], target, min(tone_factor, 0.12))
                    ).astype(np.uint8)
                else:
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
                toned = _tone_region(region, target, tone_factor)
                alpha = _feather_mask(
                    region.shape[0], region.shape[1], edge_feather, rounded_edges
                )
                source[ry0:ry1, rx0:rx1] = np.rint(
                    region * (1.0 - alpha) + toned * alpha
                ).astype(np.uint8)
            else:
                source[ry0:ry1, rx0:rx1] = target
    return Image.fromarray(source)
