from dataclasses import dataclass

import numpy as np
import qrcode
from PIL import Image
from qrcode.constants import ERROR_CORRECT_H, ERROR_CORRECT_M, ERROR_CORRECT_Q

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
