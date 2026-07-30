from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .blueprints import canonical_url_match, reference_cost
from .qr import QRBlueprint
from .validation import QRValidator


@dataclass(frozen=True, slots=True)
class QArtTarget:
    image: Image.Image
    blueprint: QRBlueprint
    threshold: int
    scan_pass_rate: float
    original_passed: int
    original_total: int
    reference_cost: float


def align_qart_for_diffqrcoder(
    qart_image: Image.Image,
    *,
    version: int,
    module_size: int,
    padding_px: int,
    border_modules: int = 10,
) -> QRBlueprint:
    """Crop the public QArt border and align its core to DiffQRCoder's 736 px canvas."""
    core_modules = 17 + 4 * version
    raw_modules = core_modules + 2 * border_modules
    expected_size = raw_modules * module_size
    if qart_image.size != (expected_size, expected_size):
        raise ValueError(
            f"QArt output is {qart_image.size}, expected "
            f"{(expected_size, expected_size)}"
        )
    border_px = border_modules * module_size
    core_size = core_modules * module_size
    core = qart_image.convert("RGB").crop(
        (border_px, border_px, border_px + core_size, border_px + core_size)
    )
    canvas_size = core_size + 2 * padding_px
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(core, (padding_px, padding_px))

    gray = np.asarray(core.convert("L"), dtype=np.float32)
    core_matrix = np.zeros((core_modules, core_modules), dtype=np.uint8)
    for row in range(core_modules):
        for column in range(core_modules):
            region = gray[
                row * module_size : (row + 1) * module_size,
                column * module_size : (column + 1) * module_size,
            ]
            core_matrix[row, column] = int(region.mean() < 128)
    matrix = np.pad(core_matrix, 4, constant_values=0)
    return QRBlueprint(
        image=canvas,
        matrix=matrix,
        version=version,
        border=4,
    )


def build_qart_target(
    reference: Image.Image,
    payload: str,
    *,
    version: int,
    module_size: int,
    padding_px: int,
    thresholds: tuple[int, ...],
    executable: str = "/usr/local/bin/qart",
    validator: QRValidator | None = None,
) -> QArtTarget:
    """Build and validate real Reed-Solomon QArt candidates.

    The public QArt implementation appends a URL fragment. The target therefore
    uses an explicit canonical-URL contract and must never be reported as an
    exact byte-for-byte payload.
    """
    if "#" in payload:
        raise ValueError("QArt requires a URL without an existing fragment")
    if not thresholds:
        raise ValueError("at least one QArt threshold is required")
    validator = validator or QRValidator()
    candidates: list[QArtTarget] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="prooftag-qart-") as directory:
        root = Path(directory)
        reference_path = root / "stage1.png"
        reference.convert("RGB").save(reference_path)
        for threshold in thresholds:
            output_path = root / f"qart-{threshold}.png"
            command = [
                executable,
                "build",
                str(version),
                payload,
                str(reference_path),
                str(output_path),
                "--module-size",
                str(module_size),
                "--threshold",
                str(threshold),
                "--benchmark",
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                with Image.open(output_path) as rendered:
                    blueprint = align_qart_for_diffqrcoder(
                        rendered,
                        version=version,
                        module_size=module_size,
                        padding_px=padding_px,
                    )
                records = validator.validate(
                    blueprint.image,
                    payload,
                    matcher=canonical_url_match,
                    match_mode="canonical_url_without_fragment",
                )
                passed = sum(record.exact_payload_match for record in records)
                originals = [
                    record for record in records if record.scenario == "original"
                ]
                candidates.append(
                    QArtTarget(
                        image=blueprint.image,
                        blueprint=blueprint,
                        threshold=threshold,
                        scan_pass_rate=passed / len(records) if records else 0.0,
                        original_passed=sum(
                            record.exact_payload_match for record in originals
                        ),
                        original_total=len(originals),
                        reference_cost=reference_cost(
                            blueprint.image,
                            reference.resize(blueprint.image.size),
                        ),
                    )
                )
            except Exception as exc:
                errors.append(f"threshold {threshold}: {type(exc).__name__}: {exc}")
    if not candidates:
        raise RuntimeError("all QArt candidates failed: " + " | ".join(errors))
    winner = max(
        candidates,
        key=lambda item: (
            item.original_total > 0
            and item.original_passed == item.original_total,
            item.scan_pass_rate,
            -item.reference_cost,
        ),
    )
    if (
        winner.original_total == 0
        or winner.original_passed != winner.original_total
    ):
        outcomes = ", ".join(
            f"{item.threshold}:{item.original_passed}/{item.original_total}"
            for item in candidates
        )
        raise RuntimeError(
            "no QArt target passed every decoder on the original image "
            f"({outcomes})"
        )
    return winner
