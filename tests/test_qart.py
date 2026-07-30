import numpy as np
import pytest
from PIL import Image

from prooftag_qr.domain import ValidationRecord
from prooftag_qr.qart import align_qart_for_diffqrcoder, build_qart_target


def test_qart_alignment_preserves_modules_and_uses_paper_crop_geometry():
    version = 3
    module_size = 4
    core_modules = 17 + 4 * version
    raw_modules = core_modules + 20
    raw = np.full(
        (raw_modules * module_size, raw_modules * module_size, 3),
        255,
        dtype=np.uint8,
    )
    border_px = 10 * module_size
    raw[
        border_px : border_px + module_size,
        border_px : border_px + module_size,
    ] = 0

    blueprint = align_qart_for_diffqrcoder(
        Image.fromarray(raw),
        version=version,
        module_size=module_size,
        padding_px=15,
    )

    assert blueprint.image.size == (
        core_modules * module_size + 30,
        core_modules * module_size + 30,
    )
    assert blueprint.matrix.shape == (core_modules + 8, core_modules + 8)
    assert blueprint.matrix[4, 4] == 1
    assert not blueprint.matrix[:4].any()


def test_qart_target_refuses_candidates_not_read_by_every_original_decoder(
    monkeypatch,
):
    version = 3
    module_size = 4
    raw_modules = 17 + 4 * version + 20

    def fake_run(command, **_):
        Image.new(
            "RGB",
            (raw_modules * module_size, raw_modules * module_size),
            "white",
        ).save(command[5])

    class Validator:
        def validate(self, *_args, **_kwargs):
            return [
                ValidationRecord(
                    decoder="decoder-a",
                    scenario="original",
                    success=True,
                    exact_payload_match=True,
                    latency_ms=1,
                ),
                ValidationRecord(
                    decoder="decoder-b",
                    scenario="original",
                    success=False,
                    exact_payload_match=False,
                    latency_ms=1,
                ),
            ]

    monkeypatch.setattr("prooftag_qr.qart.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="no QArt target passed every decoder"):
        build_qart_target(
            Image.new("RGB", (146, 146), "gray"),
            "https://pt.ag/t/1",
            version=version,
            module_size=module_size,
            padding_px=15,
            thresholds=(96, 128),
            validator=Validator(),
        )
