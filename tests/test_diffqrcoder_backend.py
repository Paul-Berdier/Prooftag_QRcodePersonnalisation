import numpy as np
import pytest
from PIL import Image

from prooftag_qr.diffqrcoder_backend import (
    _preserve_partial_schedule_stride,
    _qart_center_error_rate,
    build_paper_qart_target,
)
from prooftag_qr.qr import (
    functional_pattern_mask,
    generate_diffqrcoder_qr,
)


def _reference_artwork(size: int = 736) -> Image.Image:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    red = np.broadcast_to(x, (size, size))
    green = np.broadcast_to(y[:, None], (size, size))
    blue = np.full((size, size), 127, dtype=np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=2), mode="RGB")


def test_reconstructed_qart_preserves_canvas_and_corrects_every_module_center():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/1",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    reference = _reference_artwork()

    target = build_paper_qart_target(
        reference,
        blueprint,
        padding_px=78,
        module_size=20,
        center_fraction=0.40,
        dark_target=0.25,
        light_target=0.75,
    )

    assert target.size == reference.size == (736, 736)
    assert _qart_center_error_rate(
        target,
        blueprint,
        padding_px=78,
        module_size=20,
    ) == 0.0

    # The reconstructed target must not create a white frame around the QR core.
    assert np.array_equal(
        np.asarray(target)[:78],
        np.asarray(reference)[:78],
    )


def test_reconstructed_qart_copies_functional_modules_exactly():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/2",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    target = build_paper_qart_target(
        _reference_artwork(),
        blueprint,
        padding_px=78,
        module_size=20,
        center_fraction=0.40,
        dark_target=0.25,
        light_target=0.75,
    )
    pixels = np.asarray(target)
    border = blueprint.border
    matrix = blueprint.matrix[border:-border, border:-border]
    functional = functional_pattern_mask(blueprint)[border:-border, border:-border]

    for row, col in np.argwhere(functional):
        y0 = 78 + int(row) * 20
        x0 = 78 + int(col) * 20
        expected = 0 if matrix[row, col] else 255
        assert np.all(pixels[y0 : y0 + 20, x0 : x0 + 20] == expected)


def test_reconstructed_qart_rejects_an_unaligned_stage1_canvas():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/3",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )

    with pytest.raises(ValueError, match="QArt geometry mismatch"):
        build_paper_qart_target(
            Image.new("RGB", (512, 512), "white"),
            blueprint,
            padding_px=78,
            module_size=20,
            center_fraction=0.40,
            dark_target=0.25,
            light_target=0.75,
        )


def test_partial_stage2_schedule_keeps_the_original_ddim_stride():
    class Scheduler:
        num_inference_steps = None

        def set_timesteps(self, *args, **kwargs):
            self.num_inference_steps = len(kwargs["timesteps"])

    class Pipeline:
        scheduler = Scheduler()

    pipe = Pipeline()
    original = pipe.scheduler.set_timesteps

    with _preserve_partial_schedule_stride(
        pipe,
        base_steps=40,
        timesteps=list(range(20)),
    ):
        pipe.scheduler.set_timesteps(timesteps=list(range(20)), device="cuda")
        assert pipe.scheduler.num_inference_steps == 40

    assert pipe.scheduler.set_timesteps == original
