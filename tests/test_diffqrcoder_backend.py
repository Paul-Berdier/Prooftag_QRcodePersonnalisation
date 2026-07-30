import numpy as np
from PIL import Image

from prooftag_qr.diffqrcoder_backend import (
    UpstreamDiffQRCoderBackend,
    _control_target_center_error_rate,
    _install_partial_schedule,
)
from prooftag_qr.qr import generate_diffqrcoder_qr


def _reference_artwork(size: int = 736) -> Image.Image:
    x = np.linspace(0, 255, size, dtype=np.uint8)
    y = np.linspace(255, 0, size, dtype=np.uint8)
    red = np.broadcast_to(x, (size, size))
    green = np.broadcast_to(y[:, None], (size, size))
    blue = np.full((size, size), 127, dtype=np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=2), mode="RGB")


def test_stage2_target_is_the_exact_binary_qr_not_a_visual_proxy():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/1",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    reference = _reference_artwork()
    backend = object.__new__(UpstreamDiffQRCoderBackend)

    target = backend._stage2_target(reference, blueprint)

    assert target.size == blueprint.image.size
    assert np.array_equal(np.asarray(target), np.asarray(blueprint.image.convert("RGB")))
    assert not np.array_equal(np.asarray(target), np.asarray(reference))
    assert _control_target_center_error_rate(
        target,
        blueprint,
        padding_px=78,
        module_size=20,
    ) == 0.0

def test_partial_stage2_schedule_avoids_custom_timesteps_and_keeps_stride():
    class Scheduler:
        num_inference_steps = None
        timesteps = []

        def set_timesteps(self, num_inference_steps, *args, **kwargs):
            self.num_inference_steps = num_inference_steps
            self.timesteps = list(range(num_inference_steps))

    class Pipeline:
        scheduler = Scheduler()

    pipe = Pipeline()
    original = pipe.scheduler.set_timesteps

    with _install_partial_schedule(
        pipe,
        base_steps=40,
        effective_steps=20,
    ):
        pipe.scheduler.set_timesteps(20, device="cuda")
        assert pipe.scheduler.num_inference_steps == 40
        assert pipe.scheduler.timesteps == list(range(20, 40))

    assert pipe.scheduler.set_timesteps == original
