from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import prooftag_qr.diffqrcoder_backend as backend_module
from prooftag_qr.config import Settings
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


def test_stage2_state_export_and_import_are_hash_verified():
    torch = pytest.importorskip("torch")
    latent = torch.arange(16, dtype=torch.float16).reshape(1, 1, 4, 4)
    image = Image.new("RGB", (16, 16), "navy")
    source = UpstreamDiffQRCoderBackend(Settings(device="cpu"))
    source._last_stage2_state = {
        "latent": latent,
        "image": image,
        "reference": image,
        "control": object(),
        "diagnostics": {"diffqrcoder_stage2_reused": 0.0},
    }

    state = source.export_stage2_state()
    assert state is not None
    state["source_run_id"] = "run-srpg"
    state["source_method_id"] = "diffqrcoder_srpg"

    target = UpstreamDiffQRCoderBackend(Settings(device="cpu"))
    target.import_stage2_state(state)
    assert target._stage2_override["latent_sha256"] == state["latent_sha256"]
    assert target._stage2_override["source_run_id"] == "run-srpg"

    corrupted = {**state, "latent": state["latent"] + 1}
    with pytest.raises(RuntimeError, match="latent hash mismatch"):
        target.import_stage2_state(corrupted)


def test_stage2_target_is_the_exact_binary_qr_not_a_visual_proxy():
    blueprint = generate_diffqrcoder_qr(
        "https://pt.ag/t/1",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
    )
    reference = _reference_artwork()
    backend = UpstreamDiffQRCoderBackend(
        Settings(diffqrcoder_stage2_target_mode="binary_exact")
    )

    target = backend._stage2_target(
        reference,
        blueprint,
        "https://pt.ag/t/1",
    )

    assert target.image.size == blueprint.image.size
    assert target.match_mode == "exact"
    assert np.array_equal(
        np.asarray(target.image),
        np.asarray(blueprint.image.convert("RGB")),
    )
    assert not np.array_equal(np.asarray(target.image), np.asarray(reference))
    assert _control_target_center_error_rate(
        target.image,
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


def test_color_guard_distinguishes_warning_from_destructive_divergence(monkeypatch):
    change = {
        "changed_pixel_ratio": 0.99,
        "mean_absolute_change": 0.36,
        "clipped_pixel_ratio_increase": 0.06,
        "rgb_clipped_channel_ratio_increase": 0.03,
        "saturation_mean_increase": 0.09,
        "high_saturation_ratio_increase": 0.06,
    }
    quality = {
        "clipped_pixel_ratio": 0.06,
        "rgb_clipped_channel_ratio": 0.03,
        "saturation_mean": 0.50,
        "saturation_p95": 0.95,
    }
    monkeypatch.setattr(backend_module, "image_change_metrics", lambda *_: change)
    monkeypatch.setattr(backend_module, "image_quality_metrics", lambda *_: quality)
    backend = UpstreamDiffQRCoderBackend(Settings())
    image = Image.new("RGB", (32, 32), "red")

    backend._record_divergence_guard(image, image)

    assert backend.diagnostics()["diffqrcoder_guard_warning"] == 1.0
    assert backend.diagnostics()["diffqrcoder_guard_diverged"] == 0.0
    assert backend.candidate_guard_ok("srpg") is True

    change["saturation_mean_increase"] = 0.25
    backend._record_divergence_guard(image, image)

    assert backend.diagnostics()["diffqrcoder_guard_diverged"] == 1.0
    assert backend.candidate_guard_ok("srpg") is False


def test_hard_color_guard_must_be_at_least_the_warning_threshold():
    with pytest.raises(ValueError, match="hard saturation mean guard"):
        Settings(
            diffqrcoder_guard_max_saturation_mean_increase=0.25,
            diffqrcoder_guard_hard_max_saturation_mean_increase=0.20,
        )


@pytest.mark.parametrize(
    ("selected_iteration", "expected_variant"),
    [(0, "srpg"), (2, "srmpgd")],
)
def test_srmpgd_iteration_zero_is_reported_as_srpg(
    selected_iteration, expected_variant
):
    backend = UpstreamDiffQRCoderBackend(
        Settings(srpg_enabled=True, srmpgd_enabled=True)
    )
    image = Image.new("RGB", (32, 32), "navy")

    def fake_stage2(*_args, **_kwargs):
        backend._srmpgd_selected_iteration = selected_iteration
        return image.copy()

    backend._run_stage2 = fake_stage2
    variants = list(
        backend.variants(
            image,
            SimpleNamespace(),
            request=SimpleNamespace(),
            seed=41,
        )
    )

    assert [name for name, _ in variants] == ["raw", expected_variant]
