from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from prooftag_qr.qr import generate_qr
from prooftag_qr.srmpgd import SRMPGDConfig


def test_srmpgd_uses_exact_latent_original_qr_and_stops_on_strict_validation(monkeypatch):
    torch = pytest.importorskip("torch")
    from prooftag_qr import srmpgd

    class FakeVAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def encode(self, image):
            raise AssertionError("paper-aligned SR-MPGD must not re-encode a PNG")

        def decode(self, latent, **kwargs):
            return (latent * self.anchor,)

    class FakeImageProcessor:
        def postprocess(self, image, **kwargs):
            array = (image[0].detach().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0)
            array = np.rint(array.cpu().numpy() * 255).astype(np.uint8)
            return [Image.fromarray(array, mode="RGB")]

    class FakePipeline:
        def __init__(self):
            self.vae = FakeVAE()
            self.image_processor = FakeImageProcessor()

    class FakeLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return (image - reference).square().mean().reshape(1, 1, 1, 1)

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: FakeLPIPS())
    blueprint = generate_qr("https://example.prooftag.test/t/srmpgd", "M", size=128)
    reference = np.asarray(blueprint.image, dtype=np.float32) / 127.5 - 1
    latent = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)
    seen = []
    scanning_loss_calls = []

    def official_scanning_loss(image, qrcode):
        scanning_loss_calls.append((tuple(image.shape), tuple(qrcode.shape)))
        return (image.mean() - qrcode.mean()).square()

    def validate(image, iteration):
        seen.append(iteration)
        return {
            "passed": 2 if iteration == 1 else 0,
            "total": 2,
            "strict_all": iteration == 1,
            "worst_decoder_pass_rate": float(iteration == 1),
            "worst_scenario_pass_rate": float(iteration == 1),
        }

    result = srmpgd.run_srmpgd(
        FakePipeline(),
        latent,
        blueprint,
        SRMPGDConfig(
            max_iterations=3,
            step_size=0.01,
            lpips_weight=0.01,
            crop_padding_px=0,
        ),
        scanning_loss=official_scanning_loss,
        validation_callback=validate,
    )

    assert seen == [0, 1]
    assert len(result.steps) == 2
    assert result.selected_iteration == 1
    assert result.stop_reason == "strict_validation_passed"
    assert result.steps[0].gradient_rms is not None
    assert result.steps[0].lpips_loss == pytest.approx(0.0, abs=1e-7)
    assert result.image.size == blueprint.image.size
    assert scanning_loss_calls == [((1, 3, 128, 128), (1, 1, 128, 128))] * 2


def test_srmpgd_keeps_state_zero_when_the_gradient_is_not_finite(monkeypatch):
    torch = pytest.importorskip("torch")
    from prooftag_qr import srmpgd

    class FakeVAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def decode(self, latent, **kwargs):
            return (latent * self.anchor,)

    class FakeImageProcessor:
        def postprocess(self, image, **kwargs):
            array = image[0].detach().permute(1, 2, 0).clamp(0, 1)
            return [Image.fromarray(np.uint8(array.cpu().numpy() * 255), mode="RGB")]

    class ZeroLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return image.mean().reshape(1, 1, 1, 1) * 0

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: ZeroLPIPS())
    blueprint = generate_qr("https://example.test/nan", "M", size=128)
    latent = torch.full((1, 3, 128, 128), 0.5)

    validated_corners = []

    def validate(image, iteration):
        validated_corners.append(tuple(np.asarray(image)[0, 0]))
        return {"passed": 0, "total": 2}

    result = srmpgd.run_srmpgd(
        SimpleNamespace(vae=FakeVAE(), image_processor=FakeImageProcessor()),
        latent,
        blueprint,
        SRMPGDConfig(max_iterations=3, step_size=100.0),
        scanning_loss=lambda image, target: (image * torch.tensor(float("nan"))).mean(),
        validation_callback=validate,
    )

    assert len(result.steps) == 1
    assert result.selected_iteration == 0
    assert result.stop_reason == "non_finite_gradient_at_iteration_0"
    assert result.steps[0].gradient_rms is None
    assert validated_corners == [(255, 255, 255)]


def test_srmpgd_does_not_attempt_to_reconstruct_a_stage2_far_from_the_qr(monkeypatch):
    torch = pytest.importorskip("torch")
    from prooftag_qr import srmpgd

    class FakeVAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def decode(self, latent, **kwargs):
            return (latent * self.anchor,)

    class FakeImageProcessor:
        def postprocess(self, image, **kwargs):
            array = (image[0].detach().clamp(-1, 1) / 2 + 0.5).permute(1, 2, 0)
            return [Image.fromarray(np.uint8(array.cpu().numpy() * 255), mode="RGB")]

    class ZeroLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return image.mean().reshape(1, 1, 1, 1) * 0

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: ZeroLPIPS())
    blueprint = generate_qr("https://example.test/srmpgd-precondition", "M", size=128)
    # A flat gray image is far outside the local post-processing regime.
    latent = torch.zeros((1, 3, 128, 128))

    result = srmpgd.run_srmpgd(
        SimpleNamespace(vae=FakeVAE(), image_processor=FakeImageProcessor()),
        latent,
        blueprint,
        SRMPGDConfig(max_iterations=20, max_initial_module_error_rate=0.10),
        validation_callback=lambda image, iteration: {"passed": 0, "total": 2},
    )

    assert len(result.steps) == 1
    assert result.selected_iteration == 0
    assert result.stop_reason == "initial_module_error_rate_above_limit"
    assert result.steps[0].gradient_rms is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (SRMPGDConfig(max_iterations=0), "max_iterations"),
        (SRMPGDConfig(step_size=0), "step_size"),
        (SRMPGDConfig(lpips_weight=-1), "lpips_weight"),
        (SRMPGDConfig(lpips_net="unknown"), "lpips_net"),
        (SRMPGDConfig(crop_padding_px=-2), "crop_padding"),
        (
            SRMPGDConfig(max_initial_module_error_rate=1.1),
            "max_initial_module_error_rate",
        ),
    ],
)
def test_srmpgd_rejects_invalid_configuration(config, message):
    torch = pytest.importorskip("torch")
    from prooftag_qr.srmpgd import run_srmpgd

    with pytest.raises(ValueError, match=message):
        run_srmpgd(SimpleNamespace(), torch.zeros((1, 4, 8, 8)), SimpleNamespace(), config)
