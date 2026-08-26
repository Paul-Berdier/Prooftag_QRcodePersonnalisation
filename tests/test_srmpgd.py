from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from prooftag_qr.qr import generate_qr
from prooftag_qr.srmpgd import SRMPGDConfig


def test_robust_srmpgd_loss_keeps_every_scan_transform_differentiable():
    torch = pytest.importorskip("torch")
    from prooftag_qr.srmpgd import _robust_scanning_loss

    images = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    images.requires_grad_(True)
    target = torch.zeros((1, 1, 32, 32))
    calls = []

    def loss_fn(value, expected):
        calls.append(tuple(value.shape))
        return (value.mean() - expected.mean()).square()

    loss, components = _robust_scanning_loss(
        images,
        target,
        loss_fn,
        SRMPGDConfig(
            robust_blur_weight=1.0,
            robust_downscale_weight=1.0,
            robust_brightness_weight=1.0,
            robust_contrast_weight=1.0,
        ),
    )
    loss.backward()

    assert len(calls) == 6
    assert all(components[name] is not None for name in components)
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


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
    assert result.steps[0].applied_step_rms <= 0.02
    assert result.steps[0].eligible_for_selection is True
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
    assert len(validated_corners) == 1
    assert min(validated_corners[0]) >= 229


def test_srmpgd_iteration_zero_keeps_the_exact_stage2_raster(monkeypatch):
    torch = pytest.importorskip("torch")
    from prooftag_qr import srmpgd
    from prooftag_qr.quality import image_sha256

    class FakeVAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def decode(self, latent, **kwargs):
            # Deliberately reconstruct something different from the Stage-2 raster.
            return (torch.zeros_like(latent) * self.anchor,)

    class FakeImageProcessor:
        def postprocess(self, image, **kwargs):
            array = (image[0].detach() / 2 + 0.5).permute(1, 2, 0)
            return [
                Image.fromarray(
                    np.rint(array.cpu().numpy() * 255).astype(np.uint8),
                    mode="RGB",
                )
            ]

    class ZeroLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return image.mean().reshape(1, 1, 1, 1) * 0

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: ZeroLPIPS())
    blueprint = generate_qr("https://example.test/exact-stage2-raster", "M", size=128)
    stage2_image = blueprint.image.convert("RGB")
    stage2_hash = image_sha256(stage2_image)
    validated_hashes = []

    result = srmpgd.run_srmpgd(
        SimpleNamespace(vae=FakeVAE(), image_processor=FakeImageProcessor()),
        torch.zeros((1, 3, 128, 128)),
        blueprint,
        SRMPGDConfig(max_iterations=3, crop_padding_px=0),
        initial_image=stage2_image,
        validation_callback=lambda image, iteration: (
            validated_hashes.append(image_sha256(image))
            or {"passed": 2, "total": 2, "strict_all": True}
        ),
    )

    assert result.selected_iteration == 0
    assert result.stop_reason == "strict_validation_passed"
    assert image_sha256(result.image) == stage2_hash
    assert validated_hashes == [stage2_hash]


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


def test_srmpgd_rejects_a_tainted_iteration_and_keeps_stage2_state_zero(monkeypatch):
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

    change_calls = 0

    def changes(image, reference):
        nonlocal change_calls
        change_calls += 1
        return {
            "changed_pixel_ratio": 0.0,
            "mean_absolute_change": 0.0,
            "clipped_pixel_ratio_increase": 0.0,
            "rgb_clipped_channel_ratio_increase": 0.0,
            "saturation_mean_increase": 0.0 if change_calls == 1 else 0.5,
            "high_saturation_ratio_increase": 0.0,
        }

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: ZeroLPIPS())
    monkeypatch.setattr(srmpgd, "image_change_metrics", changes)
    blueprint = generate_qr("https://example.test/guard", "M", size=128)
    reference = np.asarray(blueprint.image, dtype=np.float32) / 127.5 - 1
    latent = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)

    result = srmpgd.run_srmpgd(
        SimpleNamespace(vae=FakeVAE(), image_processor=FakeImageProcessor()),
        latent,
        blueprint,
        SRMPGDConfig(
            max_iterations=4,
            step_size=1000.0,
            max_step_rms=0.02,
            max_saturation_mean_increase=0.04,
            crop_padding_px=0,
        ),
        scanning_loss=lambda image, target: (image.mean() - target.mean()).square(),
        validation_callback=lambda image, iteration: {"passed": 0, "total": 2},
    )

    assert len(result.steps) == 2
    assert result.steps[1].aesthetic_guard_passed is False
    assert result.steps[1].eligible_for_selection is False
    assert result.selected_iteration == 0
    assert result.stop_reason == "aesthetic_guard_failed_at_iteration_1"


def test_paper_equations_runs_fixed_iterations_without_guard_or_step_clipping(monkeypatch):
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
            return [
                Image.fromarray(
                    np.rint(array.cpu().numpy() * 255).astype(np.uint8),
                    mode="RGB",
                )
            ]

    class ZeroLPIPS(torch.nn.Module):
        def forward(self, image, reference):
            return (image - reference).square().mean().reshape(1, 1, 1, 1)

    monkeypatch.setattr(srmpgd, "_load_lpips", lambda pipeline, device, net: ZeroLPIPS())
    blueprint = generate_qr("https://example.test/paper-equations", "M", size=128)
    latent = torch.zeros((1, 3, 128, 128))
    stage2_image = Image.new("RGB", (128, 128), (127, 127, 127))

    result = srmpgd.run_srmpgd(
        SimpleNamespace(vae=FakeVAE(), image_processor=FakeImageProcessor()),
        latent,
        blueprint,
        SRMPGDConfig(
            protocol="paper_equations",
            max_iterations=2,
            step_size=1000.0,
            lpips_weight=0.01,
            crop_padding_px=0,
            max_initial_module_error_rate=0.0,
            max_step_rms=1e-6,
            max_total_delta_rms=2e-6,
        ),
        initial_image=stage2_image,
        # Paper mode must ignore the public surrogate and use Eq. 1-6 locally.
        scanning_loss=lambda image, target: image.sum() * 0,
        validation_callback=lambda image, iteration: {
            "passed": 2,
            "total": 2,
            "strict_all": True,
        },
    )

    assert len(result.steps) == 3
    assert result.selected_iteration == 2
    assert result.stop_reason == "max_iterations"
    assert result.steps[0].step_scale == 1.0
    assert result.steps[0].applied_step_rms == pytest.approx(
        result.steps[0].next_step_rms
    )
    assert result.steps[0].applied_step_rms > 1e-6
    assert result.steps[1].latent_delta_rms > 2e-6


def test_diffqrcoder_v3_crop_has_integer_module_geometry():
    from prooftag_qr.qr import generate_diffqrcoder_qr
    from prooftag_qr.srmpgd import _module_error_for_canvas

    blueprint = generate_diffqrcoder_qr(
        "https://ptag.io/t/e032",
        "M",
        version=3,
        mask_pattern=4,
        module_size=20,
        border=4,
    )
    canvas = blueprint.image.resize((736, 736), Image.Resampling.NEAREST)
    core_modules = blueprint.matrix.shape[0] - 2 * blueprint.border

    assert blueprint.matrix.shape == (37, 37)
    assert core_modules == 29
    assert 736 - 2 * 78 == 580
    assert 580 // core_modules == 20
    assert _module_error_for_canvas(
        canvas,
        blueprint,
        crop_padding_px=78,
    ) == pytest.approx(0.0)


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
        (SRMPGDConfig(max_step_rms=0), "max_step_rms"),
        (SRMPGDConfig(max_total_delta_rms=0), "max_total_delta_rms"),
        (
            SRMPGDConfig(max_step_rms=0.07, max_total_delta_rms=0.06),
            "max_step_rms",
        ),
        (
            SRMPGDConfig(min_relative_module_improvement=1.1),
            "min_relative_module_improvement",
        ),
        (SRMPGDConfig(max_mean_absolute_change=1.1), "max_mean_absolute_change"),
        (SRMPGDConfig(robust_blur_weight=-1), "robust_blur_weight"),
        (SRMPGDConfig(robust_blur_kernel=4), "robust_blur_kernel"),
        (SRMPGDConfig(robust_downscale_factor=0), "robust_downscale_factor"),
    ],
)
def test_srmpgd_rejects_invalid_configuration(config, message):
    torch = pytest.importorskip("torch")
    from prooftag_qr.srmpgd import run_srmpgd

    with pytest.raises(ValueError, match=message):
        run_srmpgd(SimpleNamespace(), torch.zeros((1, 4, 8, 8)), SimpleNamespace(), config)
