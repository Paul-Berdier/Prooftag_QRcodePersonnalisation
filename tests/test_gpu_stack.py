import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from prooftag_qr import backends
from prooftag_qr.backends import (
    GLOBAL_REPAIR_VARIANTS,
    ControlNetBackend,
    GuidedRediffusionResult,
)
from prooftag_qr.config import Settings
from prooftag_qr.qr import generate_qr
from prooftag_qr.schemas import GenerationRequest


def test_gpu_dependencies_are_pinned_to_the_torch_base_image():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = set(project["project"]["optional-dependencies"]["gpu"])
    optimizer_dependencies = set(project["project"]["optional-dependencies"]["optimizer"])

    assert {
        "accelerate==0.34.2",
        "diffusers==0.31.0",
        "huggingface-hub==0.25.2",
        "lpips==0.1.4",
        "safetensors==0.4.5",
        "transformers==4.44.2",
        "torchvision==0.19.1",
    } <= dependencies
    assert {"optuna>=4.4,<5", "scikit-learn>=1.5,<2", "joblib>=1.4,<2"} <= (
        optimizer_dependencies
    )

    dockerfile = Path("Dockerfile").read_text()
    assert "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime" in dockerfile
    assert "from diffusers import ControlNetModel" in dockerfile
    assert "StableDiffusionControlNetImg2ImgPipeline" in dockerfile
    assert "DDIMScheduler" in dockerfile
    assert "TORCH_HOME=/opt/torch-cache" in dockerfile
    assert "lpips.LPIPS(net='alex'" in dockerfile
    assert "TRANSFORMERS_CACHE" not in dockerfile

    settings = Settings()
    request = GenerationRequest(payload="https://example.prooftag.test/t/img2img")
    manifest = Path("deploy/k8s/app-config.yaml").read_text()
    assert settings.controlnet_pipeline_mode == "img2img"
    assert settings.controlnet_model_subfolder == ""
    assert settings.controlnet_conditioning_profile == "binary"
    assert settings.regenerate_before_global_repair is True
    assert settings.guided_rediffusion_enabled is False
    assert settings.guided_rediffusion_steps == 8
    assert settings.guided_rediffusion_strength == 0.30
    assert settings.guided_rediffusion_controlnet_scale == 1.75
    assert settings.guided_rediffusion_mask_dilation_px == 4
    assert settings.guided_rediffusion_mask_feather_px == 4
    assert settings.guided_rediffusion_max_mean_absolute_change == 0.12
    assert settings.guided_rediffusion_min_relative_module_improvement == 0.01
    assert settings.srpg_enabled is False
    assert settings.srpg_steps == 40
    assert settings.srpg_qr_weight == 500.0
    assert settings.srpg_perceptual_weight == 3.0
    assert settings.srpg_min_relative_module_improvement == 0.10
    assert settings.srpg_save_step_previews is False
    assert settings.srpg_preview_interval == 5
    assert settings.srpg_dark_threshold == 0.5
    assert settings.srpg_light_threshold == 0.5
    assert settings.srpg_center_fraction == pytest.approx(1 / 3)
    assert settings.srpg_robust_blur_weight == 0
    assert settings.srpg_robust_blur_kernel == 3
    assert settings.srpg_robust_downscale_weight == 0
    assert settings.srpg_robust_downscale_factor == pytest.approx(0.75)
    assert settings.srpg_robust_brightness_weight == 0
    assert settings.srpg_robust_brightness_low == pytest.approx(0.75)
    assert settings.srpg_robust_brightness_high == pytest.approx(1.25)
    assert settings.srpg_robust_contrast_weight == 0
    assert settings.srpg_robust_contrast_factor == pytest.approx(0.70)
    assert settings.srpg_eta == 0
    assert settings.latent_refinement_enabled is False
    assert settings.latent_refinement_iterations == 8
    assert settings.latent_refinement_learning_rate == 0.02
    assert settings.latent_refinement_preservation_weight == 1.0
    assert settings.latent_refinement_max_latent_delta == 0.10
    assert settings.latent_refinement_max_mean_absolute_change == 0.08
    assert settings.latent_refinement_min_relative_module_improvement == 0.01
    assert request.strength == 0.9
    assert "PROOFTAG_QR_CONTROLNET_PIPELINE_MODE: img2img" in manifest
    assert 'PROOFTAG_QR_CONTROLNET_MODEL_SUBFOLDER: ""' in manifest
    assert "PROOFTAG_QR_CONTROLNET_CONDITIONING_PROFILE: binary" in manifest
    assert 'PROOFTAG_QR_REGENERATE_BEFORE_GLOBAL_REPAIR: "true"' in manifest
    assert 'PROOFTAG_QR_GUIDED_REDIFFUSION_ENABLED: "false"' in manifest
    assert 'PROOFTAG_QR_GUIDED_REDIFFUSION_STRENGTH: "0.30"' in manifest
    assert 'PROOFTAG_QR_SRPG_ENABLED: "false"' in manifest
    assert 'PROOFTAG_QR_SRPG_STEPS: "40"' in manifest
    assert 'PROOFTAG_QR_SRPG_SAVE_STEP_PREVIEWS: "false"' in manifest
    assert 'PROOFTAG_QR_SRPG_PREVIEW_INTERVAL: "5"' in manifest
    assert 'PROOFTAG_QR_LATENT_REFINEMENT_ENABLED: "false"' in manifest
    assert 'PROOFTAG_QR_LATENT_REFINEMENT_MAX_LATENT_DELTA: "0.10"' in manifest
    assert 'PROOFTAG_QR_LATENT_REFINEMENT_MAX_MEAN_ABSOLUTE_CHANGE: "0.08"' in manifest


def test_guided_rediffusion_rejects_an_excessive_scheduler():
    with pytest.raises(ValueError, match="cannot schedule over 100 steps"):
        Settings(guided_rediffusion_steps=40, guided_rediffusion_strength=0.05)


def test_srpg_rejects_incompatible_pipeline_modes():
    with pytest.raises(ValueError, match="cannot be enabled together"):
        Settings(srpg_enabled=True, guided_rediffusion_enabled=True)
    with pytest.raises(ValueError, match="requires the img2img"):
        Settings(srpg_enabled=True, controlnet_pipeline_mode="text2img")


def test_targeted_repairs_run_before_global_module_repairs():
    blueprint = generate_qr("https://example.prooftag.test/t/profile-order", "H")
    backend = ControlNetBackend(Settings())
    names = [name for name, _ in backend.variants(blueprint.image, blueprint)]

    assert names.index("rounded_16") < names.index("perceptual_16")
    assert names.index("rounded_48") < names.index("perceptual_16")
    assert names.index("perceptual_16") < names.index("incorrect_80")
    assert names.index("perceptual_16_strong") < names.index("incorrect_80")
    assert names.index("perceptual_32_strong") < names.index("incorrect_80")
    assert names.index("perceptual_32_wide") < names.index("incorrect_80")
    assert names.index("perceptual_64") < names.index("incorrect_80")
    assert names.index("incorrect_80") < names.index("centers_45")
    assert names.index("incorrect_85") < names.index("centers_45")
    assert names.index("uncertain_16") < names.index("centers_45")
    assert names.index("uncertain_32") < names.index("centers_45")
    assert names.index("uncertain_48") < names.index("centers_45")
    assert names.index("uncertain_64") < names.index("centers_45")
    assert GLOBAL_REPAIR_VARIANTS <= set(names)
    assert all(
        names.index("uncertain_64") < names.index(variant) for variant in GLOBAL_REPAIR_VARIANTS
    )


def test_accepted_latent_becomes_the_base_for_targeted_repairs(monkeypatch):
    blueprint = generate_qr("https://example.prooftag.test/t/latent-chain", "H")
    backend = ControlNetBackend(Settings(latent_refinement_enabled=True))
    result = SimpleNamespace(
        image=blueprint.image.copy(),
        iterations=1,
        initial_module_error_rate=0.2,
        final_module_error_rate=0.1,
        final_srl=0.1,
        final_preservation_loss=0.01,
        final_mean_absolute_change=0.01,
        best_observed_module_error_rate=0.1,
        best_observed_mean_absolute_change=0.01,
        actual_initial_module_error_rate=0.2,
        actual_final_module_error_rate=0.1,
        improved=True,
        accepted=True,
        converged=False,
        rejection_reason=None,
    )
    monkeypatch.setattr(backend, "_load", lambda: object())
    monkeypatch.setattr(backends, "refine_candidate_latent", lambda *args: result)

    names = [name for name, _ in backend.variants(blueprint.image, blueprint)]

    assert names.index("latent_srl") < names.index("latent_rounded_16")
    assert names.index("latent_rounded_16") < names.index("rounded_16")
    assert names.index("latent_perceptual_64") < names.index("rounded_16")
    assert not any(name.startswith("latent_centers_") for name in names)
    assert GLOBAL_REPAIR_VARIANTS <= set(names)


def test_guided_rediffusion_becomes_the_base_for_targeted_repairs(monkeypatch):
    blueprint = generate_qr("https://example.prooftag.test/t/guided-chain", "H")
    backend = ControlNetBackend(Settings(guided_rediffusion_enabled=True))
    guided_image = blueprint.image.copy()
    guided = GuidedRediffusionResult(
        image=guided_image,
        unprojected_image=guided_image.copy(),
        control_image=blueprint.image.copy(),
        mask_image=blueprint.image.convert("L"),
        initial_module_error_rate=0.2,
        control_module_error_rate=0.0,
        final_module_error_rate=0.1,
        mask_coverage=0.2,
        changed_pixel_ratio=0.1,
        mean_absolute_change=0.02,
        unprojected_changed_pixel_ratio=0.4,
        unprojected_mean_absolute_change=0.08,
        accepted=True,
        rejection_reason=None,
    )
    monkeypatch.setattr(backend, "_guided_rediffuse", lambda *args: guided)
    request = GenerationRequest(payload="https://example.prooftag.test/t/guided-chain")

    names = [
        name
        for name, _ in backend.variants(
            blueprint.image,
            blueprint,
            request=request,
            seed=37,
        )
    ]

    assert names.index("raw") < names.index("guided")
    assert names.index("guided") < names.index("guided_rounded_16")
    assert names.index("guided_rounded_16") < names.index("rounded_16")
    assert not any(name.startswith("guided_centers_") for name in names)
    assert GLOBAL_REPAIR_VARIANTS <= set(names)
    assert set(backend.debug_artifacts()) == {
        "guided_control",
        "guided_mask",
        "guided_unprojected",
    }


def test_guided_then_latent_chain_uses_the_combined_prefix(monkeypatch):
    blueprint = generate_qr("https://example.prooftag.test/t/guided-latent-chain", "H")
    backend = ControlNetBackend(
        Settings(guided_rediffusion_enabled=True, latent_refinement_enabled=True)
    )
    guided_image = blueprint.image.copy()
    guided = GuidedRediffusionResult(
        image=guided_image,
        unprojected_image=guided_image.copy(),
        control_image=blueprint.image.copy(),
        mask_image=blueprint.image.convert("L"),
        initial_module_error_rate=0.2,
        control_module_error_rate=0.0,
        final_module_error_rate=0.1,
        mask_coverage=0.2,
        changed_pixel_ratio=0.1,
        mean_absolute_change=0.02,
        unprojected_changed_pixel_ratio=0.4,
        unprojected_mean_absolute_change=0.08,
        accepted=True,
        rejection_reason=None,
    )
    latent = SimpleNamespace(
        image=guided_image.copy(),
        iterations=2,
        initial_module_error_rate=0.1,
        final_module_error_rate=0.05,
        final_srl=0.1,
        final_preservation_loss=0.01,
        final_mean_absolute_change=0.01,
        best_observed_module_error_rate=0.05,
        best_observed_mean_absolute_change=0.01,
        actual_initial_module_error_rate=0.1,
        actual_final_module_error_rate=0.05,
        improved=True,
        accepted=True,
        converged=False,
        rejection_reason=None,
    )
    monkeypatch.setattr(backend, "_guided_rediffuse", lambda *args: guided)
    monkeypatch.setattr(backend, "_load", lambda: object())
    monkeypatch.setattr(backends, "refine_candidate_latent", lambda *args: latent)
    request = GenerationRequest(payload="https://example.prooftag.test/t/guided-latent-chain")

    names = [
        name
        for name, _ in backend.variants(
            blueprint.image,
            blueprint,
            request=request,
            seed=41,
        )
    ]

    assert names.index("guided") < names.index("guided_latent_srl")
    assert names.index("guided_latent_srl") < names.index("guided_latent_rounded_16")
    assert names.index("guided_latent_rounded_16") < names.index("rounded_16")


def test_rejected_guided_rediffusion_keeps_diagnostics_and_raw_fallback(monkeypatch):
    blueprint = generate_qr("https://example.prooftag.test/t/guided-rejected", "H")
    backend = ControlNetBackend(Settings(guided_rediffusion_enabled=True))
    guided_image = blueprint.image.copy()
    guided = GuidedRediffusionResult(
        image=guided_image,
        unprojected_image=guided_image.copy(),
        control_image=blueprint.image.copy(),
        mask_image=blueprint.image.convert("L"),
        initial_module_error_rate=0.2,
        control_module_error_rate=0.0,
        final_module_error_rate=0.1,
        mask_coverage=0.8,
        changed_pixel_ratio=0.9,
        mean_absolute_change=0.3,
        unprojected_changed_pixel_ratio=1.0,
        unprojected_mean_absolute_change=0.5,
        accepted=False,
        rejection_reason="actual_module_error_not_improved",
    )
    monkeypatch.setattr(backend, "_guided_rediffuse", lambda *args: guided)
    request = GenerationRequest(payload="https://example.prooftag.test/t/guided-rejected")

    names = [
        name
        for name, _ in backend.variants(
            blueprint.image,
            blueprint,
            request=request,
            seed=43,
        )
    ]

    assert "guided" not in names
    assert not any(name.startswith("guided_") for name in names)
    assert "rounded_16" in names
    assert "guided_projected" in backend.debug_artifacts()
