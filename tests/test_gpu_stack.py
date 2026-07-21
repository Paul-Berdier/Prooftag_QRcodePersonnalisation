import tomllib
from pathlib import Path
from types import SimpleNamespace

from prooftag_qr import backends
from prooftag_qr.backends import GLOBAL_REPAIR_VARIANTS, ControlNetBackend
from prooftag_qr.config import Settings
from prooftag_qr.qr import generate_qr
from prooftag_qr.schemas import GenerationRequest


def test_gpu_dependencies_are_pinned_to_the_torch_base_image():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = set(project["project"]["optional-dependencies"]["gpu"])

    assert {
        "accelerate==0.34.2",
        "diffusers==0.31.0",
        "huggingface-hub==0.25.2",
        "safetensors==0.4.5",
        "transformers==4.44.2",
    } <= dependencies

    dockerfile = Path("Dockerfile").read_text()
    assert "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime" in dockerfile
    assert "from diffusers import ControlNetModel" in dockerfile
    assert "StableDiffusionControlNetImg2ImgPipeline" in dockerfile
    assert "DDIMScheduler" in dockerfile
    assert "TRANSFORMERS_CACHE" not in dockerfile

    settings = Settings()
    request = GenerationRequest(payload="https://example.prooftag.test/t/img2img")
    manifest = Path("deploy/k8s/app-config.yaml").read_text()
    assert settings.controlnet_pipeline_mode == "img2img"
    assert settings.regenerate_before_global_repair is True
    assert settings.latent_refinement_enabled is False
    assert settings.latent_refinement_iterations == 8
    assert settings.latent_refinement_learning_rate == 0.02
    assert settings.latent_refinement_preservation_weight == 1.0
    assert settings.latent_refinement_max_latent_delta == 0.10
    assert settings.latent_refinement_max_mean_absolute_change == 0.08
    assert request.strength == 0.9
    assert "PROOFTAG_QR_CONTROLNET_PIPELINE_MODE: img2img" in manifest
    assert 'PROOFTAG_QR_REGENERATE_BEFORE_GLOBAL_REPAIR: "true"' in manifest
    assert 'PROOFTAG_QR_LATENT_REFINEMENT_ENABLED: "false"' in manifest
    assert 'PROOFTAG_QR_LATENT_REFINEMENT_MAX_LATENT_DELTA: "0.10"' in manifest
    assert (
        'PROOFTAG_QR_LATENT_REFINEMENT_MAX_MEAN_ABSOLUTE_CHANGE: "0.08"' in manifest
    )


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
