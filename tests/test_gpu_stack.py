import tomllib
from pathlib import Path

from prooftag_qr.backends import ControlNetBackend
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
    assert request.strength == 0.9
    assert "PROOFTAG_QR_CONTROLNET_PIPELINE_MODE: img2img" in manifest


def test_targeted_repairs_run_before_global_module_repairs():
    blueprint = generate_qr("https://example.prooftag.test/t/profile-order", "H")
    backend = ControlNetBackend(Settings())
    names = [name for name, _ in backend.variants(blueprint.image, blueprint)]

    assert names.index("incorrect_80") < names.index("centers_45")
    assert names.index("incorrect_85") < names.index("centers_45")
    assert names.index("uncertain_16") < names.index("centers_45")
    assert names.index("uncertain_32") < names.index("centers_45")
    assert names.index("uncertain_48") < names.index("centers_45")
    assert names.index("uncertain_64") < names.index("centers_45")
