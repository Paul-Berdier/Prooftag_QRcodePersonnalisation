"""One-shot capture of the immutable Stage-2 parent required by E035.

The E034 archive did not persist its Stage-2 latent.  This recovery command therefore
loads the *exact observed E034 Stage-1 PNG* (hash-checked), executes only the frozen
E033 public Stage-2 recipe once, then writes an immutable PNG + safetensors contract.
The paired E035 loss experiment is a separate command and never calls Stage 1 or
Stage 2 generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .e035_parent_artifact import (
    E034_OBSERVED_STAGE1_FILE_SHA256,
    E034_OBSERVED_STAGE1_IMAGE_SHA256,
    export_parent_artifact,
    sha256_file,
    tensor_sha256,
)

UPSTREAM_REVISION = "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"
BASE_MODEL_ID = (
    "https://huggingface.co/fp16-guy/Cetus-Mix_Whalefall_fp16_cleaned/resolve/"
    "f914b3679760c1c3baea6bb1815867bf1c9c92a4/"
    "cetusMix_Whalefall2_fp16.safetensors"
)
BASE_MODEL_REVISION = "f914b3679760c1c3baea6bb1815867bf1c9c92a4"
BASE_MODEL_CONFIG_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL_CONFIG_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CONTROLNET_MODEL_ID = "monster-labs/control_v1p_sd15_qrcode_monster"
CONTROLNET_MODEL_SUBFOLDER = "v2"
CONTROLNET_MODEL_REVISION = "560fb7b15d0badb409f8cd578a2bfe63bd4b8046"
DEFAULT_PAYLOAD = "https://ptag.io/t/e033"
DEFAULT_PROMPT = (
    "a sunlit greenhouse filled with tomato plants and terracotta pots, "
    "botanical photograph"
)
DEFAULT_NEGATIVE_PROMPT = (
    "easynegative, low quality, worst quality, blurry, deformed, watermark, text, "
    "logo, oversaturated, clipped highlights, posterized colors"
)


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    payload: str = DEFAULT_PAYLOAD
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    error_correction: str = "M"
    seed: int = 51001
    steps: int = 40
    guidance_scale: float = 7.5
    controlnet_scale: float = 1.35
    strength: float = 1.0
    qr_version: int = 3
    qr_mask_pattern: int = 4
    qr_module_size: int = 20
    qr_padding_px: int = 78
    srpg_steps: int = 40
    srpg_controlnet_scale: float = 1.05
    srpg_qr_weight: float = 50.0
    srpg_perceptual_weight: float = 20.0
    srpg_eta: float = 0.0
    srpg_seed_offset: int = 2_000_003
    stage2_initialization: str = "public_random"
    stage2_strength: float = 1.0
    stage2_target_mode: str = "binary_exact"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _image_sha256(image: Image.Image) -> str:
    from .quality import image_sha256

    return image_sha256(image)


def _validate_git_sha(value: str, *, field: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def _settings_document(config: CaptureConfig) -> dict[str, Any]:
    """Frozen E033 public Stage-2 settings; SR-MPGD is explicitly disabled."""

    return {
        "device": "cuda",
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_config_id": BASE_MODEL_CONFIG_ID,
        "base_model_config_revision": BASE_MODEL_CONFIG_REVISION,
        "controlnet_model_id": CONTROLNET_MODEL_ID,
        "controlnet_model_subfolder": CONTROLNET_MODEL_SUBFOLDER,
        "controlnet_model_revision": CONTROLNET_MODEL_REVISION,
        "controlnet_conditioning_profile": "binary",
        "controlnet_pipeline_mode": "text2img",
        "diffqrcoder_upstream_enabled": True,
        "diffqrcoder_revision": UPSTREAM_REVISION,
        "diffqrcoder_qr_version": config.qr_version,
        "diffqrcoder_qr_mask_pattern": config.qr_mask_pattern,
        "diffqrcoder_qr_module_size": config.qr_module_size,
        "diffqrcoder_qr_padding_px": config.qr_padding_px,
        "diffqrcoder_control_guidance_start": 0.0,
        "diffqrcoder_control_guidance_end": 1.0,
        "diffqrcoder_stage2_initialization": config.stage2_initialization,
        "diffqrcoder_stage2_strength": config.stage2_strength,
        "diffqrcoder_stage2_target_mode": config.stage2_target_mode,
        "srpg_enabled": True,
        "srmpgd_enabled": False,
        "srpg_steps": config.srpg_steps,
        "srpg_controlnet_scale": config.srpg_controlnet_scale,
        "srpg_qr_weight": config.srpg_qr_weight,
        "srpg_perceptual_weight": config.srpg_perceptual_weight,
        "srpg_eta": config.srpg_eta,
        "srpg_seed_offset": config.srpg_seed_offset,
        "srpg_save_step_previews": False,
        "srpg_preview_interval": 1,
        # E033/E034 divergence guards. They observe and fail closed; they do not repair.
        "diffqrcoder_guard_max_changed_pixel_ratio": 0.995,
        "diffqrcoder_guard_max_mean_absolute_change": 0.35,
        "diffqrcoder_guard_max_clipped_pixel_ratio_increase": 0.05,
        "diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase": 0.02,
        "diffqrcoder_guard_max_saturation_mean_increase": 0.08,
        "diffqrcoder_guard_max_high_saturation_ratio_increase": 0.05,
        "diffqrcoder_guard_hard_max_mean_absolute_change": 0.40,
        "diffqrcoder_guard_hard_max_clipped_pixel_ratio_increase": 0.20,
        "diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase": 0.25,
        "diffqrcoder_guard_hard_max_saturation_mean_increase": 0.20,
        "diffqrcoder_guard_hard_max_high_saturation_ratio_increase": 0.30,
    }


def _load_verified_stage1(
    path: Path,
    *,
    expected_image_sha256: str,
    expected_file_sha256: str | None,
) -> tuple[Image.Image, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed E034 Stage-1 image: {path}")
    file_hash = sha256_file(path)
    if expected_file_sha256 and file_hash != expected_file_sha256:
        raise ValueError(
            "fixed E034 Stage-1 file hash mismatch: "
            f"expected {expected_file_sha256}, got {file_hash}"
        )
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        stage1 = opened.convert("RGB")
    if stage1.size != (736, 736):
        raise ValueError(f"fixed E034 Stage-1 image must be 736x736, got {stage1.size}")
    image_hash = _image_sha256(stage1)
    if image_hash != expected_image_sha256:
        raise ValueError(
            "fixed E034 Stage-1 pixel hash mismatch: "
            f"expected {expected_image_sha256}, got {image_hash}"
        )
    return stage1, {
        "path": str(path),
        "file_sha256": file_hash,
        "image_sha256": image_hash,
        "width": stage1.width,
        "height": stage1.height,
        "mode": "RGB",
    }


def capture_parent(
    *,
    stage1_image: Path,
    output_dir: Path,
    audit_dir: Path,
    source_commit: str,
    source_plan: str = "e035-parent-capture-from-e034-stage1-v1",
    expected_stage1_image_sha256: str = E034_OBSERVED_STAGE1_IMAGE_SHA256,
    expected_stage1_file_sha256: str | None = E034_OBSERVED_STAGE1_FILE_SHA256,
    config: CaptureConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Stage 2 once from the fixed E034 Stage-1 raster and freeze its latent."""

    import diffusers
    import torch

    from .config import Settings
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .qr import generate_diffqrcoder_qr
    from .schemas import GenerationRequest

    if config is None:
        config = CaptureConfig()
    _validate_git_sha(source_commit, field="source_commit")
    _validate_git_sha(UPSTREAM_REVISION, field="diffqrcoder_revision")
    if not torch.cuda.is_available():
        raise RuntimeError("E035 parent capture requires an available CUDA GPU")
    for path, label in ((output_dir, "parent"), (audit_dir, "capture audit")):
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"{label} output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)

    stage1, stage1_provenance = _load_verified_stage1(
        stage1_image,
        expected_image_sha256=expected_stage1_image_sha256,
        expected_file_sha256=expected_stage1_file_sha256,
    )

    random.seed(config.seed)
    np.random.seed(config.seed % (2**32))
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats()

    base = Settings()
    settings = Settings.model_validate({**base.model_dump(), **_settings_document(config)})
    if str(settings.diffqrcoder_revision) != UPSTREAM_REVISION:
        raise RuntimeError("runtime DiffQRCoder revision differs from the frozen E035 revision")
    backend = UpstreamDiffQRCoderBackend(settings)
    request = GenerationRequest(
        payload=config.payload,
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        backend="controlnet",
        error_correction=config.error_correction,
        seed=config.seed,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        controlnet_scale=config.controlnet_scale,
        strength=config.strength,
        max_attempts=1,
    )
    blueprint = generate_diffqrcoder_qr(
        config.payload,
        config.error_correction,
        version=config.qr_version,
        mask_pattern=config.qr_mask_pattern,
        module_size=config.qr_module_size,
    )

    started = time.perf_counter()
    # Deliberately bypass ``backend.generate``: Stage 1 is the hash-verified E034 PNG.
    stage2 = backend._run_stage2(stage1, blueprint, request, config.seed)
    state = backend.export_stage2_state()
    if state is None:
        raise RuntimeError("Stage-2 execution completed without an exportable latent state")
    state_image = state["image"].convert("RGB")
    if _image_sha256(stage2) != _image_sha256(state_image):
        raise RuntimeError("exported Stage-2 image does not match the delivered Stage-2 raster")
    state_latent_hash = tensor_sha256(state["latent"])
    embedded_latent_hash = str(state.get("latent_sha256") or state_latent_hash)
    if state_latent_hash != embedded_latent_hash:
        raise RuntimeError("exported Stage-2 latent hash mismatch")
    if stage2.size != (736, 736):
        raise RuntimeError(f"expected a 736x736 Stage-2 raster, got {stage2.size}")

    pipe = backend._load()
    source_run_id = (
        f"e035-parent-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:12]}"
    )
    source = {
        "payload": config.payload,
        "error_correction": config.error_correction,
        "qr_version": config.qr_version,
        "qr_mask_pattern": config.qr_mask_pattern,
        "qr_module_size": config.qr_module_size,
        "qr_padding_px": config.qr_padding_px,
        "source_commit": source_commit,
        "source_plan": source_plan,
        "source_run_id": source_run_id,
        "source_method_id": "e033_public_demo_srpg_from_fixed_e034_stage1",
        "parent_origin": "stage2_replayed_from_exact_e034_stage1",
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
        "base_model_id": settings.base_model_id,
        "base_model_revision": settings.base_model_revision,
        "base_model_config_id": settings.base_model_config_id,
        "base_model_config_revision": settings.base_model_config_revision,
        "controlnet_model_id": settings.controlnet_model_id,
        "controlnet_model_subfolder": settings.controlnet_model_subfolder,
        "controlnet_model_revision": settings.controlnet_model_revision,
        "diffqrcoder_revision": settings.diffqrcoder_revision,
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "seed": config.seed,
        "stage2_seed": (config.seed + config.srpg_seed_offset) % (2**32),
        "stage1": stage1_provenance,
        "generation": {
            "stage1_regenerated": False,
            "steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "controlnet_scale": config.controlnet_scale,
            "strength": config.strength,
        },
        "srpg": {
            "steps": config.srpg_steps,
            "controlnet_scale": config.srpg_controlnet_scale,
            "qr_weight": config.srpg_qr_weight,
            "perceptual_weight": config.srpg_perceptual_weight,
            "eta": config.srpg_eta,
            "seed_offset": config.srpg_seed_offset,
            "initialization": config.stage2_initialization,
            "strength": config.stage2_strength,
            "target_mode": config.stage2_target_mode,
        },
        "stage1_image_sha256": stage1_provenance["image_sha256"],
        "stage1_file_sha256": stage1_provenance["file_sha256"],
        "stage2_image_sha256": _image_sha256(stage2),
        "stage2_latent_sha256": state_latent_hash,
        "software": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
        },
        "gpu": torch.cuda.get_device_name(0),
    }
    metadata = export_parent_artifact(
        output_dir,
        latent=state["latent"],
        image=stage2,
        source=source,
        overwrite=overwrite,
    )

    shutil.copy2(stage1_image, audit_dir / "fixed-e034-stage1.png")
    stage2.save(
        audit_dir / "captured-stage2.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    audit = {
        "schema": "prooftag.e035.parent-capture-audit.v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "duration_s": time.perf_counter() - started,
        "stage1_regenerated": False,
        "config": asdict(config),
        "settings": _settings_document(config),
        "source": source,
        "parent_contract_sha256": metadata["contract_sha256"],
        "backend_diagnostics": backend.diagnostics(),
        "backend_provenance": backend.provenance(),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    _atomic_json(audit_dir / "capture-audit.json", audit)
    _atomic_json(audit_dir / "parent-contract-copy.json", metadata)

    del state, stage1, stage2, state_image, pipe, backend
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--source-plan",
        default="e035-parent-capture-from-e034-stage1-v1",
    )
    parser.add_argument(
        "--expected-stage1-image-sha256",
        default=E034_OBSERVED_STAGE1_IMAGE_SHA256,
    )
    parser.add_argument(
        "--expected-stage1-file-sha256",
        default=E034_OBSERVED_STAGE1_FILE_SHA256,
    )
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = CaptureConfig(
        payload=args.payload,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    metadata = capture_parent(
        stage1_image=args.stage1_image,
        output_dir=args.output_dir,
        audit_dir=args.audit_dir,
        source_commit=args.source_commit,
        source_plan=args.source_plan,
        expected_stage1_image_sha256=args.expected_stage1_image_sha256,
        expected_stage1_file_sha256=(
            args.expected_stage1_file_sha256 or None
        ),
        config=config,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
