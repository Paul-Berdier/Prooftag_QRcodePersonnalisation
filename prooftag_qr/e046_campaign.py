"""E046 — campagne contrôlée de génération d'un dataset QR neuf.

Phases séparées :
1. plan CPU ;
2. génération GPU Stage 1 + Stage 2 par candidat ;
3. scoring CPU qr-scanner-wechat + qualité ;
4. sélection diverse des parents ;
5. SR-MPGD GPU par couple parent/recette ;
6. scoring CPU de tous les checkpoints ;
7. agrégation, Pareto et artefacts Jupyter.

La séparation génération/scoring garantit qu'une erreur HPS, QR-Verify ou
dépendance CPU ne détruit jamais une génération GPU déjà terminée.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .e046_catalog import (
    CANVAS_PX,
    CATALOG_SCHEMA,
    EXPERIMENT,
    NEGATIVE_PROMPT,
    PARENT_RECIPES,
    QR_MODULE_SIZE,
    QR_PADDING_PX,
    QR_SOFTWARE_ENGINE,
    QR_VERIFY_PRESET_COUNT,
    QR_VERSION,
    CandidateSpec,
    ParentRecipe,
    SRMPGDRecipe,
    catalog_document,
    parent_recipe_by_id,
    scientific_plan,
    srmpgd_recipe_by_id,
)
from .e046_quiet_zone import (
    compare_core_bytes,
    compose_scene_preserving_quiet_zone,
    core_bounds,
    core_pixel_sha256,
    quiet_zone_metrics,
)
from .resilient_experiment import (
    ArtifactPromotionError,
    atomic_write_json,
    atomic_write_text,
    build_artifact_manifest,
    promote_attempt,
    sha256_file,
    stable_hash,
    utc_now,
)

DEFAULT_OUTPUT_ROOT = Path("/data/e046-controlled-best-generator-v1")
DEFAULT_E045_ROOT = Path("/data/e045-foundation-v1")
PRIMARY_LABEL = "wechat_exact_presets"
PRIMARY_RATE = "wechat_exact_rate"


def _validate_commit(value: str) -> str:
    value = value.strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields or ["empty"],
            extrasaction="ignore",
        )
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _image_sha256(image: Image.Image) -> str:
    from .quality import image_sha256

    return image_sha256(image.convert("RGB"))


def _stage_core_change(
    candidate: Image.Image,
    reference: Image.Image,
    *,
    padding_px: int = QR_PADDING_PX,
) -> dict[str, float]:
    bounds = core_bounds(reference, padding_px)
    left = np.asarray(reference.convert("RGB").crop(bounds), dtype=np.float32) / 255.0
    right = np.asarray(candidate.convert("RGB").crop(bounds), dtype=np.float32) / 255.0
    delta = np.abs(right - left)
    return {
        "core_mean_absolute_change": float(delta.mean()),
        "core_max_absolute_change": float(delta.max()),
        "core_changed_pixel_ratio": float((delta.max(axis=2) > (1.0 / 255.0)).mean()),
    }


def _safe_settings_provenance(settings: Any) -> dict[str, Any]:
    """Record only model/runtime fields; never serialize database secrets."""
    fields = (
        "device",
        "base_model_id",
        "base_model_revision",
        "base_model_config_id",
        "base_model_config_revision",
        "controlnet_model_id",
        "controlnet_model_subfolder",
        "controlnet_model_revision",
        "diffqrcoder_revision",
        "diffqrcoder_qr_version",
        "diffqrcoder_qr_mask_pattern",
        "diffqrcoder_qr_module_size",
        "diffqrcoder_qr_padding_px",
        "diffqrcoder_control_guidance_start",
        "diffqrcoder_control_guidance_end",
        "diffqrcoder_stage2_initialization",
        "diffqrcoder_stage2_strength",
        "diffqrcoder_stage2_target_mode",
        "srpg_steps",
        "srpg_controlnet_scale",
        "srpg_qr_weight",
        "srpg_perceptual_weight",
        "srpg_eta",
        "srpg_seed_offset",
    )
    result: dict[str, Any] = {}
    for field in fields:
        value = getattr(settings, field, None)
        if isinstance(value, Path):
            value = str(value)
        result[field] = value
    return result


def _settings_for(candidate: Mapping[str, Any], recipe: ParentRecipe) -> Any:
    from .config import Settings
    from .e035_parent_capture import CaptureConfig, _settings_document

    config = CaptureConfig(
        payload=str(candidate["payload"]),
        prompt=str(candidate["prompt"]),
        negative_prompt=NEGATIVE_PROMPT,
        error_correction=recipe.error_correction,
        seed=int(candidate["seed"]),
        steps=recipe.stage1_steps,
        guidance_scale=recipe.stage1_guidance_scale,
        controlnet_scale=recipe.stage1_controlnet_scale,
        strength=1.0,
        qr_version=QR_VERSION,
        qr_mask_pattern=recipe.qr_mask_pattern,
        qr_module_size=QR_MODULE_SIZE,
        qr_padding_px=QR_PADDING_PX,
        srpg_steps=recipe.stage2_steps,
        srpg_controlnet_scale=recipe.stage2_controlnet_scale,
        srpg_qr_weight=recipe.stage2_qr_weight,
        srpg_perceptual_weight=recipe.stage2_perceptual_weight,
        srpg_eta=0.0,
        srpg_seed_offset=2_000_003,
        stage2_initialization=recipe.stage2_initialization,
        stage2_strength=recipe.stage2_strength,
        stage2_target_mode="binary_exact",
    )
    document = _settings_document(config)
    document.update(
        {
            "diffqrcoder_control_guidance_start": recipe.control_guidance_start,
            "diffqrcoder_control_guidance_end": recipe.control_guidance_end,
            "srpg_save_step_previews": False,
            "srmpgd_enabled": False,
        }
    )
    base = Settings()
    return Settings.model_validate({**base.model_dump(), **document})


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _blueprint(candidate: Mapping[str, Any], recipe: ParentRecipe) -> Any:
    from .qr import generate_diffqrcoder_qr

    return generate_diffqrcoder_qr(
        str(candidate["payload"]),
        recipe.error_correction,
        version=QR_VERSION,
        mask_pattern=recipe.qr_mask_pattern,
        module_size=QR_MODULE_SIZE,
    )


def _e045_contract(e045_root: Path) -> dict[str, Any]:
    latest_path = e045_root / "LATEST.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"E045 LATEST missing: {latest_path}")
    latest = _load_json(latest_path)
    if latest.get("status") != "complete":
        raise RuntimeError(f"E045 is not complete: {latest}")
    plan_dir = Path(str(latest["plan_dir"]))
    complete_path = plan_dir / "COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"E045 COMPLETE missing: {complete_path}")
    complete = _load_json(complete_path)
    if complete.get("complete") is not True:
        raise RuntimeError("E045 COMPLETE flag is false")
    if complete.get("resilience_selftest_passed") is not True:
        raise RuntimeError("E045 resilience selftest did not pass")
    if complete.get("production_ready") is not False:
        raise RuntimeError("E045 historical contract unexpectedly claims production readiness")
    manifest_hash = str(complete.get("artifact_manifest_sha256") or "")
    if len(manifest_hash) != 64:
        raise RuntimeError("E045 manifest SHA-256 is absent or invalid")
    return {
        "plan_id": str(complete["plan_id"]),
        "plan_dir": str(plan_dir),
        "manifest_sha256": manifest_hash,
        "source_commit": str(complete["source_commit"]),
        "summary": complete,
    }


def create_plan(
    *,
    output_root: Path,
    profile: str,
    source_commit: str,
    e045_root: Path,
) -> dict[str, Any]:
    source_commit = _validate_commit(source_commit)
    e045 = _e045_contract(e045_root)
    plan = scientific_plan(
        profile=profile,
        source_commit=source_commit,
        e045_plan_id=e045["plan_id"],
        e045_manifest_sha256=e045["manifest_sha256"],
    )
    plan["created_at_utc"] = utc_now()
    plan["e045_plan_dir"] = e045["plan_dir"]
    plan_dir = output_root / str(plan["plan_id"])
    plan_path = plan_dir / "plan.json"

    if plan_path.is_file():
        existing = _load_json(plan_path)
        if existing.get("scientific_plan_hash") != plan["scientific_plan_hash"]:
            raise RuntimeError(f"E046 plan-id collision: {plan_dir}")
        plan = existing
    else:
        plan_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(plan_path, plan)
        atomic_write_json(plan_dir / "catalog.json", catalog_document())
        for directory in ("parents", "refinements", "attempts", "failures", "pipeline"):
            (plan_dir / directory).mkdir(parents=True, exist_ok=True)

    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "LATEST.json",
        {
            "schema": "e046-latest-v1",
            "plan_id": plan["plan_id"],
            "plan_dir": str(plan_dir),
            "profile": profile,
            "source_commit": source_commit,
            "status": "planned",
            "updated_at_utc": utc_now(),
        },
    )
    return plan


def resolve_plan_dir(
    output_root: Path,
    plan_id: str | None = None,
) -> Path:
    if plan_id:
        plan_dir = output_root / plan_id
    else:
        latest = _load_json(output_root / "LATEST.json")
        plan_dir = Path(str(latest["plan_dir"]))
    if not (plan_dir / "plan.json").is_file():
        raise FileNotFoundError(f"E046 plan missing: {plan_dir / 'plan.json'}")
    return plan_dir


def load_plan(output_root: Path, plan_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    plan_dir = resolve_plan_dir(output_root, plan_id)
    plan = _load_json(plan_dir / "plan.json")
    if plan.get("experiment") != EXPERIMENT:
        raise RuntimeError(f"unexpected E046 experiment: {plan.get('experiment')}")
    return plan_dir, plan


def _candidate(plan: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = next(
        (item for item in plan["candidates"] if item["id"] == candidate_id),
        None,
    )
    if candidate is None:
        raise KeyError(f"unknown E046 candidate: {candidate_id}")
    return dict(candidate)


def _parent_final_dir(plan_dir: Path, candidate_id: str) -> Path:
    return plan_dir / "parents" / candidate_id


def _refinement_final_dir(
    plan_dir: Path,
    candidate_id: str,
    recipe_id: str,
) -> Path:
    return plan_dir / "refinements" / candidate_id / recipe_id


def _record_failure(
    *,
    plan_dir: Path,
    task_kind: str,
    task_id: str,
    error: BaseException,
) -> None:
    path = plan_dir / "failures" / (
        f"{task_kind}-{task_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}.json"
    )
    atomic_write_json(
        path,
        {
            "task_kind": task_kind,
            "task_id": task_id,
            "error_class": type(error).__name__,
            "error": str(error),
            "timestamp_utc": utc_now(),
        },
    )


def generate_parent(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    source_commit: str,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .schemas import GenerationRequest

    source_commit = _validate_commit(source_commit)
    plan_dir, plan = load_plan(output_root, plan_id)
    if str(plan["source_commit"]) != source_commit:
        raise RuntimeError("E046 parent source commit differs from plan")
    candidate = _candidate(plan, candidate_id)
    recipe = parent_recipe_by_id(str(candidate["parent_recipe_id"]))
    final_dir = _parent_final_dir(plan_dir, candidate_id)
    complete_path = final_dir / "GENERATION_COMPLETE.json"
    if complete_path.is_file():
        return _load_json(complete_path)
    if final_dir.exists():
        raise FileExistsError(
            f"E046 parent final directory exists without completion marker: {final_dir}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("E046 parent generation requires CUDA")

    attempt = (
        plan_dir
        / "attempts"
        / "parents"
        / f"{candidate_id}-{source_commit[:12]}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        attempt / "task.json",
        {
            "experiment": EXPERIMENT,
            "kind": "parent",
            "candidate": candidate,
            "parent_recipe": asdict(recipe),
            "source_commit": source_commit,
            "started_at_utc": utc_now(),
        },
    )

    started = time.perf_counter()
    try:
        settings = _settings_for(candidate, recipe)
        backend = UpstreamDiffQRCoderBackend(settings)
        blueprint = _blueprint(candidate, recipe)
        request = GenerationRequest(
            payload=str(candidate["payload"]),
            prompt=str(candidate["prompt"]),
            negative_prompt=NEGATIVE_PROMPT,
            backend="controlnet",
            error_correction=recipe.error_correction,
            seed=int(candidate["seed"]),
            steps=recipe.stage1_steps,
            guidance_scale=recipe.stage1_guidance_scale,
            controlnet_scale=recipe.stage1_controlnet_scale,
            strength=1.0,
            max_attempts=1,
        )
        _seed_everything(int(candidate["seed"]))
        torch.cuda.reset_peak_memory_stats()

        stage1 = backend.generate(request, blueprint, int(candidate["seed"])).convert("RGB")
        if stage1.size != (CANVAS_PX, CANVAS_PX):
            raise RuntimeError(f"E046 Stage1 expected 736x736, got {stage1.size}")
        _save_png(attempt / "images/stage1-raw.png", stage1)
        stage1_scene, stage1_qz = compose_scene_preserving_quiet_zone(
            stage1,
            padding_px=QR_PADDING_PX,
        )
        _save_png(attempt / "images/stage1-scene-qz.png", stage1_scene)
        atomic_write_json(attempt / "quiet-zone/stage1-scene-qz.json", stage1_qz)
        atomic_write_json(
            attempt / "STAGE1_COMPLETE.json",
            {
                "image_sha256": _image_sha256(stage1),
                "scene_qz_image_sha256": _image_sha256(stage1_scene),
                "core_sha256": core_pixel_sha256(stage1, QR_PADDING_PX),
                "completed_at_utc": utc_now(),
            },
        )

        stage2 = backend._run_stage2(
            stage1,
            blueprint,
            request,
            int(candidate["seed"]),
        ).convert("RGB")
        state = backend.export_stage2_state()
        if state is None:
            raise RuntimeError("E046 Stage2 produced no exportable latent")
        latent = state["latent"].detach().cpu().contiguous()
        if stage2.size != (CANVAS_PX, CANVAS_PX):
            raise RuntimeError(f"E046 Stage2 expected 736x736, got {stage2.size}")

        _save_png(attempt / "images/stage2-raw.png", stage2)
        stage2_scene, stage2_qz = compose_scene_preserving_quiet_zone(
            stage2,
            padding_px=QR_PADDING_PX,
        )
        _save_png(attempt / "images/stage2-scene-qz.png", stage2_scene)
        atomic_write_json(attempt / "quiet-zone/stage2-scene-qz.json", stage2_qz)
        save_file(
            {"latent": latent},
            str(attempt / "stage2-latent.safetensors"),
            metadata={
                "experiment": EXPERIMENT,
                "candidate_id": candidate_id,
                "source_commit": source_commit,
            },
        )

        metadata = {
            "experiment": EXPERIMENT,
            "kind": "parent",
            "candidate": candidate,
            "parent_recipe": asdict(recipe),
            "source_commit": source_commit,
            "runtime_image": os.environ.get("PROOFTAG_RUNTIME_IMAGE"),
            "runtime_image_digest": os.environ.get("PROOFTAG_RUNTIME_IMAGE_DIGEST"),
            "settings": _safe_settings_provenance(settings),
            "stage1_image_sha256": _image_sha256(stage1),
            "stage1_scene_qz_sha256": _image_sha256(stage1_scene),
            "stage2_image_sha256": _image_sha256(stage2),
            "stage2_scene_qz_sha256": _image_sha256(stage2_scene),
            "stage2_latent_file_sha256": sha256_file(
                attempt / "stage2-latent.safetensors"
            ),
            "stage2_latent_tensor_sha256": str(state["latent_sha256"]),
            "stage2_diagnostics": dict(state.get("diagnostics") or {}),
            "quiet_zone_policy": "raw + scene-preserving; no uniform replacement",
            "elapsed_s": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "parent-metadata.json", metadata)
        atomic_write_json(
            attempt / "GENERATION_COMPLETE.json",
            {
                "experiment": EXPERIMENT,
                "candidate_id": candidate_id,
                "source_commit": source_commit,
                "generation_complete": True,
                "stage2_latent_tensor_sha256": metadata[
                    "stage2_latent_tensor_sha256"
                ],
                "completed_at_utc": utc_now(),
            },
        )

        promoted = promote_attempt(
            attempt_dir=attempt,
            final_dir=final_dir,
            required_files=(
                "images/stage1-raw.png",
                "images/stage1-scene-qz.png",
                "images/stage2-raw.png",
                "images/stage2-scene-qz.png",
                "stage2-latent.safetensors",
                "parent-metadata.json",
                "GENERATION_COMPLETE.json",
            ),
            metadata={
                "experiment": EXPERIMENT,
                "kind": "parent",
                "candidate_id": candidate_id,
                "source_commit": source_commit,
            },
        )
        result = _load_json(final_dir / "GENERATION_COMPLETE.json")
        result["promotion_manifest_hash"] = promoted["manifest_hash"]
        backend._pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="parent",
            task_id=candidate_id,
            error=exc,
        )
        raise


def _quality_settings(candidate: Mapping[str, Any], recipe: ParentRecipe) -> Any:
    return _settings_for(candidate, recipe)


def _parent_visual_guard(
    *,
    row: Mapping[str, Any],
    stage1_quality: Mapping[str, Any],
    scene_qz_guard: bool | None,
) -> dict[str, Any]:
    """Fail only on Stage2 degeneration; Stage1 quality deltas are diagnostic.

    Stage2 is a full generative step, not a local refinement of Stage1. Relative
    CLIP/HPS/AES drops therefore remain recorded but do not make a valid Stage2
    ineligible. The tight relative visual guards remain on Stage2 -> SR-MPGD.
    """

    def finite(value: Any) -> bool:
        if value is None:
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    checks = {
        "mean_absolute_change": (
            finite(row.get("mean_absolute_change"))
            and float(row["mean_absolute_change"]) <= 0.35
        ),
        "clipped_pixel_ratio_increase": (
            finite(row.get("clipped_pixel_ratio_increase"))
            and float(row["clipped_pixel_ratio_increase"]) <= 0.20
        ),
        "rgb_clipped_channel_ratio_increase": (
            finite(row.get("rgb_clipped_channel_ratio_increase"))
            and float(row["rgb_clipped_channel_ratio_increase"]) <= 0.25
        ),
        "abs_saturation_mean_change": (
            finite(row.get("saturation_mean_increase"))
            and abs(float(row["saturation_mean_increase"])) <= 0.20
        ),
        "high_saturation_ratio_increase": (
            finite(row.get("high_saturation_ratio_increase"))
            and float(row["high_saturation_ratio_increase"]) <= 0.30
        ),
        "clip_score_finite": finite(row.get("clip_score")),
        "clip_aesthetic_finite": finite(row.get("clip_aesthetic")),
    }
    if row.get("hpsv2_1") is not None:
        checks["hpsv2_1_finite"] = finite(row.get("hpsv2_1"))
    if scene_qz_guard is not None:
        checks["scene_preserving_quiet_zone"] = bool(scene_qz_guard)

    deltas: dict[str, float | None] = {}
    for metric in ("clip_score", "clip_aesthetic", "hpsv2_1"):
        value = row.get(metric)
        reference = stage1_quality.get(metric)
        deltas[metric] = (
            float(value) - float(reference)
            if finite(value) and finite(reference)
            else None
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "policy": "e046-stage2-nondegenerate-parent-guard-v2",
        "stage1_quality_deltas_diagnostic_only": deltas,
    }


def _score_image_set(
    *,
    output_dir: Path,
    images: Mapping[str, Image.Image],
    prompt: str,
    payload: str,
    settings: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from .e035_loss_fidelity import _score_qr_verify
    from .e038_recipe_frontier import _score_quality

    qr = _score_qr_verify(output_dir, payload, dict(images))
    quality, provenance = _score_quality(dict(images), prompt, settings)
    atomic_write_json(output_dir / "quality-scores.json", quality)
    atomic_write_json(output_dir / "quality-provenance.json", provenance)
    return qr, quality, provenance


def _row_qr_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    exact = int(item.get("conservative_exact_presets", 0))

    original_exact = item.get("direct_exact_all_repetitions")
    if original_exact is None:
        from .e038_recipe_frontier import _qr_original_exact

        original_exact = _qr_original_exact(dict(item))

    return {
        "wechat_engine": QR_SOFTWARE_ENGINE,
        "wechat_preset_count": QR_VERIFY_PRESET_COUNT,
        "wechat_repetitions": 3,
        "wechat_exact_presets": exact,
        "wechat_exact_rate": exact / QR_VERIFY_PRESET_COUNT,
        "wechat_original_exact": bool(original_exact),
        "qr_primary_label": "exact payload only",
    }


def _module_fields(
    image: Image.Image,
    blueprint: Any,
) -> dict[str, Any]:
    from .qr import (
        diffqrcoder_module_error_rate,
        diffqrcoder_structure_metrics,
        module_error_breakdown,
    )

    rate = diffqrcoder_module_error_rate(
        image,
        blueprint,
        padding_px=QR_PADDING_PX,
        module_size=QR_MODULE_SIZE,
    )
    structure = diffqrcoder_structure_metrics(
        image,
        blueprint,
        padding_px=QR_PADDING_PX,
        module_size=QR_MODULE_SIZE,
    )
    return {
        "module_error_rate": float(rate),
        "full_module_error_count": int(round(rate * 29 * 29)),
        "module_error_breakdown": module_error_breakdown(image, blueprint),
        **structure,
    }


def score_parent(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    from .quality import image_change_metrics, image_quality_metrics

    plan_dir, plan = load_plan(output_root, plan_id)
    candidate = _candidate(plan, candidate_id)
    recipe = parent_recipe_by_id(str(candidate["parent_recipe_id"]))
    parent_dir = _parent_final_dir(plan_dir, candidate_id)
    generation_path = parent_dir / "GENERATION_COMPLETE.json"
    if not generation_path.is_file():
        raise FileNotFoundError(f"parent generation incomplete: {candidate_id}")
    complete_path = parent_dir / "SCORING_COMPLETE.json"
    if complete_path.is_file():
        return _load_json(complete_path)

    try:
        images = {
            "stage1_raw": Image.open(parent_dir / "images/stage1-raw.png").convert("RGB"),
            "stage1_scene_qz": Image.open(
                parent_dir / "images/stage1-scene-qz.png"
            ).convert("RGB"),
            "stage2_raw": Image.open(parent_dir / "images/stage2-raw.png").convert("RGB"),
            "stage2_scene_qz": Image.open(
                parent_dir / "images/stage2-scene-qz.png"
            ).convert("RGB"),
        }
        stage1 = images["stage1_raw"]
        parent_metadata = _load_json(parent_dir / "parent-metadata.json")
        blueprint = _blueprint(candidate, recipe)
        settings = _quality_settings(candidate, recipe)
        scoring_dir = parent_dir / "scoring"
        scoring_dir.mkdir(parents=True, exist_ok=True)
        qr, quality, _ = _score_image_set(
            output_dir=scoring_dir,
            images=images,
            prompt=str(candidate["prompt"]),
            payload=str(candidate["payload"]),
            settings=settings,
        )
        stage1_quality = quality["stage1_raw"]
        rows: list[dict[str, Any]] = []
        for variant, image in images.items():
            change = image_change_metrics(image, stage1)
            qz = quiet_zone_metrics(image, padding_px=QR_PADDING_PX)
            qz_evidence_path = (
                parent_dir
                / "quiet-zone"
                / (
                    "stage1-scene-qz.json"
                    if variant == "stage1_scene_qz"
                    else "stage2-scene-qz.json"
                )
            )
            qz_evidence = (
                _load_json(qz_evidence_path)
                if variant.endswith("scene_qz") and qz_evidence_path.is_file()
                else None
            )
            qscore = quality[variant]
            row = {
                "experiment": EXPERIMENT,
                "plan_id": plan_id,
                "source_kind": "parent",
                "source_commit": parent_metadata.get("source_commit"),
                "runtime_image": parent_metadata.get("runtime_image"),
                "runtime_image_digest": parent_metadata.get("runtime_image_digest"),
                "candidate_id": candidate_id,
                "parent_recipe_id": recipe.id,
                "prompt_id": candidate["prompt_id"],
                "prompt_family": candidate["prompt_family"],
                "prompt_variant_index": candidate["prompt_variant_index"],
                "prompt": candidate["prompt"],
                "payload": candidate["payload"],
                "payload_sha256": hashlib.sha256(
                    str(candidate["payload"]).encode("utf-8")
                ).hexdigest(),
                "seed": candidate["seed"],
                "error_correction": recipe.error_correction,
                "qr_version": QR_VERSION,
                "qr_mask_pattern": recipe.qr_mask_pattern,
                "qr_module_size": QR_MODULE_SIZE,
                "qr_padding_px": QR_PADDING_PX,
                "stage": "stage1" if variant.startswith("stage1") else "stage2",
                "variant": variant,
                "quiet_zone_variant": (
                    "scene_preserving" if variant.endswith("scene_qz") else "raw"
                ),
                "uniform_quiet_zone_replacement": False,
                "image_path": str(
                    parent_dir
                    / "images"
                    / f"{variant.replace('_', '-')}.png"
                ),
                "image_sha256": _image_sha256(image),
                "core_sha256": core_pixel_sha256(image, QR_PADDING_PX),
                "latent_path": (
                    str(parent_dir / "stage2-latent.safetensors")
                    if variant.startswith("stage2")
                    else None
                ),
                "iteration": 0,
                "gamma": 0.0,
                "lpips": 0.0,
                **_row_qr_fields(qr.get(variant) or {}),
                **_module_fields(image, blueprint),
                **change,
                **image_quality_metrics(image),
                **qscore,
                "quiet_zone_metrics": asdict(qz),
                "quiet_zone_delivery_guard_pass": (
                    bool(qz_evidence["delivery_guard_pass"])
                    if qz_evidence is not None
                    else None
                ),
                "core_byte_identical_to_raw": (
                    compare_core_bytes(
                        image,
                        images[variant.replace("_scene_qz", "_raw")],
                        padding_px=QR_PADDING_PX,
                    )
                    if variant.endswith("scene_qz")
                    else True
                ),
            }
            if row["stage"] == "stage1":
                guard = {
                    "passed": (
                        row["quiet_zone_delivery_guard_pass"]
                        if variant.endswith("scene_qz")
                        else True
                    ),
                    "checks": {
                        "baseline": True,
                        "scene_preserving_quiet_zone": (
                            row["quiet_zone_delivery_guard_pass"]
                            if variant.endswith("scene_qz")
                            else True
                        ),
                    },
                }
            else:
                guard = _parent_visual_guard(
                    row=row,
                    stage1_quality=stage1_quality,
                    scene_qz_guard=(
                        row["quiet_zone_delivery_guard_pass"]
                        if variant.endswith("scene_qz")
                        else None
                    ),
                )
            row["visual_guard_pass"] = bool(guard["passed"])
            row["visual_guard_checks"] = guard["checks"]
            row["visual_guard_policy"] = guard.get(
                "policy",
                "e046-stage1-baseline-v1",
            )
            row["stage1_quality_deltas_diagnostic_only"] = guard.get(
                "stage1_quality_deltas_diagnostic_only",
                {},
            )

            # Stage1 is a dataset baseline only. Scene-preserving variants remain
            # diagnostic until physical-phone validation. The exported Stage2
            # latent pairs only with stage2_raw.
            row["eligible_for_refinement"] = bool(
                row["stage"] == "stage2"
                and row["quiet_zone_variant"] == "raw"
                and row["visual_guard_pass"]
                and not row["uniform_quiet_zone_replacement"]
            )
            row["eligible_final"] = bool(
                row["stage"] == "stage2"
                and row["quiet_zone_variant"] == "raw"
                and row["visual_guard_pass"]
                and not row["uniform_quiet_zone_replacement"]
            )
            rows.append(row)

        atomic_write_json(scoring_dir / "comparison.json", rows)
        csv_rows = []
        for row in rows:
            flat = dict(row)
            for key in (
                "module_error_breakdown",
                "quiet_zone_metrics",
                "visual_guard_checks",
            ):
                flat[key] = json.dumps(
                    flat.get(key),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            csv_rows.append(flat)
        _write_csv(scoring_dir / "comparison.csv", csv_rows)
        summary = {
            "candidate_id": candidate_id,
            "row_count": len(rows),
            "maximum_wechat_exact_presets": max(
                int(row["wechat_exact_presets"]) for row in rows
            ),
            "stage2_safe_variant_count": sum(
                bool(row["eligible_for_refinement"]) for row in rows
            ),
            "scoring_complete": True,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(complete_path, summary)
        return summary
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="score-parent",
            task_id=candidate_id,
            error=exc,
        )
        raise


def score_all_parents(
    *,
    output_root: Path,
    plan_id: str,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    results = []
    for item in plan["candidates"]:
        results.append(
            score_parent(
                output_root=output_root,
                plan_id=plan_id,
                candidate_id=str(item["id"]),
            )
        )
    summary = {
        "plan_id": plan_id,
        "candidate_count": len(results),
        "scored_count": sum(bool(item.get("scoring_complete")) for item in results),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "PARENT_SCORING_COMPLETE.json", summary)
    return summary


def _row_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row.get("visual_guard_pass"))),
        -int(row.get("wechat_exact_presets", 0)),
        -int(bool(row.get("wechat_original_exact"))),
        # Never alter the peripheral artwork without a measurable WeChat gain.
        -int(str(row.get("quiet_zone_variant") or "raw") == "raw"),
        -int(bool(row.get("quiet_zone_delivery_guard_pass"))),
        -float(row.get("clip_aesthetic") or -1e9),
        -float(row.get("hpsv2_1") or -1e9),
        -float(
            row["clip_score"]
            if row.get("clip_score") is not None
            else -1e9
        ),
        float(
            row["module_error_rate"]
            if row.get("module_error_rate") is not None
            else 1e9
        ),
    )


def select_parents(
    *,
    output_root: Path,
    plan_id: str,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    if not (plan_dir / "PARENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("parent scoring is not complete")

    best_by_candidate: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        scoring_path = (
            _parent_final_dir(plan_dir, str(candidate["id"]))
            / "scoring/comparison.json"
        )
        rows = _load_json(scoring_path)
        options = [
            row
            for row in rows
            if row["stage"] == "stage2"
            and row.get("eligible_for_refinement") is True
        ]
        if not options:
            options = [row for row in rows if row["stage"] == "stage2"]
        best = sorted(options, key=_row_rank)[0]
        best_by_candidate.append(best)

    safe = [row for row in best_by_candidate if bool(row["visual_guard_pass"])]
    if not safe:
        raise RuntimeError("E046 has no visually-safe Stage2 parent")
    safe_sorted = sorted(safe, key=_row_rank)
    count = int(plan["selected_parent_count"])

    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for row in safe_sorted:
        if str(row["prompt_id"]) in used_prompts:
            continue
        selected.append({**row, "selection_reason": "best_scan_diverse_prompt"})
        used_prompts.add(str(row["prompt_id"]))
        if len(selected) >= max(1, count - 1):
            break

    remaining = [
        row
        for row in safe
        if str(row["candidate_id"])
        not in {str(item["candidate_id"]) for item in selected}
    ]
    if len(selected) < count and remaining:
        exact_values = sorted(
            int(row["wechat_exact_presets"]) for row in remaining
        )
        median_exact = exact_values[len(exact_values) // 2]
        frontier = sorted(
            remaining,
            key=lambda row: (
                abs(int(row["wechat_exact_presets"]) - median_exact),
                -float(row.get("clip_aesthetic") or -1e9),
                -float(row.get("hpsv2_1") or -1e9),
            ),
        )[0]
        selected.append({**frontier, "selection_reason": "frontier_learning_case"})

    for row in safe_sorted:
        if len(selected) >= count:
            break
        if str(row["candidate_id"]) in {
            str(item["candidate_id"]) for item in selected
        }:
            continue
        selected.append({**row, "selection_reason": "fill_best_remaining"})

    selected = selected[:count]
    payload = {
        "experiment": EXPERIMENT,
        "plan_id": plan_id,
        "selected_parent_count": len(selected),
        "requested_parent_count": count,
        "selection_policy": (
            "safe Stage2; top WeChat exact with prompt diversity plus one frontier case"
        ),
        "selected": selected,
        "srmpgd_recipes": plan["srmpgd_recipes"],
        "created_at_utc": utc_now(),
        "production_ready": False,
    }
    atomic_write_json(plan_dir / "selected-parents.json", payload)
    _write_csv(plan_dir / "selected-parents.csv", selected)
    return payload


def _load_parent_artifact(
    *,
    plan_dir: Path,
    candidate: Mapping[str, Any],
) -> Any:
    from safetensors.torch import load_file
    from .e035_parent_artifact import LoadedParentArtifact

    root = _parent_final_dir(plan_dir, str(candidate["id"]))
    image = Image.open(root / "images/stage2-raw.png").convert("RGB")
    latent = load_file(
        str(root / "stage2-latent.safetensors"),
        device="cpu",
    )["latent"].detach().cpu().contiguous()
    metadata = _load_json(root / "parent-metadata.json")
    return LoadedParentArtifact(
        root=root,
        image=image,
        latent=latent,
        metadata={"source": metadata},
    )


def generate_refinement(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    srmpgd_recipe_id: str,
    source_commit: str,
) -> dict[str, Any]:
    import torch
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .e035_loss_fidelity import _offload_diffusion_modules
    from .e039_limiter_scanaware import E039Config
    from .e040_checkpoint_frontier import Recipe as E040Recipe, _run_trajectory

    source_commit = _validate_commit(source_commit)
    plan_dir, plan = load_plan(output_root, plan_id)
    if str(plan["source_commit"]) != source_commit:
        raise RuntimeError("E046 refinement source commit differs from plan")
    selected_path = plan_dir / "selected-parents.json"
    if not selected_path.is_file():
        raise FileNotFoundError("selected-parents.json missing")
    selected_ids = {
        str(item["candidate_id"])
        for item in _load_json(selected_path)["selected"]
    }
    if candidate_id not in selected_ids:
        raise ValueError(f"candidate was not selected for SR-MPGD: {candidate_id}")
    candidate = _candidate(plan, candidate_id)
    parent_recipe = parent_recipe_by_id(str(candidate["parent_recipe_id"]))
    recipe_spec = srmpgd_recipe_by_id(srmpgd_recipe_id)
    if srmpgd_recipe_id not in {
        str(item["id"]) for item in plan["srmpgd_recipes"]
    }:
        raise ValueError(f"SR-MPGD recipe not enabled in profile: {srmpgd_recipe_id}")

    final_dir = _refinement_final_dir(
        plan_dir,
        candidate_id,
        srmpgd_recipe_id,
    )
    complete_path = final_dir / "GENERATION_COMPLETE.json"
    if complete_path.is_file():
        return _load_json(complete_path)
    if final_dir.exists():
        raise FileExistsError(
            f"E046 refinement final directory exists without completion marker: {final_dir}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("E046 SR-MPGD refinement requires CUDA")

    attempt = (
        plan_dir
        / "attempts"
        / "refinements"
        / f"{candidate_id}-{srmpgd_recipe_id}-{source_commit[:12]}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        attempt / "task.json",
        {
            "experiment": EXPERIMENT,
            "kind": "srmpgd",
            "candidate": candidate,
            "parent_recipe": asdict(parent_recipe),
            "srmpgd_recipe": asdict(recipe_spec),
            "source_commit": source_commit,
            "started_at_utc": utc_now(),
        },
    )

    started = time.perf_counter()
    try:
        settings = _settings_for(candidate, parent_recipe)
        backend = UpstreamDiffQRCoderBackend(settings)
        pipeline = backend._load()
        blueprint = _blueprint(candidate, parent_recipe)
        parent = _load_parent_artifact(
            plan_dir=plan_dir,
            candidate=candidate,
        )
        original_vae_dtype = next(pipeline.vae.parameters()).dtype
        checkpointing_was_enabled = bool(
            getattr(pipeline.vae, "is_gradient_checkpointing", False)
        )
        enable_checkpointing = getattr(
            pipeline.vae,
            "enable_gradient_checkpointing",
            None,
        )
        disable_checkpointing = getattr(
            pipeline.vae,
            "disable_gradient_checkpointing",
            None,
        )
        config = E039Config(
            gamma=recipe_spec.gamma,
            lpips_weight=recipe_spec.lpips_weight,
            crop_padding_px=QR_PADDING_PX,
            qr_version=QR_VERSION,
            qr_mask_pattern=parent_recipe.qr_mask_pattern,
            qr_module_size=QR_MODULE_SIZE,
            quiet_zone_mode="none",
            quiet_zone_minimum_luminance=0.78,
            functional_pattern_tone_factor=0.0,
            max_backtracks=recipe_spec.max_backtracks,
            minimum_alpha=2**-12,
        )
        trajectory_recipe = E040Recipe(
            name=recipe_spec.id,
            latent_radius_rms=recipe_spec.latent_radius_rms,
            max_iterations=recipe_spec.max_iterations,
            lpips_budget=recipe_spec.lpips_budget,
            core_mae_budget=recipe_spec.core_mae_budget,
            full_module_weight=recipe_spec.full_module_weight,
        )

        checkpoints = []
        try:
            with _offload_diffusion_modules(pipeline) as offloaded:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                atomic_write_json(
                    attempt / "runtime.json",
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0),
                        "offloaded_modules": list(offloaded),
                        "vae_original_dtype": str(original_vae_dtype),
                        "vae_effective_dtype": str(
                            next(pipeline.vae.parameters()).dtype
                        ),
                    },
                )
                checkpoints = _run_trajectory(
                    pipeline=pipeline,
                    parent=parent,
                    blueprint=blueprint,
                    recipe=trajectory_recipe,
                    config=config,
                    output_root=attempt / "trajectory",
                )
        finally:
            pipeline.vae.to(dtype=original_vae_dtype)
            if not checkpointing_was_enabled and callable(disable_checkpointing):
                disable_checkpointing()
            gc.collect()
            torch.cuda.empty_cache()

        qz_dir = attempt / "scene-qz"
        qz_dir.mkdir(parents=True, exist_ok=True)
        qz_evidence: dict[str, Any] = {}
        for checkpoint in checkpoints:
            raw = Image.open(checkpoint.image_path).convert("RGB")
            scene, evidence = compose_scene_preserving_quiet_zone(
                raw,
                padding_px=QR_PADDING_PX,
            )
            scene_path = qz_dir / f"iteration-{checkpoint.iteration:03d}.png"
            _save_png(scene_path, scene)
            qz_evidence[f"i{checkpoint.iteration:03d}"] = {
                **evidence,
                "raw_image_relative_path": (
                    f"trajectory/{srmpgd_recipe_id}/images/"
                    f"iteration-{checkpoint.iteration:03d}.png"
                ),
                "scene_image_relative_path": (
                    f"scene-qz/iteration-{checkpoint.iteration:03d}.png"
                ),
            }
        atomic_write_json(attempt / "scene-qz-evidence.json", qz_evidence)

        metadata = {
            "experiment": EXPERIMENT,
            "kind": "srmpgd",
            "candidate_id": candidate_id,
            "srmpgd_recipe": asdict(recipe_spec),
            "source_commit": source_commit,
            "runtime_image": os.environ.get("PROOFTAG_RUNTIME_IMAGE"),
            "runtime_image_digest": os.environ.get("PROOFTAG_RUNTIME_IMAGE_DIGEST"),
            "parent_stage2_image_sha256": _image_sha256(parent.image),
            "parent_stage2_latent_file_sha256": sha256_file(
                _parent_final_dir(plan_dir, candidate_id)
                / "stage2-latent.safetensors"
            ),
            "checkpoint_count": len(checkpoints),
            "quiet_zone_mode_during_srmpgd": "none",
            "uniform_quiet_zone_replacement": False,
            "elapsed_s": time.perf_counter() - started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "refinement-metadata.json", metadata)
        atomic_write_json(
            attempt / "GENERATION_COMPLETE.json",
            {
                "experiment": EXPERIMENT,
                "candidate_id": candidate_id,
                "srmpgd_recipe_id": srmpgd_recipe_id,
                "checkpoint_count": len(checkpoints),
                "generation_complete": True,
                "completed_at_utc": utc_now(),
            },
        )

        expected_checkpoint_count = recipe_spec.max_iterations + 1
        if len(checkpoints) != expected_checkpoint_count:
            raise RuntimeError(
                f"E046 refinement produced {len(checkpoints)} checkpoints; "
                f"expected {expected_checkpoint_count}"
            )
        promoted = promote_attempt(
            attempt_dir=attempt,
            final_dir=final_dir,
            required_files=(
                f"trajectory/{srmpgd_recipe_id}/trace.json",
                f"trajectory/{srmpgd_recipe_id}/images/iteration-000.png",
                f"trajectory/{srmpgd_recipe_id}/latents/iteration-000.safetensors",
                "scene-qz-evidence.json",
                "refinement-metadata.json",
                "GENERATION_COMPLETE.json",
            ),
            metadata={
                "experiment": EXPERIMENT,
                "kind": "srmpgd",
                "candidate_id": candidate_id,
                "srmpgd_recipe_id": srmpgd_recipe_id,
                "source_commit": source_commit,
            },
        )
        result = _load_json(final_dir / "GENERATION_COMPLETE.json")
        result["promotion_manifest_hash"] = promoted["manifest_hash"]
        backend._pipeline = None
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="refinement",
            task_id=f"{candidate_id}__{srmpgd_recipe_id}",
            error=exc,
        )
        raise


def _refinement_visual_guard(
    *,
    row: Mapping[str, Any],
    parent_quality: Mapping[str, Any],
    recipe: SRMPGDRecipe,
    qz_guard: bool | None,
) -> dict[str, Any]:
    checks = {
        "lpips": float(row["lpips"]) <= recipe.lpips_budget,
        "core_mae": float(row["core_mean_absolute_change"]) <= recipe.core_mae_budget,
        "clipped_pixel_ratio_increase": (
            float(row["clipped_pixel_ratio_increase"]) <= 0.005
        ),
        "rgb_clipped_channel_ratio_increase": (
            float(row["rgb_clipped_channel_ratio_increase"]) <= 0.005
        ),
        "saturation_mean_change": (
            abs(float(row["saturation_mean_increase"])) <= 0.080
        ),
        "high_saturation_ratio_increase": (
            float(row["high_saturation_ratio_increase"]) <= 0.050
        ),
        "clip_score": (
            float(row.get("clip_score") or -1e9)
            >= float(parent_quality.get("clip_score") or -1e9) - 0.030
        ),
        "clip_aesthetic": (
            float(row.get("clip_aesthetic") or -1e9)
            >= float(parent_quality.get("clip_aesthetic") or -1e9) - 0.250
        ),
    }
    if (
        row.get("hpsv2_1") is not None
        and parent_quality.get("hpsv2_1") is not None
    ):
        checks["hpsv2_1"] = (
            float(row["hpsv2_1"])
            >= float(parent_quality["hpsv2_1"]) - 0.020
        )
    if qz_guard is not None:
        checks["scene_preserving_quiet_zone"] = bool(qz_guard)
    return {"passed": all(checks.values()), "checks": checks}


def score_refinement(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    srmpgd_recipe_id: str,
) -> dict[str, Any]:
    from .quality import image_change_metrics, image_quality_metrics

    plan_dir, plan = load_plan(output_root, plan_id)
    candidate = _candidate(plan, candidate_id)
    parent_recipe = parent_recipe_by_id(str(candidate["parent_recipe_id"]))
    recipe = srmpgd_recipe_by_id(srmpgd_recipe_id)
    root = _refinement_final_dir(plan_dir, candidate_id, srmpgd_recipe_id)
    if not (root / "GENERATION_COMPLETE.json").is_file():
        raise FileNotFoundError(
            f"refinement generation incomplete: {candidate_id}/{srmpgd_recipe_id}"
        )
    complete_path = root / "SCORING_COMPLETE.json"
    if complete_path.is_file():
        return _load_json(complete_path)

    try:
        trajectory = root / "trajectory" / srmpgd_recipe_id
        trace = _load_json(trajectory / "trace.json")
        refinement_metadata = _load_json(root / "refinement-metadata.json")
        qz_evidence = _load_json(root / "scene-qz-evidence.json")
        images: dict[str, Image.Image] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for step in trace:
            iteration = int(step["iteration"])
            raw_key = f"i{iteration:03d}_raw"
            scene_key = f"i{iteration:03d}_scene_qz"
            raw_path = trajectory / "images" / f"iteration-{iteration:03d}.png"
            scene_path = root / "scene-qz" / f"iteration-{iteration:03d}.png"
            images[raw_key] = Image.open(raw_path).convert("RGB")
            images[scene_key] = Image.open(scene_path).convert("RGB")
            metadata[raw_key] = {
                "iteration": iteration,
                "quiet_zone_variant": "raw",
                "image_path": str(raw_path),
                "latent_path": str(
                    trajectory
                    / "latents"
                    / f"iteration-{iteration:03d}.safetensors"
                ),
                "trace": step,
                "qz_evidence": None,
            }
            metadata[scene_key] = {
                "iteration": iteration,
                "quiet_zone_variant": "scene_preserving",
                "image_path": str(scene_path),
                "latent_path": str(
                    trajectory
                    / "latents"
                    / f"iteration-{iteration:03d}.safetensors"
                ),
                "trace": step,
                "qz_evidence": qz_evidence[f"i{iteration:03d}"],
            }

        settings = _quality_settings(candidate, parent_recipe)
        scoring_dir = root / "scoring"
        scoring_dir.mkdir(parents=True, exist_ok=True)
        qr, quality, _ = _score_image_set(
            output_dir=scoring_dir,
            images=images,
            prompt=str(candidate["prompt"]),
            payload=str(candidate["payload"]),
            settings=settings,
        )
        parent_image = Image.open(
            _parent_final_dir(plan_dir, candidate_id) / "images/stage2-raw.png"
        ).convert("RGB")
        parent_quality = _load_json(
            _parent_final_dir(plan_dir, candidate_id)
            / "scoring/quality-scores.json"
        )["stage2_raw"]
        blueprint = _blueprint(candidate, parent_recipe)
        rows: list[dict[str, Any]] = []
        for key, image in images.items():
            info = metadata[key]
            step = info["trace"]
            change = image_change_metrics(image, parent_image)
            qz = quiet_zone_metrics(image, padding_px=QR_PADDING_PX)
            core_change = _stage_core_change(
                image,
                parent_image,
                padding_px=QR_PADDING_PX,
            )
            qscore = quality[key]
            qz_guard = (
                bool(info["qz_evidence"]["delivery_guard_pass"])
                if info["qz_evidence"] is not None
                else None
            )
            row = {
                "experiment": EXPERIMENT,
                "plan_id": plan_id,
                "source_kind": "srmpgd",
                "source_commit": refinement_metadata.get("source_commit"),
                "runtime_image": refinement_metadata.get("runtime_image"),
                "runtime_image_digest": refinement_metadata.get("runtime_image_digest"),
                "candidate_id": candidate_id,
                "parent_recipe_id": parent_recipe.id,
                "srmpgd_recipe_id": srmpgd_recipe_id,
                "prompt_id": candidate["prompt_id"],
                "prompt_family": candidate["prompt_family"],
                "prompt_variant_index": candidate["prompt_variant_index"],
                "prompt": candidate["prompt"],
                "payload": candidate["payload"],
                "payload_sha256": hashlib.sha256(
                    str(candidate["payload"]).encode("utf-8")
                ).hexdigest(),
                "seed": candidate["seed"],
                "error_correction": parent_recipe.error_correction,
                "qr_version": QR_VERSION,
                "qr_mask_pattern": parent_recipe.qr_mask_pattern,
                "qr_module_size": QR_MODULE_SIZE,
                "qr_padding_px": QR_PADDING_PX,
                "stage": "srmpgd",
                "variant": key,
                "quiet_zone_variant": info["quiet_zone_variant"],
                "uniform_quiet_zone_replacement": False,
                "image_path": info["image_path"],
                "image_sha256": _image_sha256(image),
                "core_sha256": core_pixel_sha256(image, QR_PADDING_PX),
                "latent_path": info["latent_path"],
                "iteration": info["iteration"],
                "gamma": recipe.gamma,
                "latent_radius_rms": recipe.latent_radius_rms,
                "lpips": float(step.get("lpips_loss") or 0.0),
                "latent_delta_rms": step.get("latent_delta_rms"),
                "accepted_alpha": step.get("accepted_alpha"),
                "accepted_step_rms": step.get("accepted_step_rms"),
                "raw_step_rms": step.get("raw_step_rms"),
                "projected_step_rms": step.get("projected_step_rms"),
                "rejected_trial_count": step.get("rejected_trial_count"),
                "acceptance_reason": step.get("acceptance_reason"),
                "projection_was_active": bool(
                    step.get("raw_step_rms") is not None
                    and step.get("projected_step_rms") is not None
                    and float(step["raw_step_rms"])
                    > float(step["projected_step_rms"]) + 1e-9
                ),
                **_row_qr_fields(qr.get(key) or {}),
                **_module_fields(image, blueprint),
                **change,
                **core_change,
                **image_quality_metrics(image),
                **qscore,
                "quiet_zone_metrics": asdict(qz),
                "quiet_zone_delivery_guard_pass": qz_guard,
                "core_byte_identical_to_raw": (
                    compare_core_bytes(
                        image,
                        images[key.replace("_scene_qz", "_raw")],
                        padding_px=QR_PADDING_PX,
                    )
                    if key.endswith("_scene_qz")
                    else True
                ),
            }
            guard = _refinement_visual_guard(
                row=row,
                parent_quality=parent_quality,
                recipe=recipe,
                qz_guard=qz_guard,
            )
            row["visual_guard_pass"] = bool(guard["passed"])
            row["visual_guard_checks"] = guard["checks"]
            row["eligible_final"] = bool(
                row["quiet_zone_variant"] == "raw"
                and row["visual_guard_pass"]
                and not row["uniform_quiet_zone_replacement"]
            )
            rows.append(row)

        atomic_write_json(scoring_dir / "comparison.json", rows)
        csv_rows = []
        for row in rows:
            flat = dict(row)
            for field in (
                "module_error_breakdown",
                "quiet_zone_metrics",
                "visual_guard_checks",
            ):
                flat[field] = json.dumps(
                    flat.get(field),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            csv_rows.append(flat)
        _write_csv(scoring_dir / "comparison.csv", csv_rows)
        summary = {
            "candidate_id": candidate_id,
            "srmpgd_recipe_id": srmpgd_recipe_id,
            "row_count": len(rows),
            "maximum_wechat_exact_presets": max(
                int(row["wechat_exact_presets"]) for row in rows
            ),
            "safe_row_count": sum(bool(row["eligible_final"]) for row in rows),
            "scoring_complete": True,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(complete_path, summary)
        return summary
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="score-refinement",
            task_id=f"{candidate_id}__{srmpgd_recipe_id}",
            error=exc,
        )
        raise


def refinement_tasks(plan_dir: Path, plan: Mapping[str, Any]) -> list[tuple[str, str]]:
    selected_path = plan_dir / "selected-parents.json"
    if not selected_path.is_file():
        return []
    selected = _load_json(selected_path)["selected"]
    recipe_ids = [str(item["id"]) for item in plan["srmpgd_recipes"]]
    return [
        (str(parent["candidate_id"]), recipe_id)
        for parent in selected
        for recipe_id in recipe_ids
    ]


def score_all_refinements(
    *,
    output_root: Path,
    plan_id: str,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    tasks = refinement_tasks(plan_dir, plan)
    results = []
    for candidate_id, recipe_id in tasks:
        results.append(
            score_refinement(
                output_root=output_root,
                plan_id=plan_id,
                candidate_id=candidate_id,
                srmpgd_recipe_id=recipe_id,
            )
        )
    summary = {
        "plan_id": plan_id,
        "task_count": len(tasks),
        "scored_count": sum(bool(item.get("scoring_complete")) for item in results),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "REFINEMENT_SCORING_COMPLETE.json", summary)
    return summary


def _flatten_rows_for_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for field in (
            "module_error_breakdown",
            "quiet_zone_metrics",
            "visual_guard_checks",
        ):
            if isinstance(row.get(field), (dict, list)):
                row[field] = json.dumps(
                    row[field],
                    ensure_ascii=False,
                    sort_keys=True,
                )
        output.append(row)
    return output


def _pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in rows if bool(row.get("eligible_final"))]
    front: list[dict[str, Any]] = []
    for row in candidates:
        row_values = (
            float(row.get("wechat_exact_presets") or 0),
            float(row.get("clip_aesthetic") or -1e9),
            float(row.get("hpsv2_1") or -1e9),
            float(row.get("clip_score") or -1e9),
            -float(row.get("lpips") or 0),
        )
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_values = (
                float(other.get("wechat_exact_presets") or 0),
                float(other.get("clip_aesthetic") or -1e9),
                float(other.get("hpsv2_1") or -1e9),
                float(other.get("clip_score") or -1e9),
                -float(other.get("lpips") or 0),
            )
            if all(a >= b for a, b in zip(other_values, row_values)) and any(
                a > b for a, b in zip(other_values, row_values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=_row_rank)


def _contact_sheet(
    path: Path,
    items: Sequence[tuple[str, Image.Image, str]],
    *,
    columns: int = 4,
) -> None:
    if not items:
        return
    tile_w = 360
    image_h = 300
    label_h = 78
    rows = math.ceil(len(items) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_w, rows * (image_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image, subtitle) in enumerate(items):
        row, col = divmod(index, columns)
        preview = image.convert("RGB").copy()
        preview.thumbnail((tile_w - 12, image_h - 12), Image.Resampling.LANCZOS)
        x0 = col * tile_w
        y0 = row * (image_h + label_h)
        x = x0 + (tile_w - preview.width) // 2
        y = y0 + (image_h - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((x0 + 8, y0 + image_h + 4), label[:52], fill="black")
        draw.text(
            (x0 + 8, y0 + image_h + 26),
            subtitle[:58],
            fill=(45, 45, 45),
        )
        if len(subtitle) > 58:
            draw.text(
                (x0 + 8, y0 + image_h + 47),
                subtitle[58:116],
                fill=(45, 45, 45),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=False, compress_level=9)


def aggregate(
    *,
    output_root: Path,
    plan_id: str,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    if not (plan_dir / "PARENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("parent scoring incomplete")
    if not (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("refinement scoring incomplete")

    rows: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        rows.extend(
            _load_json(
                _parent_final_dir(plan_dir, str(candidate["id"]))
                / "scoring/comparison.json"
            )
        )
    for candidate_id, recipe_id in refinement_tasks(plan_dir, plan):
        rows.extend(
            _load_json(
                _refinement_final_dir(plan_dir, candidate_id, recipe_id)
                / "scoring/comparison.json"
            )
        )

    first_by_hash: dict[str, str] = {}
    for row in rows:
        image_hash = str(row["image_sha256"])
        row_id = stable_hash(
            {
                "candidate_id": row["candidate_id"],
                "source_kind": row["source_kind"],
                "variant": row["variant"],
                "srmpgd_recipe_id": row.get("srmpgd_recipe_id"),
            }
        )[:20]
        row["row_id"] = row_id
        duplicate_of = first_by_hash.get(image_hash)
        row["pixel_duplicate_of"] = duplicate_of
        row["pixel_duplicate"] = duplicate_of is not None
        if duplicate_of is None:
            first_by_hash[image_hash] = row_id

    safe = [
        row
        for row in rows
        if bool(row.get("eligible_final"))
        and not bool(row.get("uniform_quiet_zone_replacement"))
    ]
    if not safe:
        raise RuntimeError("E046 produced no safe final candidate")
    winner = sorted(safe, key=_row_rank)[0]
    pareto = _pareto_front(rows)

    dataset_dir = plan_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dataset_dir / "e046-observations.json", rows)
    with (dataset_dir / "e046-observations.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )
    _write_csv(
        dataset_dir / "e046-observations.csv",
        _flatten_rows_for_csv(rows),
    )

    dataset_hashes = {
        name: sha256_file(dataset_dir / name)
        for name in (
            "e046-observations.json",
            "e046-observations.jsonl",
            "e046-observations.csv",
        )
    }
    atomic_write_json(
        dataset_dir / "e047-training-contract.json",
        {
            "schema": "e047-training-contract-from-e046-v1",
            "experiment": EXPERIMENT,
            "plan_id": plan_id,
            "primary_target": "wechat_exact_presets",
            "secondary_targets": [
                "wechat_original_exact",
                "clip_aesthetic",
                "hpsv2_1",
                "clip_score",
                "module_error_rate",
                "visual_guard_pass",
            ],
            "dataset_hashes": dataset_hashes,
            "split_group_keys": [
                "prompt_id",
                "payload_sha256",
                "image_sha256",
                "candidate_id",
            ],
            "forbid_pixel_hash_across_splits": True,
            "forbid_prompt_id_across_train_test": True,
            "required_checkpoint_fields": [
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "gradient_scaler_state",
                "epoch",
                "global_step",
                "best_metric",
                "rng_python",
                "rng_numpy",
                "rng_torch_cpu",
                "rng_torch_cuda",
                "dataset_hash",
                "split_hash",
                "architecture_hash",
                "source_commit",
                "runtime_image_digest",
            ],
            "resume_requires_exact_hash_match": True,
            "oom_same_spec_retry": False,
            "automatic_training_authorized": False,
        },
    )

    atomic_write_json(dataset_dir / "pareto-front.json", pareto)
    _write_csv(dataset_dir / "pareto-front.csv", _flatten_rows_for_csv(pareto))

    best_by_prompt: list[dict[str, Any]] = []
    for prompt_id in sorted({str(row["prompt_id"]) for row in safe}):
        prompt_rows = [row for row in safe if str(row["prompt_id"]) == prompt_id]
        best_by_prompt.append(sorted(prompt_rows, key=_row_rank)[0])
    atomic_write_json(dataset_dir / "best-by-prompt.json", best_by_prompt)
    _write_csv(
        dataset_dir / "best-by-prompt.csv",
        _flatten_rows_for_csv(best_by_prompt),
    )

    # Stratified physical-phone queue: Pareto + prompt winners + attractive
    # hard negatives. It contains no fabricated phone label.
    phone_sample: list[dict[str, Any]] = []
    seen_phone_hashes: set[str] = set()

    def add_phone_row(row: Mapping[str, Any], reason: str) -> None:
        image_hash = str(row["image_sha256"])
        if image_hash in seen_phone_hashes or len(phone_sample) >= 32:
            return
        seen_phone_hashes.add(image_hash)
        phone_sample.append(
            {
                **dict(row),
                "phone_sample_reason": reason,
                "phone_label_status": "pending_physical_capture",
                "phone_attempts": 0,
                "phone_successes": 0,
            }
        )

    for row in best_by_prompt:
        add_phone_row(row, "best_by_prompt")
    for row in pareto:
        add_phone_row(row, "pareto")
    hard_negatives = sorted(
        [
            row
            for row in rows
            if bool(row.get("eligible_final"))
            and int(row.get("wechat_exact_presets", 0)) <= 5
        ],
        key=lambda row: (
            -float(row.get("clip_aesthetic") or -1e9),
            -float(row.get("hpsv2_1") or -1e9),
        ),
    )
    for row in hard_negatives:
        add_phone_row(row, "aesthetic_hard_negative")
    atomic_write_json(dataset_dir / "phone-sample-pending.json", phone_sample)
    _write_csv(
        dataset_dir / "phone-sample-pending.csv",
        _flatten_rows_for_csv(phone_sample),
    )

    pipeline = plan_dir / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    winner_image = Image.open(str(winner["image_path"])).convert("RGB")
    _save_png(pipeline / "99-FINAL-QR.png", winner_image)
    if winner.get("latent_path") and Path(str(winner["latent_path"])).is_file():
        shutil.copy2(
            Path(str(winner["latent_path"])),
            pipeline / "99-FINAL-latent.safetensors",
        )
    atomic_write_json(
        pipeline / "99-FINAL-metadata.json",
        {
            "winner": winner,
            "phone_validated": False,
            "production_ready": False,
            "uniform_quiet_zone_replacement": False,
        },
    )

    best_items = []
    for row in best_by_prompt:
        best_items.append(
            (
                str(row["prompt_id"]),
                Image.open(str(row["image_path"])).convert("RGB"),
                (
                    f"WeChat={row['wechat_exact_presets']}/37 "
                    f"{row['source_kind']} {row['variant']}"
                ),
            )
        )
    _contact_sheet(
        pipeline / "best-by-prompt-contact-sheet.png",
        best_items,
        columns=4,
    )

    pareto_items = []
    for row in pareto[:24]:
        pareto_items.append(
            (
                f"{row['candidate_id']} · {row['variant']}",
                Image.open(str(row["image_path"])).convert("RGB"),
                (
                    f"WeChat={row['wechat_exact_presets']}/37 "
                    f"AES={float(row.get('clip_aesthetic') or 0):.3f}"
                ),
            )
        )
    _contact_sheet(
        pipeline / "pareto-contact-sheet.png",
        pareto_items,
        columns=4,
    )

    phone_items = [
        (
            f"{row['prompt_id']} · {row['phone_sample_reason']}",
            Image.open(str(row["image_path"])).convert("RGB"),
            (
                f"WeChat={row['wechat_exact_presets']}/37 "
                f"{row['source_kind']} {row['variant']}"
            ),
        )
        for row in phone_sample
    ]
    _contact_sheet(
        pipeline / "phone-sample-contact-sheet.png",
        phone_items,
        columns=4,
    )

    exact_histogram = {
        f"{lower:02d}-{upper:02d}": sum(
            lower <= int(row["wechat_exact_presets"]) <= upper for row in rows
        )
        for lower, upper in ((0, 5), (6, 15), (16, 25), (26, 35), (36, 37))
    }
    verdict = {
        "experiment": EXPERIMENT,
        "plan_id": plan_id,
        "profile": plan["profile"],
        "source_commit": plan["source_commit"],
        "e045_plan_id": plan["e045_plan_id"],
        "qr_software_engine": QR_SOFTWARE_ENGINE,
        "qr_primary_label": "exact payload only",
        "row_count": len(rows),
        "unique_pixel_hash_count": len(first_by_hash),
        "pixel_duplicate_count": sum(bool(row["pixel_duplicate"]) for row in rows),
        "safe_final_row_count": len(safe),
        "pareto_row_count": len(pareto),
        "phone_sample_pending_count": len(phone_sample),
        "wechat_bucket_counts": exact_histogram,
        "winner_candidate_id": winner["candidate_id"],
        "winner_source_kind": winner["source_kind"],
        "winner_variant": winner["variant"],
        "winner_srmpgd_recipe_id": winner.get("srmpgd_recipe_id"),
        "winner_iteration": winner.get("iteration"),
        "winner_gamma": winner.get("gamma"),
        "winner_wechat_exact_presets": winner["wechat_exact_presets"],
        "winner_wechat_exact_rate": winner["wechat_exact_rate"],
        "winner_wechat_original_exact": winner["wechat_original_exact"],
        "winner_clip_aesthetic": winner.get("clip_aesthetic"),
        "winner_hpsv2_1": winner.get("hpsv2_1"),
        "winner_clip_score": winner.get("clip_score"),
        "winner_uniform_quiet_zone_replacement": False,
        "software_dataset_complete": True,
        "software_advisor_training_candidate": (
            plan["profile"] in {"pilot", "full"} and len(rows) >= 40
        ),
        "automatic_advisor_training_authorized": False,
        "phone_truth_available": False,
        "phone_surrogate_training_authorized": False,
        "production_ready": False,
        "next_action": (
            "REVIEW_E046_NOTEBOOK_THEN_FREEZE_E047_TRAIN_SPLITS_AND_PHONE_SAMPLE"
        ),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "verdict.json", verdict)
    atomic_write_text(
        plan_dir / "report.md",
        "\n".join(
            [
                "# E046 — controlled best-generator dataset",
                "",
                f"- profile: **{plan['profile']}**",
                f"- rows: **{len(rows)}**",
                f"- unique rasters: **{len(first_by_hash)}**",
                f"- Pareto rows: **{len(pareto)}**",
                (
                    "- winner WeChat exact: "
                    f"**{winner['wechat_exact_presets']}/37**"
                ),
                f"- winner: `{winner['candidate_id']} / {winner['variant']}`",
                "- uniform quiet-zone replacement: **forbidden**",
                "- phone validated: **no**",
                "- production ready: **no**",
                "",
                "The dataset is eligible for E047 review, not automatic training.",
            ]
        ),
    )

    manifest = [
        item
        for item in build_artifact_manifest(plan_dir)
        if item["path"] not in {"artifact-manifest.json", "COMPLETE.json"}
        and "/qr-verify-cache/" not in f"/{item['path']}/"
    ]
    atomic_write_json(plan_dir / "artifact-manifest.json", manifest)
    complete = {
        **verdict,
        "complete": True,
        "artifact_manifest_sha256": sha256_file(
            plan_dir / "artifact-manifest.json"
        ),
    }
    atomic_write_json(plan_dir / "COMPLETE.json", complete)
    atomic_write_json(
        output_root / "LATEST.json",
        {
            "schema": "e046-latest-v1",
            "plan_id": plan_id,
            "plan_dir": str(plan_dir),
            "profile": plan["profile"],
            "source_commit": plan["source_commit"],
            "status": "complete",
            "complete_path": str(plan_dir / "COMPLETE.json"),
            "updated_at_utc": utc_now(),
        },
    )
    return complete


def status(
    *,
    output_root: Path,
    plan_id: str | None = None,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    parent_generated = []
    parent_scored = []
    for candidate in plan["candidates"]:
        cid = str(candidate["id"])
        root = _parent_final_dir(plan_dir, cid)
        if (root / "GENERATION_COMPLETE.json").is_file():
            parent_generated.append(cid)
        if (root / "SCORING_COMPLETE.json").is_file():
            parent_scored.append(cid)

    tasks = refinement_tasks(plan_dir, plan)
    refinement_generated = [
        [cid, rid]
        for cid, rid in tasks
        if (
            _refinement_final_dir(plan_dir, cid, rid)
            / "GENERATION_COMPLETE.json"
        ).is_file()
    ]
    refinement_scored = [
        [cid, rid]
        for cid, rid in tasks
        if (
            _refinement_final_dir(plan_dir, cid, rid)
            / "SCORING_COMPLETE.json"
        ).is_file()
    ]
    return {
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "profile": plan["profile"],
        "plan_dir": str(plan_dir),
        "parent_total": len(plan["candidates"]),
        "parent_generated": len(parent_generated),
        "parent_scored": len(parent_scored),
        "selection_complete": (plan_dir / "selected-parents.json").is_file(),
        "refinement_total": len(tasks),
        "refinement_generated": len(refinement_generated),
        "refinement_scored": len(refinement_scored),
        "aggregate_complete": (plan_dir / "COMPLETE.json").is_file(),
        "failure_file_count": len(list((plan_dir / "failures").glob("*.json"))),
        "pending_parents": [
            str(item["id"])
            for item in plan["candidates"]
            if str(item["id"]) not in parent_generated
        ],
        "pending_refinements": [
            [cid, rid]
            for cid, rid in tasks
            if [cid, rid] not in refinement_generated
        ],
    }


def verify(
    *,
    output_root: Path,
    plan_id: str | None = None,
) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    complete_path = plan_dir / "COMPLETE.json"
    manifest_path = plan_dir / "artifact-manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("E046 is not complete")
    manifest = _load_json(manifest_path)
    missing = []
    mismatched = []
    for item in manifest:
        path = plan_dir / str(item["path"])
        if not path.is_file():
            missing.append(str(item["path"]))
        elif sha256_file(path) != str(item["sha256"]):
            mismatched.append(str(item["path"]))
    result = {
        "plan_id": plan["plan_id"],
        "plan_dir": str(plan_dir),
        "manifest_entry_count": len(manifest),
        "missing": missing,
        "mismatched": mismatched,
        "valid": not missing and not mismatched,
    }
    if not result["valid"]:
        raise RuntimeError(f"E046 manifest invalid: {result}")
    return result


def list_parent_ids(
    *,
    output_root: Path,
    plan_id: str,
    pending_only: bool,
) -> list[str]:
    plan_dir, plan = load_plan(output_root, plan_id)
    result = []
    for item in plan["candidates"]:
        cid = str(item["id"])
        if pending_only and (
            _parent_final_dir(plan_dir, cid) / "GENERATION_COMPLETE.json"
        ).is_file():
            continue
        result.append(cid)
    return result


def list_refinement_ids(
    *,
    output_root: Path,
    plan_id: str,
    pending_only: bool,
) -> list[tuple[str, str]]:
    plan_dir, plan = load_plan(output_root, plan_id)
    result = []
    for cid, rid in refinement_tasks(plan_dir, plan):
        if pending_only and (
            _refinement_final_dir(plan_dir, cid, rid)
            / "GENERATION_COMPLETE.json"
        ).is_file():
            continue
        result.append((cid, rid))
    return result



def reclassify_existing(
    *,
    output_root: Path,
    plan_id: str,
) -> dict[str, Any]:
    """Reclassify existing E046 scoring rows without regenerating any raster.

    This is a deterministic metadata migration for smoke plans created before
    parent-guard-v2/final-eligibility-v2. It consumes persisted QR-Verify
    evidence and quality metrics, rewrites only scoring tables/flags, removes
    derived aggregate outputs, then lets `aggregate` rebuild them.
    """
    plan_dir, plan = load_plan(output_root, plan_id)
    changed_rows = 0
    parent_rows = 0
    refinement_rows = 0

    for candidate in plan["candidates"]:
        candidate_id = str(candidate["id"])
        root = _parent_final_dir(plan_dir, candidate_id)
        comparison_path = root / "scoring/comparison.json"
        if not comparison_path.is_file():
            continue

        rows = _load_json(comparison_path)
        qr_evidence = _load_json(root / "scoring/qr-verify-evidence.json")
        quality = _load_json(root / "scoring/quality-scores.json")
        stage1_quality = quality["stage1_raw"]

        for row in rows:
            variant = str(row["variant"])
            qitem = qr_evidence.get(variant) or {}
            qr_fields = _row_qr_fields(qitem)
            row.update(qr_fields)

            if row["stage"] == "stage1":
                row["visual_guard_policy"] = "e046-stage1-baseline-v1"
                row["eligible_for_refinement"] = False
                row["eligible_final"] = False
            else:
                guard = _parent_visual_guard(
                    row=row,
                    stage1_quality=stage1_quality,
                    scene_qz_guard=(
                        row.get("quiet_zone_delivery_guard_pass")
                        if row.get("quiet_zone_variant") == "scene_preserving"
                        else None
                    ),
                )
                row["visual_guard_pass"] = bool(guard["passed"])
                row["visual_guard_checks"] = guard["checks"]
                row["visual_guard_policy"] = guard["policy"]
                row["stage1_quality_deltas_diagnostic_only"] = guard[
                    "stage1_quality_deltas_diagnostic_only"
                ]
                row["eligible_for_refinement"] = bool(
                    row["quiet_zone_variant"] == "raw"
                    and row["visual_guard_pass"]
                )
                row["eligible_final"] = bool(
                    row["quiet_zone_variant"] == "raw"
                    and row["visual_guard_pass"]
                )
            changed_rows += 1
            parent_rows += 1

        atomic_write_json(comparison_path, rows)
        _write_csv(
            root / "scoring/comparison.csv",
            _flatten_rows_for_csv(rows),
        )
        summary_path = root / "SCORING_COMPLETE.json"
        if summary_path.is_file():
            summary = _load_json(summary_path)
            summary["stage2_safe_variant_count"] = sum(
                bool(row.get("eligible_for_refinement"))
                for row in rows
                if row.get("stage") == "stage2"
            )
            summary["visual_guard_policy"] = (
                "e046-stage2-nondegenerate-parent-guard-v2"
            )
            summary["reclassified_at_utc"] = utc_now()
            atomic_write_json(summary_path, summary)

    for candidate_id, recipe_id in refinement_tasks(plan_dir, plan):
        root = _refinement_final_dir(plan_dir, candidate_id, recipe_id)
        comparison_path = root / "scoring/comparison.json"
        if not comparison_path.is_file():
            continue

        rows = _load_json(comparison_path)
        qr_evidence = _load_json(root / "scoring/qr-verify-evidence.json")
        for row in rows:
            variant = str(row["variant"])
            row.update(_row_qr_fields(qr_evidence.get(variant) or {}))
            row["eligible_final"] = bool(
                row.get("quiet_zone_variant") == "raw"
                and row.get("visual_guard_pass") is True
                and not row.get("uniform_quiet_zone_replacement", False)
            )
            changed_rows += 1
            refinement_rows += 1

        atomic_write_json(comparison_path, rows)
        _write_csv(
            root / "scoring/comparison.csv",
            _flatten_rows_for_csv(rows),
        )

    # Selection must be rebuilt because parent eligibility changed.
    for path in (
        plan_dir / "selected-parents.json",
        plan_dir / "selected-parents.csv",
    ):
        path.unlink(missing_ok=True)
    select_parents(output_root=output_root, plan_id=plan_id)

    # Derived products are rebuilt from the migrated scoring evidence.
    for path in (
        plan_dir / "verdict.json",
        plan_dir / "report.md",
        plan_dir / "artifact-manifest.json",
        plan_dir / "COMPLETE.json",
    ):
        path.unlink(missing_ok=True)
    shutil.rmtree(plan_dir / "dataset", ignore_errors=True)
    shutil.rmtree(plan_dir / "pipeline", ignore_errors=True)
    (plan_dir / "pipeline").mkdir(parents=True, exist_ok=True)

    result = aggregate(output_root=output_root, plan_id=plan_id)
    return {
        "plan_id": plan_id,
        "parent_rows_reclassified": parent_rows,
        "refinement_rows_reclassified": refinement_rows,
        "changed_rows": changed_rows,
        "winner_candidate_id": result["winner_candidate_id"],
        "winner_variant": result["winner_variant"],
        "winner_wechat_exact_presets": result["winner_wechat_exact_presets"],
        "winner_wechat_original_exact": result["winner_wechat_original_exact"],
        "manifest_sha256": result["artifact_manifest_sha256"],
        "valid": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--e045-root", type=Path, default=DEFAULT_E045_ROOT)
    plan.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    plan.add_argument("--source-commit", required=True)

    for name in (
        "generate-parent",
        "score-parent",
        "generate-refinement",
        "score-refinement",
        "score-parents",
        "select",
        "score-refinements",
        "aggregate",
        "reclassify-existing",
        "status",
        "verify",
        "list-parents",
        "list-refinements",
    ):
        command = sub.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--plan-id", required=False)
        if name in {"generate-parent", "score-parent", "generate-refinement", "score-refinement"}:
            command.add_argument("--candidate-id", required=True)
        if name in {"generate-refinement", "score-refinement"}:
            command.add_argument("--srmpgd-recipe-id", required=True)
        if name in {"generate-parent", "generate-refinement"}:
            command.add_argument("--source-commit", required=True)
        if name in {"list-parents", "list-refinements"}:
            command.add_argument("--pending-only", action="store_true")

    return parser


def _require_plan_id(args: Any) -> str:
    if args.plan_id:
        return str(args.plan_id)
    latest = _load_json(args.output_root / "LATEST.json")
    return str(latest["plan_id"])


def _cli() -> int:
    args = _parser().parse_args()
    action = args.action
    if action == "plan":
        result = create_plan(
            output_root=args.output_root,
            profile=args.profile,
            source_commit=args.source_commit,
            e045_root=args.e045_root,
        )
    else:
        plan_id = _require_plan_id(args)
        if action == "generate-parent":
            result = generate_parent(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                source_commit=args.source_commit,
            )
        elif action == "score-parent":
            result = score_parent(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
            )
        elif action == "score-parents":
            result = score_all_parents(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "select":
            result = select_parents(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "generate-refinement":
            result = generate_refinement(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                srmpgd_recipe_id=args.srmpgd_recipe_id,
                source_commit=args.source_commit,
            )
        elif action == "score-refinement":
            result = score_refinement(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                srmpgd_recipe_id=args.srmpgd_recipe_id,
            )
        elif action == "score-refinements":
            result = score_all_refinements(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "aggregate":
            result = aggregate(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "reclassify-existing":
            result = reclassify_existing(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "status":
            result = status(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "verify":
            result = verify(
                output_root=args.output_root,
                plan_id=plan_id,
            )
        elif action == "list-parents":
            for candidate_id in list_parent_ids(
                output_root=args.output_root,
                plan_id=plan_id,
                pending_only=args.pending_only,
            ):
                print(candidate_id)
            return 0
        elif action == "list-refinements":
            for candidate_id, recipe_id in list_refinement_ids(
                output_root=args.output_root,
                plan_id=plan_id,
                pending_only=args.pending_only,
            ):
                print(f"{candidate_id}\t{recipe_id}")
            return 0
        else:
            raise AssertionError(action)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
