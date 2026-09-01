"""E041 — gamma ablation + selective QR functional-pattern reinforcement.

E041 deliberately changes the prompt and therefore creates a fresh Stage-1/Stage-2
parent.  All gamma candidates are nevertheless paired on that *same* fresh parent,
so the internal gamma comparison remains controlled.  E040 is kept only as a
historical control because its greenhouse parent uses a different prompt.

Phase A varies gamma under the E039/E040 scan-aware-v2 objective with radius=.20 and
scores every checkpoint i0..i8.  Phase B takes the three best visually-safe Phase-A
checkpoints (distinct gammas) and applies only the project's existing selective
functional-pattern tone repair.  Data modules are never projected back onto the
artwork.  QR-Verify remains authoritative; advisor/E016 outputs are metadata only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .e035_loss_fidelity import (
    _atomic_json,
    _atomic_text,
    _image_sha256,
    _offload_diffusion_modules,
    _score_qr_verify,
)
from .e035_parent_artifact import LoadedParentArtifact, sha256_file, tensor_sha256
from .e035_parent_capture import (
    CaptureConfig,
    DEFAULT_NEGATIVE_PROMPT,
    UPSTREAM_REVISION,
    _settings_document,
)
from .e038_recipe_frontier import _comparison_sheet, _qr_original_exact, _score_quality, _visual_guard
from .e039_limiter_scanaware import E039Config
from .e040_checkpoint_frontier import Recipe as E040Recipe, _run_trajectory
from .e040_model_bridge import advisor_preview, score_surrogate_images

EXPERIMENT = "e041-gamma-functional-pattern-frontier-v1"
E040_REQUIRED_EXPERIMENT = "e040-srmpgd-checkpoint-frontier-v1"
PAYLOAD = "https://ptag.io/t/e041"
PROMPT = (
    "a sunlit botanical reading room inside a glass conservatory, oak shelves, "
    "climbing vines, terracotta pots and a small writing desk, refined editorial "
    "interior photograph"
)
SEED = 71041
ERROR_CORRECTION = "M"
QR_VERSION = 3
QR_MASK_PATTERN = 4
QR_MODULE_SIZE = 20
QR_PADDING_PX = 78
LATENT_RADIUS_RMS = 0.200
MAX_ITERATIONS = 8
GAMMAS = (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)
FUNCTIONAL_TONE_FACTORS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30)
PHASE_B_BASE_COUNT = 3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=False, compress_level=9)


def _load_e040_control(path: Path) -> dict[str, Any]:
    verdict_path = path / "verdict.json"
    if not verdict_path.is_file():
        raise FileNotFoundError(f"E040 verdict missing: {verdict_path}")
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict.get("experiment") != E040_REQUIRED_EXPERIMENT:
        raise RuntimeError(f"unexpected E040 experiment: {verdict.get('experiment')}")
    if verdict.get("gamma_preserved") is not True:
        raise RuntimeError("E041 requires the finalized E040 control")
    if int(verdict.get("checkpoint_count", 0)) != 45:
        raise RuntimeError("E041 requires finalized E040 with 45 checkpoints")
    return verdict


def _generate_fresh_parent(
    *,
    backend: Any,
    blueprint: Any,
    output_dir: Path,
    source_commit: str,
) -> LoadedParentArtifact:
    import torch
    from safetensors.torch import save_file
    from .schemas import GenerationRequest

    root = output_dir / "parent"
    root.mkdir(parents=True, exist_ok=True)
    capture = CaptureConfig(payload=PAYLOAD, prompt=PROMPT, seed=SEED)
    request = GenerationRequest(
        payload=PAYLOAD,
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        backend="controlnet",
        error_correction=ERROR_CORRECTION,
        seed=SEED,
        steps=capture.steps,
        guidance_scale=capture.guidance_scale,
        controlnet_scale=capture.controlnet_scale,
        strength=capture.strength,
        max_attempts=1,
    )

    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    started = time.perf_counter()
    stage1 = backend.generate(request, blueprint, SEED)
    stage2 = backend._run_stage2(stage1, blueprint, request, SEED)
    state = backend.export_stage2_state()
    if state is None:
        raise RuntimeError("E041 Stage 2 produced no exportable latent")
    latent = state["latent"].detach().cpu().contiguous()
    latent_hash = tensor_sha256(latent)
    if latent_hash != str(state.get("latent_sha256") or latent_hash):
        raise RuntimeError("E041 Stage-2 latent hash mismatch")
    if stage1.size != (736, 736) or stage2.size != (736, 736):
        raise RuntimeError("E041 expects 736x736 Stage 1 and Stage 2")

    stage1_path = root / "stage1.png"
    stage2_path = root / "stage2.png"
    latent_path = root / "stage2-latent.safetensors"
    _save_png(stage1_path, stage1)
    _save_png(stage2_path, stage2)
    save_file({"latent": latent}, str(latent_path))
    metadata = {
        "experiment": EXPERIMENT,
        "payload": PAYLOAD,
        "prompt": PROMPT,
        "seed": SEED,
        "source_commit": source_commit,
        "diffqrcoder_revision": UPSTREAM_REVISION,
        "stage1_image_sha256": _image_sha256(stage1),
        "stage2_image_sha256": _image_sha256(stage2),
        "stage2_latent_tensor_sha256": latent_hash,
        "stage2_latent_file_sha256": sha256_file(latent_path),
        "elapsed_s": time.perf_counter() - started,
        "note": "fresh E041 parent; not paired with E040 greenhouse prompt",
    }
    _atomic_json(root / "parent-metadata.json", metadata)
    return LoadedParentArtifact(root=root, image=stage2.convert("RGB"), latent=latent, metadata={"source": metadata})


def _gamma_recipe(gamma: float) -> E040Recipe:
    token = str(int(gamma)).zfill(4)
    return E040Recipe(
        name=f"e041_gamma_{token}_r200_i08",
        latent_radius_rms=LATENT_RADIUS_RMS,
        max_iterations=MAX_ITERATIONS,
    )


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["qr_verify_exact_presets"]),
        -int(bool(row["original_exact"])),
        float(row.get("functional_center_error_rate", 1.0)),
        int(row.get("full_module_error_count", 10**9)),
        float(row.get("lpips", 1e9)),
        int(row.get("iteration", 10**9)),
    )


def _lpips_core_scores(reference: Image.Image, variants: dict[str, Image.Image]) -> dict[str, float]:
    import lpips
    import torch

    model = lpips.LPIPS(net="vgg", verbose=False).requires_grad_(False).eval().cpu()

    def tensor(image: Image.Image) -> Any:
        core = image.convert("RGB").crop(
            (QR_PADDING_PX, QR_PADDING_PX, image.width - QR_PADDING_PX, image.height - QR_PADDING_PX)
        )
        array = np.asarray(core, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)

    ref = tensor(reference)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for key, image in variants.items():
            scores[key] = float(model(tensor(image), ref).mean().cpu())
    return scores


def _score_rows(
    *,
    images: dict[str, Image.Image],
    metadata: dict[str, dict[str, Any]],
    output_dir: Path,
    backend: Any,
    blueprint: Any,
    parent_image: Image.Image,
    trace_lpips: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .quality import image_change_metrics, image_quality_metrics
    from .qr import diffqrcoder_module_error_rate, diffqrcoder_structure_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    qr_verify = _score_qr_verify(output_dir, PAYLOAD, images)
    quality_scores, quality_provenance = _score_quality(images, PROMPT, backend.settings)
    _atomic_json(output_dir / "quality-scores.json", quality_scores)
    _atomic_json(output_dir / "quality-provenance.json", quality_provenance)
    surrogate_scores, surrogate_status = score_surrogate_images(images)
    _atomic_json(output_dir / "e016-surrogate-scores.json", surrogate_scores)
    _atomic_json(output_dir / "e016-surrogate-status.json", surrogate_status)

    lpips_scores = trace_lpips or _lpips_core_scores(parent_image, images)
    parent_key = next(key for key, info in metadata.items() if info.get("is_parent_reference"))
    parent_quality = quality_scores.get(parent_key) or {
        "clip_score": 0.0,
        "clip_aesthetic": 0.0,
        "hpsv2_1": None,
    }
    guard_config = E039Config()
    rows: list[dict[str, Any]] = []
    for key, image in images.items():
        info = dict(metadata[key])
        qitem = (qr_verify or {}).get(key) or {}
        qscore = quality_scores.get(key) or {}
        surrogate = surrogate_scores.get(key) or {}
        change = image_change_metrics(image, parent_image)
        quality = image_quality_metrics(image)
        structure = diffqrcoder_structure_metrics(
            image,
            blueprint,
            padding_px=QR_PADDING_PX,
            module_size=QR_MODULE_SIZE,
        )
        module_rate = diffqrcoder_module_error_rate(
            image,
            blueprint,
            padding_px=QR_PADDING_PX,
            module_size=QR_MODULE_SIZE,
        )
        row = {
            "variant": key,
            **info,
            "qr_verify_exact_presets": int(qitem.get("conservative_exact_presets", 0)),
            "ssr": int(qitem.get("conservative_exact_presets", 0)) / 37.0,
            "original_exact": _qr_original_exact(qitem),
            "full_module_error_count": int(round(module_rate * 841)),
            "lpips": float(lpips_scores[key]),
            "clip_score": qscore.get("clip_score"),
            "clip_aesthetic": qscore.get("clip_aesthetic"),
            "hpsv2_1": qscore.get("hpsv2_1"),
            "surrogate_mean_success_probability": surrogate.get("mean_success_probability"),
            "surrogate_min_success_probability": surrogate.get("min_success_probability"),
            **structure,
            **change,
            **quality,
        }
        guard = _visual_guard(row, parent_quality, guard_config)
        row["visual_guard_pass"] = bool(guard["passed"])
        row["visual_guard_checks"] = guard["checks"]
        rows.append(row)

    _atomic_json(output_dir / "comparison.json", rows)
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["visual_guard_checks"] = json.dumps(flat["visual_guard_checks"], ensure_ascii=False, sort_keys=True)
        csv_rows.append(flat)
    _write_csv(output_dir / "comparison.csv", csv_rows)
    return rows, surrogate_status


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "e041-artifact-manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def run_e041(
    *,
    output_dir: Path,
    e040_results_dir: Path,
    source_commit: str,
) -> dict[str, Any]:
    import torch
    from .config import Settings
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .qr import generate_diffqrcoder_qr, prepare_scan_ready_image

    if not torch.cuda.is_available():
        raise RuntimeError("E041 requires CUDA")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E041 output must be empty: {output_dir}")
    if 1000.0 not in GAMMAS or len(set(GAMMAS)) != len(GAMMAS):
        raise RuntimeError("E041 gamma grid must contain the historical gamma=1000 baseline exactly once")

    output_dir.mkdir(parents=True, exist_ok=True)
    e040_control = _load_e040_control(e040_results_dir)
    _atomic_json(output_dir / "e040-historical-control.json", e040_control)
    _atomic_json(
        output_dir / "plan.json",
        {
            "experiment": EXPERIMENT,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_commit": source_commit,
            "payload": PAYLOAD,
            "prompt": PROMPT,
            "seed": SEED,
            "prompt_changed_from_e040": True,
            "e040_is_historical_not_paired": True,
            "gamma_grid": list(GAMMAS),
            "historical_gamma_baseline": 1000.0,
            "latent_radius_rms": LATENT_RADIUS_RMS,
            "max_iterations": MAX_ITERATIONS,
            "functional_tone_factors": list(FUNCTIONAL_TONE_FACTORS),
            "phase_b_base_count": PHASE_B_BASE_COUNT,
            "selection": "visual guard -> QR-Verify SSR -> original exact -> functional errors -> MER -> LPIPS",
            "production_ready": False,
            "generalization_authorized": False,
        },
    )

    capture = CaptureConfig(payload=PAYLOAD, prompt=PROMPT, seed=SEED)
    base = Settings()
    settings = Settings.model_validate({**base.model_dump(), **_settings_document(capture)})
    if str(settings.diffqrcoder_revision) != UPSTREAM_REVISION:
        raise RuntimeError("runtime DiffQRCoder revision differs from E041 preregistration")
    backend = UpstreamDiffQRCoderBackend(settings)
    pipeline = backend._load()
    blueprint = generate_diffqrcoder_qr(
        PAYLOAD,
        ERROR_CORRECTION,
        version=QR_VERSION,
        mask_pattern=QR_MASK_PATTERN,
        module_size=QR_MODULE_SIZE,
    )
    parent = _generate_fresh_parent(
        backend=backend,
        blueprint=blueprint,
        output_dir=output_dir,
        source_commit=source_commit,
    )

    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
    all_checkpoints: list[Any] = []
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            try:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                _atomic_json(
                    output_dir / "runtime.json",
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0),
                        "offloaded_modules": list(offloaded),
                        "vae_original_dtype": str(original_vae_dtype),
                        "gamma_grid": list(GAMMAS),
                    },
                )
                for gamma in GAMMAS:
                    recipe = _gamma_recipe(gamma)
                    config = E039Config(gamma=gamma)
                    all_checkpoints.extend(
                        _run_trajectory(
                            pipeline=pipeline,
                            parent=parent,
                            blueprint=blueprint,
                            recipe=recipe,
                            config=config,
                            output_root=output_dir / "phase-a-trajectories",
                        )
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                pipeline.vae.to(dtype=original_vae_dtype)
                if not checkpointing_was_enabled and callable(disable_checkpointing):
                    disable_checkpointing()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        if next(pipeline.vae.parameters()).dtype != original_vae_dtype:
            pipeline.vae.to(dtype=original_vae_dtype)
        gc.collect()
        torch.cuda.empty_cache()

    # Free the diffusion stack before CLIP/HPS/LPIPS quality scoring.
    backend._pipeline = None
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    phase_a_images: dict[str, Image.Image] = {}
    phase_a_meta: dict[str, dict[str, Any]] = {}
    trace_lpips: dict[str, float] = {}
    first_key: str | None = None
    for item in all_checkpoints:
        gamma = float(item.trace_step["gamma"])
        key = f"{item.recipe}__i{item.iteration:02d}"
        if first_key is None:
            first_key = key
        phase_a_images[key] = Image.open(item.image_path).convert("RGB")
        phase_a_meta[key] = {
            "phase": "A",
            "gamma": gamma,
            "iteration": int(item.iteration),
            "radius": LATENT_RADIUS_RMS,
            "image_path": item.image_path,
            "latent_path": item.latent_path,
            "raw_step_rms": item.trace_step.get("raw_step_rms"),
            "projected_step_rms": item.trace_step.get("projected_step_rms"),
            "accepted_step_rms": item.trace_step.get("accepted_step_rms"),
            "accepted_alpha": item.trace_step.get("accepted_alpha"),
            "latent_delta_rms": item.trace_step.get("latent_delta_rms"),
            "projection_was_active": bool(
                item.trace_step.get("raw_step_rms") is not None
                and item.trace_step.get("projected_step_rms") is not None
                and float(item.trace_step["raw_step_rms"]) > float(item.trace_step["projected_step_rms"]) + 1e-9
            ),
            "is_parent_reference": False,
        }
        trace_lpips[key] = float(item.trace_step["lpips_loss"])
    if first_key is None:
        raise RuntimeError("E041 Phase A generated no checkpoints")
    phase_a_meta[first_key]["is_parent_reference"] = True
    parent_image = phase_a_images[first_key]
    _save_png(output_dir / "parent/stage2-scan-ready.png", parent_image)

    phase_a_rows, phase_a_surrogate = _score_rows(
        images=phase_a_images,
        metadata=phase_a_meta,
        output_dir=output_dir / "phase-a-scoring",
        backend=backend,
        blueprint=blueprint,
        parent_image=parent_image,
        trace_lpips=trace_lpips,
    )

    best_per_gamma: list[dict[str, Any]] = []
    for gamma in GAMMAS:
        options = [row for row in phase_a_rows if math.isclose(float(row["gamma"]), gamma) and row["visual_guard_pass"]]
        if options:
            best_per_gamma.append(sorted(options, key=_rank_key)[0])
    if len(best_per_gamma) < PHASE_B_BASE_COUNT:
        raise RuntimeError("E041 has fewer than three visually-safe gamma candidates")
    best_per_gamma = sorted(best_per_gamma, key=_rank_key)
    phase_b_bases = best_per_gamma[:PHASE_B_BASE_COUNT]
    _atomic_json(output_dir / "phase-a-best-per-gamma.json", best_per_gamma)
    _atomic_json(output_dir / "phase-b-selected-bases.json", phase_b_bases)

    phase_b_images: dict[str, Image.Image] = {}
    phase_b_meta: dict[str, dict[str, Any]] = {}
    phase_b_root = output_dir / "phase-b-functional"
    for base_row in phase_b_bases:
        base_key = str(base_row["variant"])
        base_image = phase_a_images[base_key]
        for factor in FUNCTIONAL_TONE_FACTORS:
            factor_token = f"{int(round(factor * 100)):02d}"
            key = f"{base_key}__tone{factor_token}"
            if factor == 0.0:
                image = base_image.copy()
            else:
                image = prepare_scan_ready_image(
                    base_image,
                    blueprint,
                    quiet_zone_mode="adaptive_light",
                    quiet_zone_minimum_luminance=0.90,
                    functional_pattern_tone_factor=factor,
                )
            image_path = phase_b_root / base_key / f"tone-{factor_token}.png"
            _save_png(image_path, image)
            phase_b_images[key] = image
            phase_b_meta[key] = {
                "phase": "B",
                "base_checkpoint": base_key,
                "gamma": float(base_row["gamma"]),
                "iteration": int(base_row["iteration"]),
                "radius": LATENT_RADIUS_RMS,
                "functional_tone_factor": factor,
                "image_path": str(image_path),
                "latent_path": base_row["latent_path"],
                "latent_delta_rms": base_row.get("latent_delta_rms"),
                "is_parent_reference": False,
            }
    # Add the untouched parent solely as the visual-guard reference for Phase B scoring.
    reference_key = "phase-b-parent-reference"
    phase_b_images[reference_key] = parent_image
    phase_b_meta[reference_key] = {
        "phase": "B_REFERENCE",
        "gamma": 0.0,
        "iteration": 0,
        "radius": 0.0,
        "functional_tone_factor": 0.0,
        "image_path": str(output_dir / "parent/stage2-scan-ready.png"),
        "latent_path": str(output_dir / "parent/stage2-latent.safetensors"),
        "is_parent_reference": True,
    }

    phase_b_rows_all, phase_b_surrogate = _score_rows(
        images=phase_b_images,
        metadata=phase_b_meta,
        output_dir=output_dir / "phase-b-scoring",
        backend=backend,
        blueprint=blueprint,
        parent_image=parent_image,
        trace_lpips=None,
    )
    phase_b_rows = [row for row in phase_b_rows_all if row["phase"] == "B"]
    safe_b = [row for row in phase_b_rows if row["visual_guard_pass"]]
    if not safe_b:
        raise RuntimeError("E041 Phase B has no visually-safe candidate")
    winner = sorted(safe_b, key=_rank_key)[0]

    advisor = advisor_preview(
        prompt=PROMPT,
        payload_length=len(PAYLOAD),
        error_correction=ERROR_CORRECTION,
        qr_context={
            "qr_version": QR_VERSION,
            "qr_mask_pattern": QR_MASK_PATTERN,
            "qr_module_size": QR_MODULE_SIZE,
            "qr_padding_px": QR_PADDING_PX,
        },
    )
    _atomic_json(output_dir / "advisor-preview.json", advisor)

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    _save_png(pipeline_dir / "01-qr-reference.png", blueprint.image)
    _save_png(pipeline_dir / "02-control-condition.png", blueprint.image.resize((736, 736), Image.Resampling.NEAREST))
    shutil.copy2(output_dir / "parent/stage1.png", pipeline_dir / "03-stage1.png")
    shutil.copy2(output_dir / "parent/stage2.png", pipeline_dir / "04-stage2.png")
    shutil.copy2(output_dir / "parent/stage2-scan-ready.png", pipeline_dir / "05-stage2-scan-ready.png")
    final_image_path = pipeline_dir / "99-FINAL-QR.png"
    final_latent_path = pipeline_dir / "99-FINAL-latent.safetensors"
    shutil.copy2(Path(winner["image_path"]), final_image_path)
    shutil.copy2(Path(winner["latent_path"]), final_latent_path)
    _atomic_json(
        pipeline_dir / "99-FINAL-raster-postprocess.json",
        {
            "base_checkpoint": winner["base_checkpoint"],
            "selected_gamma": winner["gamma"],
            "functional_tone_factor": winner["functional_tone_factor"],
            "note": "final raster includes selective functional-pattern tone postprocess; data modules are not projected",
        },
    )

    sheet_items: list[tuple[str, Image.Image, str]] = [
        ("QR reference", blueprint.image.convert("RGB"), PAYLOAD),
        ("Stage 1", Image.open(output_dir / "parent/stage1.png").convert("RGB"), "fresh E041 prompt"),
        ("Stage 2", Image.open(output_dir / "parent/stage2.png").convert("RGB"), "fresh SRPG parent"),
        ("Stage 2 scan-ready", parent_image, "tone=0 baseline"),
    ]
    for row in phase_b_bases:
        sheet_items.append(
            (
                f"Gamma {int(row['gamma'])} i{int(row['iteration'])}",
                phase_a_images[str(row["variant"])],
                f"SSR={row['qr_verify_exact_presets']}/37 safe={row['visual_guard_pass']}",
            )
        )
    sheet_items.append(
        (
            "FINAL E041",
            Image.open(final_image_path).convert("RGB"),
            f"gamma={int(winner['gamma'])} tone={winner['functional_tone_factor']:.2f} SSR={winner['qr_verify_exact_presets']}/37",
        )
    )
    _comparison_sheet(pipeline_dir / "full-pipeline-contact-sheet.png", sheet_items, columns=4)

    projection_summary = []
    for gamma in GAMMAS:
        rows = [row for row in phase_a_rows if math.isclose(float(row["gamma"]), gamma)]
        projection_summary.append(
            {
                "gamma": gamma,
                "checkpoint_count": len(rows),
                "projection_active_count": sum(bool(row.get("projection_was_active")) for row in rows),
                "mean_accepted_alpha": float(np.mean([row["accepted_alpha"] for row in rows if row.get("accepted_alpha") is not None]))
                if any(row.get("accepted_alpha") is not None for row in rows)
                else None,
                "best_ssr_exact_presets": max(int(row["qr_verify_exact_presets"]) for row in rows),
            }
        )
    _atomic_json(output_dir / "gamma-projection-summary.json", projection_summary)

    verdict = {
        "experiment": EXPERIMENT,
        "prompt": PROMPT,
        "prompt_changed_from_e040": True,
        "e040_paired_comparison_allowed": False,
        "gamma_grid": list(GAMMAS),
        "historical_gamma_baseline": 1000.0,
        "selected_gamma": winner["gamma"],
        "selected_iteration": winner["iteration"],
        "selected_functional_tone_factor": winner["functional_tone_factor"],
        "winner_variant": winner["variant"],
        "winner_ssr_exact_presets": winner["qr_verify_exact_presets"],
        "winner_ssr": winner["ssr"],
        "winner_original_exact": winner["original_exact"],
        "winner_full_module_error_count": winner["full_module_error_count"],
        "winner_functional_center_error_rate": winner["functional_center_error_rate"],
        "winner_data_center_error_rate": winner["data_center_error_rate"],
        "winner_lpips": winner["lpips"],
        "winner_visual_guard_pass": winner["visual_guard_pass"],
        "winner_visual_guard_checks": winner["visual_guard_checks"],
        "phase_a_checkpoint_count": len(phase_a_rows),
        "phase_a_safe_checkpoint_count": sum(bool(row["visual_guard_pass"]) for row in phase_a_rows),
        "phase_b_variant_count": len(phase_b_rows),
        "phase_b_safe_variant_count": len(safe_b),
        "advisor_available": bool(advisor.get("available")),
        "e016_surrogate_research_usable": bool(phase_b_surrogate.get("research_usable") or phase_a_surrogate.get("research_usable")),
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": "REVIEW_GAMMA_EFFECT_AND_FUNCTIONAL_PATTERN_GAIN_BEFORE_ANY_GENERALIZATION",
    }
    _atomic_json(output_dir / "verdict.json", verdict)
    _atomic_text(
        output_dir / "report.md",
        "\n".join(
            [
                "# E041 — gamma + motifs fonctionnels",
                "",
                f"- nouveau prompt : `{PROMPT}`",
                f"- gamma testés : {', '.join(str(int(x)) for x in GAMMAS)}",
                "- gamma=1000 est un contrôle historique, pas une constante imposée",
                f"- rayon latent : {LATENT_RADIUS_RMS}",
                f"- meilleur gamma : **{winner['gamma']}**",
                f"- checkpoint : **i{winner['iteration']}**",
                f"- renforcement fonctionnel : **{winner['functional_tone_factor']}**",
                f"- SSR : **{winner['qr_verify_exact_presets']}/37**",
                f"- original exact : **{winner['original_exact']}**",
                f"- visual guard : **{winner['visual_guard_pass']}**",
                "- E040 est seulement historique car le prompt/parent E041 a changé",
                "- production/generalisation : **non**",
                "",
            ]
        ),
    )
    _atomic_json(output_dir / "e041-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--e040-results-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    verdict = run_e041(
        output_dir=args.output_dir,
        e040_results_dir=args.e040_results_dir,
        source_commit=args.source_commit,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
