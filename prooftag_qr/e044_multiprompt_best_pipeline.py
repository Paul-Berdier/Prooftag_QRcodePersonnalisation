"""E044 — multi-prompt benchmark of the strongest learned Prooftag pipeline.

E044 intentionally returns to a broad prompt screen instead of adding another loss.
Each prompt gets a fresh Stage-1/Stage-2 parent under one frozen pipeline, then two
paired SR-MPGD trajectories (gamma 500 and 1000) from exactly the same Stage-2
latent. Every i0..i8 checkpoint is persisted and rescored with the project's
authoritative conservative QR-Verify adapter and visual-quality guard.

The Stage-2 recipe is the frozen public-parent recipe used by E035:
public_random / binary_exact / SRPG 50:20. This is NOT presented as the exact
paper configuration. The official paper comparison is documented separately.

E044 fixes the DiffQRCoder raster geometry to 736 / padding 78 / core 580 and never
uses functional-pattern toning, E043 scanner-cell losses, or pixel projection.
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
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .e035_loss_fidelity import (
    _atomic_json,
    _atomic_text,
    _decode_latent_tensor,
    _image_sha256,
    _offload_diffusion_modules,
    _score_qr_verify,
)
from .e035_parent_artifact import LoadedParentArtifact, sha256_file, tensor_sha256
from .e035_parent_capture import CaptureConfig, UPSTREAM_REVISION, _settings_document
from .e038_recipe_frontier import _qr_original_exact, _score_quality, _visual_guard
from .e039_limiter_scanaware import E039Config
from .e040_checkpoint_frontier import Recipe as E040Recipe, _run_trajectory
from .e040_model_bridge import score_surrogate_images
from .e043_scanner_cell_frontier import _decoded_to_exact_scan_ready, _scanner_diagnostics

EXPERIMENT = "e044-multi-prompt-best-pipeline-v1"
PAYLOAD = "https://ptag.io/t/e044"
ERROR_CORRECTION = "M"
QR_VERSION = 3
QR_MASK_PATTERN = 4
QR_MODULE_SIZE = 20
QR_PADDING_PX = 78
SEED = 72044
GAMMAS = (500.0, 1000.0)
LATENT_RADIUS_RMS = 0.200
MAX_ITERATIONS = 8
EXPECTED_CHECKPOINTS_PER_PROMPT = len(GAMMAS) * (MAX_ITERATIONS + 1)
NEGATIVE_PROMPT = (
    "easynegative, low quality, worst quality, blurry, deformed, watermark, text, "
    "letters, logo, signature, oversaturated, clipped highlights, posterized colors"
)

PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "p01_greenhouse", "family": "organic_grid", "text": "a sunlit greenhouse filled with tomato plants, narrow wooden paths and rows of terracotta pots, balanced botanical photograph, natural geometry, soft daylight"},
    {"id": "p02_blue_vase", "family": "minimal_object", "text": "a single cobalt blue ceramic vase holding one yellow tulip, centered on a warm cream background, soft window light, clean still-life photograph"},
    {"id": "p03_lighthouse", "family": "high_contrast_vertical", "text": "a solitary white stone lighthouse beside a calm blue sea at sunrise, centered composition, clean vintage travel poster without lettering"},
    {"id": "p04_brutalist_grid", "family": "architectural_grid", "text": "a monumental brutalist concrete library facade with repeated square windows, frontal symmetrical architectural photograph, overcast sky, precise geometric rhythm"},
    {"id": "p05_sleeping_cat", "family": "soft_illustration", "text": "one orange cat curled asleep on a round moss-green cushion, plain light background, gentle children's-book illustration, simple composition, soft shapes"},
    {"id": "p06_mycelium", "family": "high_frequency_difficult", "text": "a transparent glass cube containing a living bioluminescent mycelium circuit, cyan branching veins and tiny amber spores, dark laboratory, surreal macro photography"},
    {"id": "p07_winter_cabin", "family": "official_demo_like", "text": "winter wonderland with fresh snowfall, evergreen trees, a cozy log cabin with smoke rising from the chimney, aurora borealis in the night sky, cinematic landscape"},
)


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


def _prompt(prompt_id: str) -> dict[str, str]:
    item = next((row for row in PROMPTS if row["id"] == prompt_id), None)
    if item is None:
        raise ValueError(f"unknown E044 prompt id: {prompt_id}")
    return item


def _capture_config(prompt: str) -> CaptureConfig:
    return CaptureConfig(
        payload=PAYLOAD,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        error_correction=ERROR_CORRECTION,
        seed=SEED,
        steps=40,
        guidance_scale=7.5,
        controlnet_scale=1.35,
        strength=1.0,
        qr_version=QR_VERSION,
        qr_mask_pattern=QR_MASK_PATTERN,
        qr_module_size=QR_MODULE_SIZE,
        qr_padding_px=QR_PADDING_PX,
        srpg_steps=40,
        srpg_controlnet_scale=1.05,
        srpg_qr_weight=50.0,
        srpg_perceptual_weight=20.0,
        srpg_eta=0.0,
        stage2_initialization="public_random",
        stage2_strength=1.0,
        stage2_target_mode="binary_exact",
    )


def _fresh_parent(*, backend: Any, blueprint: Any, prompt_item: dict[str, str], output_dir: Path, source_commit: str) -> LoadedParentArtifact:
    import torch
    from safetensors.torch import save_file
    from .schemas import GenerationRequest

    root = output_dir / "parent"
    root.mkdir(parents=True, exist_ok=True)
    config = _capture_config(prompt_item["text"])
    request = GenerationRequest(
        payload=PAYLOAD,
        prompt=prompt_item["text"],
        negative_prompt=NEGATIVE_PROMPT,
        backend="controlnet",
        error_correction=ERROR_CORRECTION,
        seed=SEED,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        controlnet_scale=config.controlnet_scale,
        strength=config.strength,
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
        raise RuntimeError("E044 Stage 2 produced no exportable latent")
    latent = state["latent"].detach().cpu().contiguous()
    if stage1.size != (736, 736) or stage2.size != (736, 736):
        raise RuntimeError(f"E044 expects 736x736 Stage1/2, got {stage1.size} and {stage2.size}")

    stage1_path = root / "stage1.png"
    stage2_path = root / "stage2.png"
    latent_path = root / "stage2-latent.safetensors"
    _save_png(stage1_path, stage1)
    _save_png(stage2_path, stage2)
    save_file({"latent": latent}, str(latent_path))
    metadata = {
        "experiment": EXPERIMENT,
        "prompt_id": prompt_item["id"],
        "prompt_family": prompt_item["family"],
        "payload": PAYLOAD,
        "prompt": prompt_item["text"],
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": SEED,
        "source_commit": source_commit,
        "diffqrcoder_revision": UPSTREAM_REVISION,
        "stage1_image_sha256": _image_sha256(stage1),
        "stage2_image_sha256": _image_sha256(stage2),
        "stage2_latent_tensor_sha256": tensor_sha256(latent),
        "stage2_latent_file_sha256": sha256_file(latent_path),
        "elapsed_s": time.perf_counter() - started,
        "stage2_recipe": asdict(config),
        "note": "E044 fresh paired prompt parent; Prooftag learned baseline, not claimed paper-exact.",
    }
    _atomic_json(root / "parent-metadata.json", metadata)
    return LoadedParentArtifact(root=root, image=stage2.convert("RGB"), latent=latent, metadata={"source": metadata})


def _gamma_recipe(prompt_id: str, gamma: float) -> E040Recipe:
    return E040Recipe(
        name=f"e044_{prompt_id}_gamma{int(gamma):04d}_r200_i08",
        latent_radius_rms=LATENT_RADIUS_RMS,
        max_iterations=MAX_ITERATIONS,
    )


def _redecode_exact_checkpoints(pipeline: Any, checkpoints: list[Any]) -> None:
    import torch
    from safetensors.torch import load_file
    for checkpoint in checkpoints:
        latent = load_file(checkpoint.latent_path, device="cpu")["latent"].to(device="cuda", dtype=torch.float32)
        with torch.no_grad():
            decoded = _decode_latent_tensor(pipeline, latent).float()
        exact = _decoded_to_exact_scan_ready(decoded)
        _save_png(Path(checkpoint.image_path), exact)
        del latent, decoded
        gc.collect()
        torch.cuda.empty_cache()


def _decoder_diagnostics(images: dict[str, Image.Image]) -> dict[str, dict[str, Any]]:
    from .validation import OpenCVDecoder, PyzbarDecoder, ZXingCPPDecoder, WeChatQRCodeDecoder
    decoder_types = (("opencv", OpenCVDecoder), ("zbar", PyzbarDecoder), ("zxingcpp", ZXingCPPDecoder), ("wechat_qrcode", WeChatQRCodeDecoder))
    output: dict[str, dict[str, Any]] = {key: {} for key in images}
    for decoder_name, decoder_type in decoder_types:
        try:
            decoder = decoder_type()
        except Exception as exc:
            for key in images:
                output[key][decoder_name] = {"available": False, "exact": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        for key, image in images.items():
            try:
                text = decoder.decode(image)
                output[key][decoder_name] = {"available": True, "exact": text == PAYLOAD, "decoded_text": text[:200]}
            except Exception as exc:
                output[key][decoder_name] = {"available": True, "exact": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    return output


def _score_prompt(*, output_dir: Path, prompt_item: dict[str, str], backend: Any, blueprint: Any, checkpoints: list[Any], parent_exact: Image.Image) -> list[dict[str, Any]]:
    from .quality import image_change_metrics, image_quality_metrics
    from .qr import diffqrcoder_module_error_rate, diffqrcoder_structure_metrics

    images: dict[str, Image.Image] = {"stage2_parent": parent_exact}
    metadata: dict[str, dict[str, Any]] = {
        "stage2_parent": {"kind": "stage2_parent", "gamma": 0.0, "iteration": 0, "image_path": str(output_dir / "parent/stage2-exact-qz.png"), "latent_path": str(output_dir / "parent/stage2-latent.safetensors"), "is_parent_reference": True, "lpips_trace": 0.0}
    }
    for checkpoint in checkpoints:
        gamma = float(checkpoint.trace_step["gamma"])
        key = f"g{int(gamma):04d}_i{int(checkpoint.iteration):02d}"
        images[key] = Image.open(checkpoint.image_path).convert("RGB")
        metadata[key] = {
            "kind": "srmpgd_checkpoint", "gamma": gamma, "iteration": int(checkpoint.iteration),
            "image_path": checkpoint.image_path, "latent_path": checkpoint.latent_path,
            "is_parent_reference": False, "lpips_trace": float(checkpoint.trace_step.get("lpips_loss") or 0.0),
            "latent_delta_rms": checkpoint.trace_step.get("latent_delta_rms"), "accepted_alpha": checkpoint.trace_step.get("accepted_alpha"),
            "accepted_step_rms": checkpoint.trace_step.get("accepted_step_rms"), "raw_step_rms": checkpoint.trace_step.get("raw_step_rms"),
            "projected_step_rms": checkpoint.trace_step.get("projected_step_rms"), "rejected_trial_count": checkpoint.trace_step.get("rejected_trial_count"),
            "acceptance_reason": checkpoint.trace_step.get("acceptance_reason"),
            "projection_was_active": bool(checkpoint.trace_step.get("raw_step_rms") is not None and checkpoint.trace_step.get("projected_step_rms") is not None and float(checkpoint.trace_step["raw_step_rms"]) > float(checkpoint.trace_step["projected_step_rms"]) + 1e-9),
        }

    scoring_root = output_dir / "scoring"
    scoring_root.mkdir(parents=True, exist_ok=True)
    qr_verify = _score_qr_verify(scoring_root, PAYLOAD, images)
    quality_scores, quality_provenance = _score_quality(images, prompt_item["text"], backend.settings)
    _atomic_json(scoring_root / "quality-scores.json", quality_scores)
    _atomic_json(scoring_root / "quality-provenance.json", quality_provenance)
    surrogate_scores, surrogate_status = score_surrogate_images(images)
    _atomic_json(scoring_root / "e016-surrogate-scores.json", surrogate_scores)
    _atomic_json(scoring_root / "e016-surrogate-status.json", surrogate_status)
    decoder_diag = _decoder_diagnostics(images)
    _atomic_json(scoring_root / "decoder-diagnostics.json", decoder_diag)

    parent_quality = quality_scores.get("stage2_parent") or {"clip_score": 0.0, "clip_aesthetic": 0.0, "hpsv2_1": None}
    guard_config = E039Config()
    rows: list[dict[str, Any]] = []
    for key, image in images.items():
        info = dict(metadata[key])
        qitem = (qr_verify or {}).get(key) or {}
        qscore = quality_scores.get(key) or {}
        surrogate = surrogate_scores.get(key) or {}
        change = image_change_metrics(image, parent_exact)
        quality = image_quality_metrics(image)
        structure = diffqrcoder_structure_metrics(image, blueprint, padding_px=QR_PADDING_PX, module_size=QR_MODULE_SIZE)
        module_rate = diffqrcoder_module_error_rate(image, blueprint, padding_px=QR_PADDING_PX, module_size=QR_MODULE_SIZE)
        row = {
            "prompt_id": prompt_item["id"], "prompt_family": prompt_item["family"], "prompt": prompt_item["text"], "variant": key,
            **info,
            "qr_verify_exact_presets": int(qitem.get("conservative_exact_presets", 0)),
            "ssr": int(qitem.get("conservative_exact_presets", 0)) / 37.0,
            "original_exact": _qr_original_exact(qitem),
            "full_module_error_count": int(round(module_rate * 841)),
            "lpips": float(info.get("lpips_trace") or 0.0),
            "clip_score": qscore.get("clip_score"), "clip_aesthetic": qscore.get("clip_aesthetic"), "hpsv2_1": qscore.get("hpsv2_1"),
            "surrogate_mean_success_probability": surrogate.get("mean_success_probability"),
            "surrogate_min_success_probability": surrogate.get("min_success_probability"),
            "decoder_diagnostics": decoder_diag.get(key) or {},
            **structure, **change, **quality, **_scanner_diagnostics(image, blueprint),
        }
        guard = _visual_guard(row, parent_quality, guard_config)
        row["visual_guard_pass"] = bool(guard["passed"])
        row["visual_guard_checks"] = guard["checks"]
        rows.append(row)

    _atomic_json(scoring_root / "comparison.json", rows)
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["visual_guard_checks"] = json.dumps(flat.get("visual_guard_checks") or {}, ensure_ascii=False, sort_keys=True)
        flat["decoder_diagnostics"] = json.dumps(flat.get("decoder_diagnostics") or {}, ensure_ascii=False, sort_keys=True)
        csv_rows.append(flat)
    _write_csv(scoring_root / "comparison.csv", csv_rows)
    return rows


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (-int(row.get("qr_verify_exact_presets", 0)), -int(bool(row.get("original_exact"))), int(row.get("full_module_error_count", 10**9)), float(row.get("lpips", 1e9)), -float(row.get("clip_aesthetic") or -1e9), int(row.get("iteration", 10**9)), abs(float(row.get("gamma", 0.0)) - 500.0))


def run_prompt(*, root: Path, prompt_id: str, source_commit: str) -> dict[str, Any]:
    import torch
    from .config import Settings
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .qr import generate_diffqrcoder_qr

    if not torch.cuda.is_available():
        raise RuntimeError("E044 prompt job requires CUDA")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    prompt_item = _prompt(prompt_id)
    final_dir = root / "prompts" / prompt_id
    if (final_dir / "COMPLETE.json").is_file():
        return json.loads((final_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    if final_dir.exists():
        raise FileExistsError(f"E044 final prompt dir exists without COMPLETE marker; preserve and inspect: {final_dir}")

    attempt = root / "attempts" / f"{prompt_id}-{source_commit[:12]}-{uuid.uuid4().hex[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    _atomic_json(attempt / "plan.json", {
        "experiment": EXPERIMENT, "prompt_id": prompt_id, "prompt_family": prompt_item["family"], "prompt": prompt_item["text"],
        "negative_prompt": NEGATIVE_PROMPT, "payload": PAYLOAD, "seed": SEED, "source_commit": source_commit,
        "gamma_grid": list(GAMMAS), "latent_radius_rms": LATENT_RADIUS_RMS, "max_iterations": MAX_ITERATIONS,
        "stage2_recipe": asdict(_capture_config(prompt_item["text"])), "quiet_zone_geometry": "exact 736/78/580",
        "forbidden_additions": ["functional-pattern toning", "E043 scanner-cell losses", "pixel QR projection"],
        "production_ready": False, "generalization_authorized": False,
    })

    config = _capture_config(prompt_item["text"])
    base = Settings()
    settings = Settings.model_validate({**base.model_dump(), **_settings_document(config)})
    if str(settings.diffqrcoder_revision) != UPSTREAM_REVISION:
        raise RuntimeError("runtime DiffQRCoder revision differs from E044 registration")
    backend = UpstreamDiffQRCoderBackend(settings)
    pipeline = backend._load()
    blueprint = generate_diffqrcoder_qr(PAYLOAD, ERROR_CORRECTION, version=QR_VERSION, mask_pattern=QR_MASK_PATTERN, module_size=QR_MODULE_SIZE)
    parent = _fresh_parent(backend=backend, blueprint=blueprint, prompt_item=prompt_item, output_dir=attempt, source_commit=source_commit)

    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
    all_checkpoints: list[Any] = []
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            if not checkpointing_was_enabled and callable(enable_checkpointing):
                enable_checkpointing()
            pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
            _atomic_json(attempt / "runtime.json", {"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "device_name": torch.cuda.get_device_name(0), "offloaded_modules": list(offloaded), "vae_original_dtype": str(original_vae_dtype), "gamma_grid": list(GAMMAS)})
            with torch.no_grad():
                parent_latent = parent.latent.to(device="cuda", dtype=torch.float32)
                parent_decoded = _decode_latent_tensor(pipeline, parent_latent).float()
            parent_exact = _decoded_to_exact_scan_ready(parent_decoded)
            _save_png(attempt / "parent/stage2-exact-qz.png", parent_exact)
            del parent_latent, parent_decoded
            for gamma in GAMMAS:
                checkpoints = _run_trajectory(pipeline=pipeline, parent=parent, blueprint=blueprint, recipe=_gamma_recipe(prompt_id, gamma), config=E039Config(gamma=gamma), output_root=attempt / "trajectories")
                _redecode_exact_checkpoints(pipeline, checkpoints)
                all_checkpoints.extend(checkpoints)
                gc.collect(); torch.cuda.empty_cache()
    finally:
        pipeline.vae.to(dtype=original_vae_dtype)
        if not checkpointing_was_enabled and callable(disable_checkpointing):
            disable_checkpointing()
        gc.collect(); torch.cuda.empty_cache()

    if len(all_checkpoints) != EXPECTED_CHECKPOINTS_PER_PROMPT:
        raise RuntimeError(f"E044 {prompt_id} produced {len(all_checkpoints)} checkpoints; expected {EXPECTED_CHECKPOINTS_PER_PROMPT}")
    backend._pipeline = None
    del pipeline
    gc.collect(); torch.cuda.empty_cache()

    parent_exact = Image.open(attempt / "parent/stage2-exact-qz.png").convert("RGB")
    rows = _score_prompt(output_dir=attempt, prompt_item=prompt_item, backend=backend, blueprint=blueprint, checkpoints=all_checkpoints, parent_exact=parent_exact)
    safe = [row for row in rows if bool(row.get("visual_guard_pass"))]
    if not safe:
        raise RuntimeError(f"E044 {prompt_id}: no visually-safe row, including parent")
    winner = sorted(safe, key=_rank_key)[0]
    raw_best = sorted(rows, key=_rank_key)[0]
    best_per_gamma = []
    for gamma in GAMMAS:
        options = [row for row in rows if math.isclose(float(row.get("gamma", 0.0)), gamma) and bool(row.get("visual_guard_pass"))]
        if options:
            best_per_gamma.append(sorted(options, key=_rank_key)[0])
    _atomic_json(attempt / "best-per-gamma.json", best_per_gamma)

    pipeline_dir = attempt / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(attempt / "parent/stage1.png", pipeline_dir / "01-stage1.png")
    shutil.copy2(attempt / "parent/stage2.png", pipeline_dir / "02-stage2.png")
    shutil.copy2(attempt / "parent/stage2-exact-qz.png", pipeline_dir / "03-stage2-exact-qz.png")
    shutil.copy2(Path(str(winner["image_path"])), pipeline_dir / "99-FINAL-QR.png")
    shutil.copy2(Path(str(winner["latent_path"])), pipeline_dir / "99-FINAL-latent.safetensors")
    _atomic_json(pipeline_dir / "99-FINAL-metadata.json", {"prompt_id": prompt_id, "winner_variant": winner["variant"], "gamma": winner["gamma"], "iteration": winner["iteration"], "ssr_exact_presets": winner["qr_verify_exact_presets"], "original_exact": winner["original_exact"], "visual_guard_pass": winner["visual_guard_pass"]})

    verdict = {
        "experiment": EXPERIMENT, "prompt_id": prompt_id, "prompt_family": prompt_item["family"], "prompt": prompt_item["text"], "seed": SEED,
        "source_commit": source_commit, "checkpoint_count": len(all_checkpoints), "scored_image_count": len(rows), "safe_image_count": len(safe),
        "gamma_grid": list(GAMMAS), "winner_variant": winner["variant"], "winner_gamma": float(winner["gamma"]), "winner_iteration": int(winner["iteration"]),
        "winner_ssr_exact_presets": int(winner["qr_verify_exact_presets"]), "winner_ssr": float(winner["ssr"]), "winner_original_exact": bool(winner["original_exact"]),
        "winner_lpips": float(winner["lpips"]), "winner_clip_score": winner.get("clip_score"), "winner_clip_aesthetic": winner.get("clip_aesthetic"), "winner_hpsv2_1": winner.get("hpsv2_1"),
        "winner_full_module_error_count": int(winner["full_module_error_count"]), "winner_visual_guard_pass": bool(winner["visual_guard_pass"]),
        "raw_best_variant": raw_best["variant"], "raw_best_ssr_exact_presets": int(raw_best["qr_verify_exact_presets"]),
        "exact_quiet_zone_geometry": True, "production_ready": False, "generalization_authorized": False,
    }
    _atomic_json(attempt / "verdict.json", verdict)
    _atomic_text(attempt / "report.md", "\n".join([f"# E044 — {prompt_id}", "", f"- family: `{prompt_item['family']}`", f"- prompt: `{prompt_item['text']}`", f"- seed: `{SEED}`", f"- gammas: `{GAMMAS}`", f"- winner: `{winner['variant']}`", f"- SSR: **{winner['qr_verify_exact_presets']}/37**", f"- original exact: **{winner['original_exact']}**", f"- visual guard: **{winner['visual_guard_pass']}**", "", "This is a prompt-screen result only. Production/generalization remain false."]))
    _atomic_json(attempt / "COMPLETE.json", verdict)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, final_dir)
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    verdict = run_prompt(root=args.root, prompt_id=args.prompt_id, source_commit=args.source_commit)
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
