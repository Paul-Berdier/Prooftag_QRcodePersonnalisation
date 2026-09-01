"""E040 — checkpoint-aware SR-MPGD frontier and full QR pipeline evidence.

E040 keeps gamma=1000 and the E039 scan-aware-v2 objective. It varies only the
latent trust-region radius around the useful E039 zone and evaluates *every*
checkpoint i0..i8 with real QR-Verify + visual guards. The best visually safe
checkpoint wins; the last iteration is no longer privileged.

E016's trained differentiable scan surrogate is used only as a research score
(and tie-break after real QR-Verify), never as a replacement for real decoders.
The E026/E031 advisor is recorded in pipeline metadata as a parameter
recommendation, while the frozen E035 parent remains authoritative for this
single-parent optimization experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .e035_loss_fidelity import (
    _atomic_json,
    _atomic_text,
    _core_blueprint,
    _crop_core,
    _cuda_snapshot,
    _decode_latent_tensor,
    _decoded_to_scan_ready_image,
    _gradient_scales,
    _image_sha256,
    _load_lpips,
    _load_official_upstream_srl,
    _load_pipeline,
    _offload_diffusion_modules,
    _score_qr_verify,
)
from .e035_losses import module_diagnostics, prepare_upstream_torch_layout, upstream_qrcode_tensor
from .e035_parent_artifact import (
    E034_OBSERVED_STAGE1_FILE_SHA256,
    E034_OBSERVED_STAGE1_IMAGE_SHA256,
    LoadedParentArtifact,
    load_parent_artifact,
    sha256_file,
    tensor_sha256,
)
from .e036_trust_region import project_latent_candidate
from .e038_recipe_frontier import (
    _comparison_sheet,
    _qr_original_exact,
    _score_quality,
    _visual_guard,
)
from .e039_limiter_scanaware import (
    E039Config,
    Recipe as E039Recipe,
    _candidate_metrics,
    _qr_objective,
    _rms,
)
from .e040_model_bridge import advisor_preview, score_surrogate_images

EXPERIMENT = "e040-srmpgd-checkpoint-frontier-v1"
E039_REQUIRED_EXPERIMENT = "e039-srmpgd-limiter-scanaware-v1"
E039_REQUIRED_WINNER = "e039_scanaware_r200_i08"
DEFAULT_RADII = (0.150, 0.175, 0.200, 0.225, 0.250)


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    latent_radius_rms: float
    max_iterations: int = 8
    lpips_budget: float = 0.050
    core_mae_budget: float = 0.050
    full_module_weight: float = 0.10

    def e039(self) -> E039Recipe:
        return E039Recipe(
            name=self.name,
            profile="scanaware_v2",
            max_iterations=self.max_iterations,
            latent_radius_rms=self.latent_radius_rms,
            lpips_budget=self.lpips_budget,
            core_mae_budget=self.core_mae_budget,
            full_module_weight=self.full_module_weight,
        )


DEFAULT_RECIPES: tuple[Recipe, ...] = tuple(
    Recipe(f"e040_scanaware_r{int(round(radius * 1000)):03d}_i08", radius)
    for radius in DEFAULT_RADII
)


@dataclass(frozen=True, slots=True)
class CheckpointResult:
    recipe: str
    iteration: int
    image_path: str
    latent_path: str
    trace_step: dict[str, Any]


def recipe_catalog() -> list[dict[str, Any]]:
    return [asdict(recipe) for recipe in DEFAULT_RECIPES]


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


def _load_e039_control(path: Path) -> dict[str, Any]:
    verdict_path = path / "verdict.json"
    comparison_path = path / "method-comparison.json"
    if not verdict_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError("E039 verdict/comparison missing")
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict.get("experiment") != E039_REQUIRED_EXPERIMENT:
        raise RuntimeError(f"unexpected E039 experiment: {verdict.get('experiment')}")
    if verdict.get("gamma") != 1000.0 or verdict.get("gamma_preserved") is not True:
        raise RuntimeError("E040 requires E039 gamma=1000 preserved")
    if verdict.get("research_winner") != E039_REQUIRED_WINNER:
        raise RuntimeError(f"E040 requires E039 winner {E039_REQUIRED_WINNER}")
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))
    row = next((item for item in rows if item.get("method") == E039_REQUIRED_WINNER), None)
    if row is None:
        raise RuntimeError("E039 winner row missing")
    return {"verdict": verdict, "winner": row}


def _run_trajectory(
    *,
    pipeline: Any,
    parent: LoadedParentArtifact,
    blueprint: Any,
    recipe: Recipe,
    config: E039Config,
    output_root: Path,
) -> list[CheckpointResult]:
    """E039 scan-aware-v2 with every latent persisted as a first-class checkpoint."""
    import torch
    from safetensors.torch import save_file

    base_recipe = recipe.e039()
    root = output_root / recipe.name
    images_root = root / "images"
    latents_root = root / "latents"
    images_root.mkdir(parents=True, exist_ok=True)
    latents_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    initial = parent.latent.detach().to(device=device, dtype=torch.float32).clone()
    working = initial.clone()
    with torch.no_grad():
        reference_decoded = _decode_latent_tensor(pipeline, initial).float().detach()
    reference_core = _crop_core(reference_decoded, config.crop_padding_px).detach()
    core_height, core_width = reference_core.shape[-2:]
    core_blueprint = _core_blueprint(blueprint, core_width, core_height)

    from .guidance import prepare_torch_layout

    paper_layout = prepare_torch_layout(
        core_blueprint, core_height, core_width, device=device, dtype=torch.float32, center_fraction=1 / 3
    )
    upstream_layout = prepare_upstream_torch_layout(
        core_blueprint, core_height, core_width, device=device, dtype=torch.float32
    )
    upstream_target = upstream_qrcode_tensor(
        core_blueprint, core_height, core_width, device=device, dtype=torch.float32
    )
    official_upstream_srl = _load_official_upstream_srl(config.qr_module_size, device=device)
    lpips_model = _load_lpips(pipeline, net=config.lpips_net)
    lpips_parameter = next(iter(lpips_model.parameters()), None)
    lpips_dtype = lpips_parameter.dtype if lpips_parameter is not None else torch.float32
    reference_lpips = reference_core.to(device="cpu", dtype=lpips_dtype).detach()

    rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    checkpoints: list[CheckpointResult] = []
    started = time.perf_counter()

    for iteration in range(recipe.max_iterations + 1):
        working = working.detach()
        with torch.no_grad():
            decoded = reference_decoded if iteration == 0 else _decode_latent_tensor(pipeline, working).float()
            decoded_core = _crop_core(decoded, config.crop_padding_px).detach()
            decoded_unit = (decoded_core / 2 + 0.5).clamp(0, 1)
            qr_tensor, parts = _qr_objective(
                decoded_unit,
                recipe=base_recipe,
                core_blueprint=core_blueprint,
                upstream_layout=upstream_layout,
                upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl,
                config=config,
                iteration=iteration,
                phase="evaluation",
            )
            lpips_loss = float(
                lpips_model(decoded_core.to(device="cpu", dtype=lpips_dtype), reference_lpips)
                .mean().detach().cpu()
            )
            qr_objective_value = float(qr_tensor.detach().cpu())
            objective = qr_objective_value + config.lpips_weight * lpips_loss
            diagnostics = module_diagnostics(
                decoded_unit, core_blueprint, paper_layout=paper_layout, upstream_layout=upstream_layout
            )

        image = _decoded_to_scan_ready_image(pipeline, decoded, blueprint, config)
        image_path = images_root / f"iteration-{iteration:03d}.png"
        latent_path = latents_root / f"iteration-{iteration:03d}.safetensors"
        image.save(image_path, format="PNG", optimize=False, compress_level=9)
        save_file({"latent": working.detach().cpu().contiguous()}, str(latent_path))

        latent_gradient_rms = raw_step_rms = projected_step_rms = None
        accepted_step_rms = accepted_alpha = effective_gradient_scale = None
        acceptance_reason: str | None = None
        candidate_checks: dict[str, bool] | None = None
        next_working = working
        rejected_this_iteration = 0

        if iteration < recipe.max_iterations and qr_objective_value > 0.0:
            objective_core = decoded_core.detach().requires_grad_(True)
            objective_unit = (objective_core / 2 + 0.5).clamp(0, 1)
            qr_loss, _ = _qr_objective(
                objective_unit,
                recipe=base_recipe,
                core_blueprint=core_blueprint,
                upstream_layout=upstream_layout,
                upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl,
                config=config,
                iteration=iteration,
                phase="gradient",
            )
            qr_gradient = torch.autograd.grad(qr_loss, objective_core, only_inputs=True)[0]
            lpips_core = decoded_core.detach().to(device="cpu", dtype=lpips_dtype).requires_grad_(True)
            lpips_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(lpips_tensor, lpips_core, only_inputs=True)[0]
            objective_gradient = qr_gradient.clone()
            objective_gradient.add_(
                lpips_gradient_cpu.to(device=device, dtype=objective_gradient.dtype),
                alpha=config.lpips_weight,
            )

            gradient = None
            if torch.isfinite(objective_gradient).all():
                for scale in _gradient_scales(config.gradient_scale):
                    candidate = working.detach().requires_grad_(True)
                    candidate_decoded = _decode_latent_tensor(pipeline, candidate).float()
                    candidate_core = _crop_core(candidate_decoded, config.crop_padding_px)
                    candidate_gradient = torch.autograd.grad(
                        candidate_core,
                        candidate,
                        grad_outputs=objective_gradient.to(dtype=candidate_core.dtype) * scale,
                        only_inputs=True,
                    )[0] / scale
                    del candidate_core, candidate_decoded, candidate
                    if torch.isfinite(candidate_gradient).all():
                        gradient = candidate_gradient
                        effective_gradient_scale = float(scale)
                        break
            del objective_core, objective_unit, qr_loss, qr_gradient
            del lpips_core, lpips_tensor, lpips_gradient_cpu, objective_gradient

            if gradient is None:
                acceptance_reason = "no_finite_latent_gradient"
            else:
                latent_gradient_rms = float(_rms(gradient).detach().cpu())
                raw_target = working - config.gamma * gradient
                raw_step_rms = float(_rms(raw_target - working).detach().cpu())
                projected_target = project_latent_candidate(raw_target, initial, recipe.latent_radius_rms)
                projected_step_rms = float(_rms(projected_target - working).detach().cpu())
                direction = projected_target - working
                alpha = 1.0
                last_metrics = None
                for backtrack_index in range(config.max_backtracks + 1):
                    if alpha < config.minimum_alpha:
                        break
                    trial = (working + alpha * direction).detach()
                    metrics = _candidate_metrics(
                        pipeline=pipeline,
                        candidate=trial,
                        initial=initial,
                        reference_core=reference_core,
                        reference_lpips=reference_lpips,
                        lpips_model=lpips_model,
                        lpips_dtype=lpips_dtype,
                        core_blueprint=core_blueprint,
                        upstream_layout=upstream_layout,
                        upstream_target=upstream_target,
                        official_upstream_srl=official_upstream_srl,
                        recipe=base_recipe,
                        current_objective=objective,
                        config=config,
                    )
                    last_metrics = metrics
                    failed = [key for key, passed in metrics.checks.items() if not passed]
                    rejection_rows.append(
                        {
                            "recipe": recipe.name,
                            "iteration": iteration,
                            "backtrack_index": backtrack_index,
                            "alpha": alpha,
                            "accepted": metrics.accepted,
                            "rejection_reasons": ";".join(failed),
                            "objective": metrics.objective,
                            "lpips": metrics.lpips,
                            "core_mae": metrics.core_mae,
                            "latent_delta_rms": metrics.latent_delta_rms,
                            **{f"check_{key}": passed for key, passed in metrics.checks.items()},
                        }
                    )
                    if metrics.accepted:
                        next_working = trial
                        accepted_alpha = alpha
                        accepted_step_rms = float(_rms(next_working - working).detach().cpu())
                        acceptance_reason = "accepted"
                        candidate_checks = metrics.checks
                        break
                    rejected_this_iteration += 1
                    alpha *= 0.5
                if acceptance_reason != "accepted":
                    acceptance_reason = "trust_region_rejected_all_candidates"
                    accepted_alpha = 0.0
                    accepted_step_rms = 0.0
                    candidate_checks = last_metrics.checks if last_metrics else None
                del gradient, raw_target, projected_target, direction
        elif iteration < recipe.max_iterations:
            acceptance_reason = "objective_zero_hold_state"
            accepted_alpha = 0.0
            accepted_step_rms = 0.0

        latent_delta_rms = float(_rms(working - initial).detach().cpu())
        step = {
            "recipe": recipe.name,
            "iteration": iteration,
            "elapsed_s": time.perf_counter() - started,
            "image_sha256": _image_sha256(image),
            "latent_sha256": tensor_sha256(working),
            "upstream_srl": parts["upstream_srl"],
            "full_module_loss": parts["full_module_loss"],
            "robust_loss": parts["robust_loss"],
            "lpips_loss": lpips_loss,
            "objective": objective,
            "upstream_active_modules": int(diagnostics["upstream_margin_active_count"]),
            "full_module_error_count": int(diagnostics["full_module_error_count"]),
            "full_module_error_rate": float(diagnostics["full_module_error_rate"]),
            "gamma": config.gamma,
            "latent_gradient_rms": latent_gradient_rms,
            "raw_step_rms": raw_step_rms,
            "projected_step_rms": projected_step_rms,
            "accepted_step_rms": accepted_step_rms,
            "accepted_alpha": accepted_alpha,
            "latent_delta_rms": latent_delta_rms,
            "effective_gradient_scale": effective_gradient_scale,
            "acceptance_reason": acceptance_reason,
            "candidate_checks": candidate_checks,
            "rejected_trial_count": rejected_this_iteration,
            "cuda": _cuda_snapshot(),
        }
        rows.append(step)
        checkpoints.append(
            CheckpointResult(
                recipe=recipe.name,
                iteration=iteration,
                image_path=str(image_path),
                latent_path=str(latent_path),
                trace_step=step,
            )
        )
        working = next_working.detach()
        gc.collect()
        torch.cuda.empty_cache()

    _atomic_json(root / "trace.json", rows)
    _atomic_json(root / "rejection-log.json", rejection_rows)
    _write_csv(root / "trace.csv", [
        {
            **{k: v for k, v in row.items() if k not in {"candidate_checks", "cuda"}},
            **{f"check_{k}": v for k, v in (row.get("candidate_checks") or {}).items()},
            **{f"cuda_{k}": v for k, v in (row.get("cuda") or {}).items()},
        }
        for row in rows
    ])
    _write_csv(root / "rejection-log.csv", rejection_rows)
    _atomic_json(root / "recipe.json", asdict(recipe))
    return checkpoints


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "e040-artifact-manifest.json":
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def run_e040(
    *,
    parent_dir: Path,
    e039_results_dir: Path,
    output_dir: Path,
    expected_parent_commit: str | None = None,
    recipes: tuple[Recipe, ...] = DEFAULT_RECIPES,
    config: E039Config | None = None,
    skip_quality: bool = False,
    skip_qr_verify: bool = False,
) -> dict[str, Any]:
    import torch
    from .quality import image_change_metrics, image_quality_metrics
    from .qr import generate_diffqrcoder_qr

    config = config or E039Config()
    if config.gamma != 1000.0:
        raise ValueError("E040 keeps gamma fixed at 1000")
    if not torch.cuda.is_available():
        raise RuntimeError("E040 requires CUDA")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E040 output must be empty: {output_dir}")

    control = _load_e039_control(e039_results_dir)
    expected = {
        "qr_version": config.qr_version,
        "qr_mask_pattern": config.qr_mask_pattern,
        "qr_module_size": config.qr_module_size,
        "qr_padding_px": config.crop_padding_px,
        "stage1_image_sha256": E034_OBSERVED_STAGE1_IMAGE_SHA256,
        "stage1_file_sha256": E034_OBSERVED_STAGE1_FILE_SHA256,
    }
    if expected_parent_commit:
        expected["source_commit"] = expected_parent_commit
    parent = load_parent_artifact(parent_dir, device="cpu", expected=expected)
    source = parent.metadata["source"]
    payload = str(source["payload"])
    prompt = str(source.get("prompt") or "")
    blueprint = generate_diffqrcoder_qr(
        payload,
        str(source["error_correction"]),
        version=config.qr_version,
        mask_pattern=config.qr_mask_pattern,
        module_size=config.qr_module_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "plan.json", {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "gamma": config.gamma,
        "gamma_is_fixed": True,
        "recipes": [asdict(recipe) for recipe in recipes],
        "selection": "best visually-safe checkpoint, not necessarily final iteration",
        "e016_role": "research score / tie-break only after real QR-Verify",
        "advisor_role": "prospective parameter recommendation only",
        "production_ready": False,
    })
    _atomic_json(output_dir / "e039-control.json", control)

    backend, pipeline = _load_pipeline()
    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
    all_checkpoints: list[CheckpointResult] = []
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            try:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                _atomic_json(output_dir / "runtime.json", {
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(0),
                    "offloaded_modules": list(offloaded),
                    "vae_original_dtype": str(original_vae_dtype),
                    "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                })
                for recipe in recipes:
                    all_checkpoints.extend(_run_trajectory(
                        pipeline=pipeline,
                        parent=parent,
                        blueprint=blueprint,
                        recipe=recipe,
                        config=config,
                        output_root=output_dir,
                    ))
                    gc.collect(); torch.cuda.empty_cache()
            finally:
                pipeline.vae.to(dtype=original_vae_dtype)
                if not checkpointing_was_enabled and callable(disable_checkpointing):
                    disable_checkpointing()
                gc.collect(); torch.cuda.empty_cache()
    finally:
        if next(pipeline.vae.parameters()).dtype != original_vae_dtype:
            pipeline.vae.to(dtype=original_vae_dtype)
        del pipeline
        gc.collect(); torch.cuda.empty_cache()

    checkpoint_images = {
        f"{item.recipe}__i{item.iteration:02d}": Image.open(item.image_path).convert("RGB")
        for item in all_checkpoints
    }
    parent_key = f"{recipes[0].name}__i00"
    parent_fp32 = checkpoint_images[parent_key]
    parent_fp32.save(output_dir / "parent-fp32-redecoded.png", format="PNG", optimize=False, compress_level=9)

    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, payload, checkpoint_images)
    if skip_qr_verify:
        _atomic_json(output_dir / "qr-verify-evidence.json", {"skipped": True})

    quality_scores: dict[str, Any] = {}
    quality_provenance: dict[str, Any] = {"skipped": True}
    if not skip_quality:
        quality_scores, quality_provenance = _score_quality(checkpoint_images, prompt, backend.settings)
    _atomic_json(output_dir / "quality-scores.json", quality_scores)
    _atomic_json(output_dir / "quality-provenance.json", quality_provenance)

    surrogate_scores, surrogate_info = score_surrogate_images(checkpoint_images)
    _atomic_json(output_dir / "e016-surrogate-scores.json", surrogate_scores)
    _atomic_json(output_dir / "e016-surrogate-status.json", surrogate_info)

    parent_quality = quality_scores.get(parent_key) or {
        "clip_score": 0.0, "clip_aesthetic": 0.0, "hpsv2_1": None
    }
    rows: list[dict[str, Any]] = []
    by_key = {(item.recipe, item.iteration): item for item in all_checkpoints}
    for recipe in recipes:
        for iteration in range(recipe.max_iterations + 1):
            item = by_key[(recipe.name, iteration)]
            key = f"{recipe.name}__i{iteration:02d}"
            image = checkpoint_images[key]
            qitem = (qr_verify or {}).get(key) or {}
            exact = int(qitem.get("conservative_exact_presets", 0))
            qscore = quality_scores.get(key) or {}
            change = image_change_metrics(image, parent_fp32)
            quality = image_quality_metrics(image)
            surrogate = surrogate_scores.get(key) or {}
            step = item.trace_step
            row = {
                "checkpoint": key,
                "method": recipe.name,
                "source": "E040",
                "profile": "scanaware_v2",
                "iteration": iteration,
                "max_iterations": recipe.max_iterations,
                "radius": recipe.latent_radius_rms,
                "gamma": config.gamma,
                "qr_verify_exact_presets": exact,
                "ssr": exact / 37.0,
                "original_exact": _qr_original_exact(qitem),
                "full_module_error_count": step["full_module_error_count"],
                "upstream_active_modules": step["upstream_active_modules"],
                "upstream_srl": step["upstream_srl"],
                "full_module_loss": step["full_module_loss"],
                "robust_loss": step["robust_loss"],
                "lpips": step["lpips_loss"],
                "latent_delta_rms": step["latent_delta_rms"],
                "acceptance_reason": step.get("acceptance_reason"),
                "image_path": item.image_path,
                "latent_path": item.latent_path,
                **change,
                **quality,
                "clip_score": qscore.get("clip_score"),
                "clip_aesthetic": qscore.get("clip_aesthetic"),
                "hpsv2_1": qscore.get("hpsv2_1"),
                "surrogate_mean_success_probability": surrogate.get("mean_success_probability"),
                "surrogate_min_success_probability": surrogate.get("min_success_probability"),
            }
            guard = _visual_guard(row, parent_quality, config) if not skip_quality else {
                "passed": (
                    float(row["lpips"]) <= config.max_lpips_for_ranking
                    and float(row["mean_absolute_change"]) <= config.max_mean_absolute_change
                    and float(row["clipped_pixel_ratio_increase"]) <= config.max_clipped_pixel_ratio_increase
                    and float(row["rgb_clipped_channel_ratio_increase"]) <= config.max_rgb_clipped_channel_ratio_increase
                ),
                "checks": {},
            }
            row["visual_guard_pass"] = guard["passed"]
            row["visual_guard_checks"] = guard["checks"]
            rows.append(row)

    safe = [row for row in rows if row["visual_guard_pass"]]
    ranked = sorted(
        safe,
        key=lambda row: (
            -int(row["qr_verify_exact_presets"]),
            -int(bool(row["original_exact"])),
            int(row["full_module_error_count"]),
            -float(row.get("surrogate_mean_success_probability") or -1.0),
            float(row["lpips"]),
            int(row["iteration"]),
            abs(float(row["radius"]) - 0.20),
        ),
    )
    winner = ranked[0] if ranked else None

    json_rows = rows
    _atomic_json(output_dir / "checkpoint-comparison.json", json_rows)
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["visual_guard_checks"] = json.dumps(flat["visual_guard_checks"], ensure_ascii=False, sort_keys=True)
        csv_rows.append(flat)
    _write_csv(output_dir / "checkpoint-comparison.csv", csv_rows)

    best_per_radius = []
    for radius in DEFAULT_RADII:
        options = [row for row in safe if math.isclose(float(row["radius"]), radius, abs_tol=1e-9)]
        if options:
            options = sorted(options, key=lambda row: (
                -int(row["qr_verify_exact_presets"]), int(row["full_module_error_count"]), float(row["lpips"])
            ))
            best_per_radius.append(options[0])
    _atomic_json(output_dir / "best-checkpoint-per-radius.json", best_per_radius)

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    qr_reference = blueprint.image.convert("RGB")
    qr_reference.save(pipeline_dir / "01-qr-reference.png", format="PNG", optimize=False, compress_level=9)
    condition = qr_reference.resize(parent.image.size, Image.Resampling.NEAREST)
    condition.save(pipeline_dir / "02-control-condition.png", format="PNG", optimize=False, compress_level=9)

    stage1_asset = Path(__file__).resolve().parents[1] / "docs/e035-assets/e034-observed-stage1.png"
    if stage1_asset.is_file():
        shutil.copy2(stage1_asset, pipeline_dir / "03-stage1.png")
    parent.image.convert("RGB").save(pipeline_dir / "04-stage2.png", format="PNG", optimize=False, compress_level=9)

    selected_image_path = selected_latent_path = None
    if winner is not None:
        selected_image_path = pipeline_dir / "99-FINAL-QR.png"
        selected_latent_path = pipeline_dir / "99-FINAL-latent.safetensors"
        shutil.copy2(Path(winner["image_path"]), selected_image_path)
        shutil.copy2(Path(winner["latent_path"]), selected_latent_path)

    advisor = advisor_preview(
        prompt=prompt,
        payload_length=len(payload),
        error_correction=str(source["error_correction"]),
        qr_context={
            "qr_version": config.qr_version,
            "qr_mask_pattern": config.qr_mask_pattern,
            "qr_module_size": config.qr_module_size,
            "qr_padding_px": config.crop_padding_px,
        },
    )
    _atomic_json(output_dir / "advisor-preview.json", advisor)

    sheet_items: list[tuple[str, Image.Image, str]] = [
        ("QR reference", qr_reference, "exact payload"),
        ("Control condition", condition, "binary QR condition"),
    ]
    if (pipeline_dir / "03-stage1.png").is_file():
        sheet_items.append(("Stage 1", Image.open(pipeline_dir / "03-stage1.png").convert("RGB"), "Cetus-Mix + QR Monster"))
    sheet_items.append(("Stage 2", parent.image.convert("RGB"), "frozen exact latent parent"))
    if winner is not None:
        winner_recipe = str(winner["method"])
        for iteration in range(0, 9):
            checkpoint = next(row for row in rows if row["method"] == winner_recipe and row["iteration"] == iteration)
            sheet_items.append((
                f"SR-MPGD i{iteration}",
                checkpoint_images[f"{winner_recipe}__i{iteration:02d}"],
                f"SSR={checkpoint['qr_verify_exact_presets']}/37 LPIPS={checkpoint['lpips']:.4f} safe={checkpoint['visual_guard_pass']}",
            ))
        sheet_items.append(("FINAL selected", Image.open(selected_image_path).convert("RGB"), f"winner={winner['checkpoint']}"))
    _comparison_sheet(pipeline_dir / "full-pipeline-contact-sheet.png", sheet_items, columns=4)

    pipeline_manifest = {
        "payload": payload,
        "prompt": prompt,
        "seed": source.get("seed"),
        "advisor_preview": advisor,
        "surrogate": surrogate_info,
        "stages": {
            "qr_reference": str(pipeline_dir / "01-qr-reference.png"),
            "control_condition": str(pipeline_dir / "02-control-condition.png"),
            "stage1": str(pipeline_dir / "03-stage1.png") if (pipeline_dir / "03-stage1.png").is_file() else None,
            "stage2": str(pipeline_dir / "04-stage2.png"),
            "srmpgd_winner_trajectory": None if winner is None else str(output_dir / winner["method"] / "images"),
            "final": str(selected_image_path) if selected_image_path else None,
            "final_latent": str(selected_latent_path) if selected_latent_path else None,
        },
        "note": "advisor recommendation is prospective; E040 optimization stays on the frozen E035 Stage-2 parent",
    }
    _atomic_json(output_dir / "pipeline-manifest.json", pipeline_manifest)

    verdict = {
        "experiment": EXPERIMENT,
        "gamma": config.gamma,
        "gamma_preserved": True,
        "radii": list(DEFAULT_RADII),
        "max_iterations": 8,
        "checkpoint_count": len(rows),
        "visual_safe_checkpoint_count": len(safe),
        "research_winner_checkpoint": None if winner is None else winner["checkpoint"],
        "research_winner_recipe": None if winner is None else winner["method"],
        "winner_iteration": None if winner is None else winner["iteration"],
        "winner_radius": None if winner is None else winner["radius"],
        "winner_ssr_exact_presets": None if winner is None else winner["qr_verify_exact_presets"],
        "winner_ssr": None if winner is None else winner["ssr"],
        "winner_original_exact": None if winner is None else winner["original_exact"],
        "winner_visual_guard_checks": None if winner is None else winner["visual_guard_checks"],
        "e039_control_ssr_exact_presets": control["verdict"].get("winner_ssr_exact_presets"),
        "e016_surrogate_research_usable": bool(surrogate_info.get("research_usable")),
        "advisor_available": bool(advisor.get("available")),
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": "REVIEW_FULL_PIPELINE_AND_WINNER_THEN_DECIDE_GENERALIZATION",
    }
    _atomic_json(output_dir / "verdict.json", verdict)
    _atomic_text(output_dir / "report.md", f"""# E040 — checkpoint frontier + final pipeline\n\n- gamma: **1000** (fixed)\n- radii: {', '.join(map(str, DEFAULT_RADII))}\n- every checkpoint i0..i8 is rescored with real QR-Verify\n- winner: **{verdict['research_winner_checkpoint']}**\n- SSR: **{verdict['winner_ssr_exact_presets']}/37**\n- E016 surrogate research-usable: **{verdict['e016_surrogate_research_usable']}**\n- advisor available: **{verdict['advisor_available']}**\n- production ready: **no**\n""")
    _atomic_json(output_dir / "e040-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--e039-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-commit", default=None)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-qr-verify", action="store_true")
    args = parser.parse_args()
    verdict = run_e040(
        parent_dir=args.parent_dir,
        e039_results_dir=args.e039_results_dir,
        output_dir=args.output_dir,
        expected_parent_commit=args.expected_parent_commit,
        skip_quality=args.skip_quality,
        skip_qr_verify=args.skip_qr_verify,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
