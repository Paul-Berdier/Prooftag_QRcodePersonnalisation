"""E039 — diagnose SR-MPGD limiter and improve scan-aware robustness.

E039 stays on the exact frozen E035/E036/E038 parent and preserves gamma=1000.
It starts from the E038 winner (hybrid r=.150, four updates), then varies only:
- number of SR-MPGD updates,
- scan-aware robust objective strength/profile,
- trust-region radius for the scan-aware profile.

Every backtracking candidate is logged with all acceptance checks so E039 can
identify the actual limiter (latent radius, LPIPS, core MAE or objective
non-increase) instead of inferring it from final images.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .e035_loss_fidelity import (
    _assert_upstream_reference_match,
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
from .e035_losses import (
    UPSTREAM_REVISION,
    module_diagnostics,
    prepare_upstream_torch_layout,
    upstream_code_scanning_robust_loss,
    upstream_qrcode_tensor,
)
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
    _downscale_restore,
    _gaussian_blur_3x3,
    _qr_original_exact,
    _score_quality,
    _visual_guard,
    full_module_margin_loss,
)

EXPERIMENT = "e039-srmpgd-limiter-scanaware-v1"
E038_REQUIRED_EXPERIMENT = "e038-srmpgd-ssr-aesthetic-frontier-v1"
E038_REQUIRED_WINNER = "e038_hybrid_r150"

Profile = Literal["e038_hybrid", "scanaware_v2"]


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    profile: Profile
    max_iterations: int
    latent_radius_rms: float
    lpips_budget: float = 0.050
    core_mae_budget: float = 0.050
    full_module_weight: float = 0.10


# First four recipes isolate iteration count using the exact E038 winner objective.
# The remaining six use stronger/multi-scale differentiable scan perturbations.
DEFAULT_RECIPES: tuple[Recipe, ...] = (
    Recipe("e039_e038hybrid_r150_i04", "e038_hybrid", 4, 0.150),
    Recipe("e039_e038hybrid_r150_i06", "e038_hybrid", 6, 0.150),
    Recipe("e039_e038hybrid_r150_i08", "e038_hybrid", 8, 0.150),
    Recipe("e039_e038hybrid_r150_i12", "e038_hybrid", 12, 0.150),
    Recipe("e039_scanaware_r150_i04", "scanaware_v2", 4, 0.150),
    Recipe("e039_scanaware_r150_i06", "scanaware_v2", 6, 0.150),
    Recipe("e039_scanaware_r150_i08", "scanaware_v2", 8, 0.150),
    Recipe("e039_scanaware_r200_i08", "scanaware_v2", 8, 0.200),
    Recipe("e039_scanaware_r300_i08", "scanaware_v2", 8, 0.300),
    Recipe("e039_scanaware_r300_i12", "scanaware_v2", 12, 0.300),
)


@dataclass(frozen=True, slots=True)
class E039Config:
    gamma: float = 1000.0
    gradient_scale: float = 32768.0
    lpips_weight: float = 0.01
    lpips_net: str = "vgg"
    crop_padding_px: int = 78
    qr_version: int = 3
    qr_mask_pattern: int = 4
    qr_module_size: int = 20
    quiet_zone_mode: str = "adaptive_light"
    quiet_zone_minimum_luminance: float = 0.90
    functional_pattern_tone_factor: float = 0.0
    upstream_reference_atol: float = 2e-6
    upstream_reference_rtol: float = 2e-6
    max_backtracks: int = 12
    minimum_alpha: float = 2 ** -12
    objective_nonincrease_tolerance: float = 2e-6

    # Ranking guard: identical philosophy to E038.
    max_lpips_for_ranking: float = 0.050
    max_mean_absolute_change: float = 0.080
    max_clipped_pixel_ratio_increase: float = 0.005
    max_rgb_clipped_channel_ratio_increase: float = 0.005
    max_abs_saturation_mean_change: float = 0.080
    max_high_saturation_ratio_increase: float = 0.050
    max_clip_score_drop: float = 0.030
    max_clip_aesthetic_drop: float = 0.250
    max_hps_drop: float = 0.020


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    objective: float
    upstream_srl: float
    full_module_loss: float
    robust_loss: float
    lpips: float
    core_mae: float
    latent_delta_rms: float
    accepted: bool
    checks: dict[str, bool]


@dataclass(frozen=True, slots=True)
class Step:
    recipe: str
    iteration: int
    elapsed_s: float
    image_sha256: str
    latent_sha256: str
    upstream_srl: float
    full_module_loss: float
    robust_loss: float
    lpips_loss: float
    objective: float
    upstream_active_modules: int
    full_module_error_count: int
    full_module_error_rate: float
    gamma: float
    latent_gradient_rms: float | None
    raw_step_rms: float | None
    projected_step_rms: float | None
    accepted_step_rms: float | None
    accepted_alpha: float | None
    latent_delta_rms: float
    effective_gradient_scale: float | None
    acceptance_reason: str | None
    candidate_checks: dict[str, bool] | None
    rejected_trial_count: int
    cuda: dict[str, int | None]


@dataclass(frozen=True, slots=True)
class RecipeResult:
    name: str
    recipe: dict[str, Any]
    output_dir: str
    final_image_path: str
    final_latent_path: str
    trace_path: str
    rejection_log_path: str
    final_step: dict[str, Any]


def recipe_catalog() -> list[dict[str, Any]]:
    return [asdict(recipe) for recipe in DEFAULT_RECIPES]


def _rms(tensor: Any) -> Any:
    return tensor.square().mean().sqrt()


def _blur_twice(images: Any) -> Any:
    return _gaussian_blur_3x3(_gaussian_blur_3x3(images))


def _upstream_loss(images: Any, blueprint: Any, layout: Any) -> Any:
    value, _ = upstream_code_scanning_robust_loss(images, blueprint, layout=layout)
    return value


def robust_scan_loss(images: Any, blueprint: Any, layout: Any, profile: Profile) -> Any:
    """Differentiable scanner perturbation ensemble.

    e038_hybrid reproduces the E038 winner robust terms exactly.
    scanaware_v2 adds stronger blur, 50% downscale and ±15% photometric cases.
    """

    total = images.new_tensor(0.0)

    # Exact E038 robust terms.
    total = total + 0.15 * _upstream_loss(_gaussian_blur_3x3(images), blueprint, layout)
    total = total + 0.15 * _upstream_loss(_downscale_restore(images, 0.75), blueprint, layout)

    brightness_90 = _upstream_loss((images * 0.90).clamp(0, 1), blueprint, layout)
    brightness_110 = _upstream_loss((images * 1.10).clamp(0, 1), blueprint, layout)
    total = total + 0.05 * (brightness_90 + brightness_110) / 2

    contrast_90 = _upstream_loss(((images - 0.5) * 0.90 + 0.5).clamp(0, 1), blueprint, layout)
    contrast_110 = _upstream_loss(((images - 0.5) * 1.10 + 0.5).clamp(0, 1), blueprint, layout)
    total = total + 0.05 * (contrast_90 + contrast_110) / 2

    if profile == "e038_hybrid":
        return total
    if profile != "scanaware_v2":
        raise ValueError(f"unknown E039 profile: {profile}")

    # Stronger/multi-scale scanner-aware terms.
    total = total + 0.10 * _upstream_loss(_blur_twice(images), blueprint, layout)
    total = total + 0.10 * _upstream_loss(_downscale_restore(images, 0.50), blueprint, layout)

    brightness_85 = _upstream_loss((images * 0.85).clamp(0, 1), blueprint, layout)
    brightness_115 = _upstream_loss((images * 1.15).clamp(0, 1), blueprint, layout)
    total = total + 0.05 * (brightness_85 + brightness_115) / 2

    contrast_85 = _upstream_loss(((images - 0.5) * 0.85 + 0.5).clamp(0, 1), blueprint, layout)
    contrast_115 = _upstream_loss(((images - 0.5) * 1.15 + 0.5).clamp(0, 1), blueprint, layout)
    total = total + 0.05 * (contrast_85 + contrast_115) / 2
    return total


def _qr_objective(
    unit: Any,
    *,
    recipe: Recipe,
    core_blueprint: Any,
    upstream_layout: Any,
    upstream_target: Any,
    official_upstream_srl: Any,
    config: E039Config,
    iteration: int,
    phase: str,
) -> tuple[Any, dict[str, float]]:
    local_upstream, _ = upstream_code_scanning_robust_loss(
        unit, core_blueprint, layout=upstream_layout
    )
    official = official_upstream_srl(unit, upstream_target)
    upstream_value, _, _ = _assert_upstream_reference_match(
        local=local_upstream,
        official=official,
        config=config,
        branch=recipe.name,
        iteration=iteration,
        phase=phase,
    )
    full = full_module_margin_loss(unit, upstream_layout)
    robust = robust_scan_loss(unit, core_blueprint, upstream_layout, recipe.profile)
    qr_objective = official + recipe.full_module_weight * full + robust
    return qr_objective, {
        "upstream_srl": upstream_value,
        "full_module_loss": float(full.detach().cpu()),
        "robust_loss": float(robust.detach().cpu()),
    }


def _core_mae(current: Any, reference: Any) -> float:
    return float(((current - reference).abs().mean() / 2).detach().cpu())


def _candidate_metrics(
    *,
    pipeline: Any,
    candidate: Any,
    initial: Any,
    reference_core: Any,
    reference_lpips: Any,
    lpips_model: Any,
    lpips_dtype: Any,
    core_blueprint: Any,
    upstream_layout: Any,
    upstream_target: Any,
    official_upstream_srl: Any,
    recipe: Recipe,
    current_objective: float,
    config: E039Config,
) -> CandidateMetrics:
    import torch

    with torch.no_grad():
        decoded = _decode_latent_tensor(pipeline, candidate).float()
        core = _crop_core(decoded, config.crop_padding_px).detach()
        unit = (core / 2 + 0.5).clamp(0, 1)
        qr_tensor, parts = _qr_objective(
            unit,
            recipe=recipe,
            core_blueprint=core_blueprint,
            upstream_layout=upstream_layout,
            upstream_target=upstream_target,
            official_upstream_srl=official_upstream_srl,
            config=config,
            iteration=-1,
            phase="candidate",
        )
        lpips = float(
            lpips_model(
                core.to(device="cpu", dtype=lpips_dtype),
                reference_lpips,
            ).mean().detach().cpu()
        )
        objective = float(qr_tensor.detach().cpu()) + config.lpips_weight * lpips
        core_mae = _core_mae(core, reference_core)
        latent_delta = float(_rms(candidate - initial).detach().cpu())

    checks = {
        "latent_radius": latent_delta <= recipe.latent_radius_rms + 1e-9,
        "lpips_budget": lpips <= recipe.lpips_budget + 1e-9,
        "core_mae_budget": core_mae <= recipe.core_mae_budget + 1e-9,
        "objective_nonincrease": (
            objective <= current_objective + config.objective_nonincrease_tolerance
        ),
    }
    return CandidateMetrics(
        objective=objective,
        upstream_srl=parts["upstream_srl"],
        full_module_loss=parts["full_module_loss"],
        robust_loss=parts["robust_loss"],
        lpips=lpips,
        core_mae=core_mae,
        latent_delta_rms=latent_delta,
        accepted=all(checks.values()),
        checks=checks,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_steps(steps: list[Step]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        row = asdict(step)
        checks = row.pop("candidate_checks") or {}
        cuda = row.pop("cuda") or {}
        row.update({f"check_{key}": value for key, value in checks.items()})
        row.update({f"cuda_{key}": value for key, value in cuda.items()})
        rows.append(row)
    return rows


def _run_recipe(
    *,
    pipeline: Any,
    parent: LoadedParentArtifact,
    blueprint: Any,
    recipe: Recipe,
    config: E039Config,
    output_root: Path,
) -> RecipeResult:
    import torch
    from safetensors.torch import save_file

    branch_root = output_root / recipe.name
    images_root = branch_root / "images"
    branch_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

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
        core_blueprint,
        core_height,
        core_width,
        device=device,
        dtype=torch.float32,
        center_fraction=1 / 3,
    )
    upstream_layout = prepare_upstream_torch_layout(
        core_blueprint,
        core_height,
        core_width,
        device=device,
        dtype=torch.float32,
    )
    upstream_target = upstream_qrcode_tensor(
        core_blueprint,
        core_height,
        core_width,
        device=device,
        dtype=torch.float32,
    )
    official_upstream_srl = _load_official_upstream_srl(
        config.qr_module_size,
        device=device,
    )
    lpips_model = _load_lpips(pipeline, net=config.lpips_net)
    lpips_parameter = next(iter(lpips_model.parameters()), None)
    lpips_dtype = lpips_parameter.dtype if lpips_parameter is not None else torch.float32
    reference_lpips = reference_core.to(device="cpu", dtype=lpips_dtype).detach()

    steps: list[Step] = []
    rejection_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for iteration in range(recipe.max_iterations + 1):
        working = working.detach()
        with torch.no_grad():
            decoded = (
                reference_decoded
                if iteration == 0
                else _decode_latent_tensor(pipeline, working).float()
            )
            decoded_core = _crop_core(decoded, config.crop_padding_px).detach()
            decoded_unit = (decoded_core / 2 + 0.5).clamp(0, 1)
            qr_tensor, parts = _qr_objective(
                decoded_unit,
                recipe=recipe,
                core_blueprint=core_blueprint,
                upstream_layout=upstream_layout,
                upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl,
                config=config,
                iteration=iteration,
                phase="evaluation",
            )
            lpips_loss = float(
                lpips_model(
                    decoded_core.to(device="cpu", dtype=lpips_dtype),
                    reference_lpips,
                ).mean().detach().cpu()
            )
            qr_objective_value = float(qr_tensor.detach().cpu())
            objective = qr_objective_value + config.lpips_weight * lpips_loss
            diagnostics = module_diagnostics(
                decoded_unit,
                core_blueprint,
                paper_layout=paper_layout,
                upstream_layout=upstream_layout,
            )

        image = _decoded_to_scan_ready_image(pipeline, decoded, blueprint, config)
        image_path = images_root / f"iteration-{iteration:03d}.png"
        image.save(image_path, format="PNG", optimize=False, compress_level=9)

        latent_gradient_rms = None
        raw_step_rms = None
        projected_step_rms = None
        accepted_step_rms = None
        accepted_alpha = None
        effective_gradient_scale = None
        acceptance_reason = None
        candidate_checks = None
        next_working = working
        rejected_this_iteration = 0

        if iteration < recipe.max_iterations and qr_objective_value > 0.0:
            objective_core = decoded_core.detach().requires_grad_(True)
            objective_unit = (objective_core / 2 + 0.5).clamp(0, 1)
            qr_loss, _ = _qr_objective(
                objective_unit,
                recipe=recipe,
                core_blueprint=core_blueprint,
                upstream_layout=upstream_layout,
                upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl,
                config=config,
                iteration=iteration,
                phase="gradient",
            )
            qr_gradient = torch.autograd.grad(qr_loss, objective_core, only_inputs=True)[0]

            lpips_core = (
                decoded_core.detach()
                .to(device="cpu", dtype=lpips_dtype)
                .requires_grad_(True)
            )
            lpips_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(
                lpips_tensor, lpips_core, only_inputs=True
            )[0]

            objective_gradient = qr_gradient.clone()
            if config.lpips_weight:
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
                projected_target = project_latent_candidate(
                    raw_target,
                    initial,
                    recipe.latent_radius_rms,
                )
                projected_step_rms = float(_rms(projected_target - working).detach().cpu())
                direction = projected_target - working

                alpha = 1.0
                accepted_metrics: CandidateMetrics | None = None
                last_metrics: CandidateMetrics | None = None
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
                        recipe=recipe,
                        current_objective=objective,
                        config=config,
                    )
                    last_metrics = metrics
                    failed_checks = [key for key, passed in metrics.checks.items() if not passed]
                    rejection_rows.append(
                        {
                            "recipe": recipe.name,
                            "profile": recipe.profile,
                            "iteration": iteration,
                            "backtrack_index": backtrack_index,
                            "alpha": alpha,
                            "accepted": metrics.accepted,
                            "rejection_reasons": ";".join(failed_checks),
                            "objective": metrics.objective,
                            "upstream_srl": metrics.upstream_srl,
                            "full_module_loss": metrics.full_module_loss,
                            "robust_loss": metrics.robust_loss,
                            "lpips": metrics.lpips,
                            "core_mae": metrics.core_mae,
                            "latent_delta_rms": metrics.latent_delta_rms,
                            **{f"check_{key}": passed for key, passed in metrics.checks.items()},
                        }
                    )
                    if metrics.accepted:
                        next_working = trial
                        accepted_metrics = metrics
                        accepted_alpha = alpha
                        accepted_step_rms = float(_rms(next_working - working).detach().cpu())
                        acceptance_reason = "accepted"
                        candidate_checks = metrics.checks
                        break
                    rejected_this_iteration += 1
                    alpha *= 0.5

                if accepted_metrics is None:
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
        steps.append(
            Step(
                recipe=recipe.name,
                iteration=iteration,
                elapsed_s=time.perf_counter() - started,
                image_sha256=_image_sha256(image),
                latent_sha256=tensor_sha256(working),
                upstream_srl=parts["upstream_srl"],
                full_module_loss=parts["full_module_loss"],
                robust_loss=parts["robust_loss"],
                lpips_loss=lpips_loss,
                objective=objective,
                upstream_active_modules=int(diagnostics["upstream_margin_active_count"]),
                full_module_error_count=int(diagnostics["full_module_error_count"]),
                full_module_error_rate=float(diagnostics["full_module_error_rate"]),
                gamma=config.gamma,
                latent_gradient_rms=latent_gradient_rms,
                raw_step_rms=raw_step_rms,
                projected_step_rms=projected_step_rms,
                accepted_step_rms=accepted_step_rms,
                accepted_alpha=accepted_alpha,
                latent_delta_rms=latent_delta_rms,
                effective_gradient_scale=effective_gradient_scale,
                acceptance_reason=acceptance_reason,
                candidate_checks=candidate_checks,
                rejected_trial_count=rejected_this_iteration,
                cuda=_cuda_snapshot(),
            )
        )
        working = next_working.detach()
        gc.collect()
        torch.cuda.empty_cache()

    final_image_path = images_root / f"iteration-{recipe.max_iterations:03d}.png"
    final_latent_path = branch_root / "final-latent.safetensors"
    save_file({"latent": working.detach().cpu().contiguous()}, str(final_latent_path))

    trace_path = branch_root / "trace.json"
    rejection_path = branch_root / "rejection-log.json"
    _atomic_json(trace_path, [asdict(step) for step in steps])
    _atomic_json(rejection_path, rejection_rows)
    _write_csv(branch_root / "trace.csv", _flatten_steps(steps))
    _write_csv(branch_root / "rejection-log.csv", rejection_rows)

    result = RecipeResult(
        name=recipe.name,
        recipe=asdict(recipe),
        output_dir=str(branch_root),
        final_image_path=str(final_image_path),
        final_latent_path=str(final_latent_path),
        trace_path=str(trace_path),
        rejection_log_path=str(rejection_path),
        final_step=asdict(steps[-1]),
    )
    _atomic_json(branch_root / "recipe-result.json", asdict(result))
    return result


def _load_e038_control(e038_results_dir: Path) -> tuple[dict[str, Any], Image.Image]:
    verdict_path = e038_results_dir / "verdict.json"
    comparison_path = e038_results_dir / "method-comparison.json"
    image_path = e038_results_dir / E038_REQUIRED_WINNER / "images/iteration-004.png"
    if not verdict_path.is_file():
        raise FileNotFoundError(verdict_path)
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    if verdict.get("experiment") != E038_REQUIRED_EXPERIMENT:
        raise RuntimeError(f"unexpected E038 experiment: {verdict.get('experiment')}")
    if verdict.get("gamma") != 1000.0 or verdict.get("gamma_preserved") is not True:
        raise RuntimeError("E039 requires E038 gamma=1000 preserved")
    if verdict.get("research_winner") != E038_REQUIRED_WINNER:
        raise RuntimeError(f"E039 requires winner {E038_REQUIRED_WINNER}")
    if not comparison_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("E038 winner comparison/image missing")
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))
    row = next((item for item in rows if item.get("method") == E038_REQUIRED_WINNER), None)
    if row is None:
        raise RuntimeError("E038 winner row missing")
    return row, Image.open(image_path).convert("RGB")


def _blocker_summary(results: dict[str, RecipeResult]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    check_names = ("latent_radius", "lpips_budget", "core_mae_budget", "objective_nonincrease")
    for name, result in results.items():
        rejection_rows = json.loads(Path(result.rejection_log_path).read_text(encoding="utf-8"))
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        counts = {key: 0 for key in check_names}
        accepted_trials = 0
        for row in rejection_rows:
            if row.get("accepted"):
                accepted_trials += 1
            for key in check_names:
                if row.get(f"check_{key}") is False:
                    counts[key] += 1
        dominant = None
        if any(counts.values()):
            dominant = max(counts, key=counts.get)
        summaries.append(
            {
                "recipe": name,
                "profile": result.recipe["profile"],
                "max_iterations": result.recipe["max_iterations"],
                "radius": result.recipe["latent_radius_rms"],
                "candidate_trials": len(rejection_rows),
                "accepted_trials": accepted_trials,
                "rejected_trials": len(rejection_rows) - accepted_trials,
                "rejected_by_latent_radius": counts["latent_radius"],
                "rejected_by_lpips_budget": counts["lpips_budget"],
                "rejected_by_core_mae_budget": counts["core_mae_budget"],
                "rejected_by_objective_nonincrease": counts["objective_nonincrease"],
                "dominant_blocker": dominant,
                "accepted_updates": sum(1 for row in trace if row.get("acceptance_reason") == "accepted"),
                "rejected_all_iterations": sum(
                    1 for row in trace if row.get("acceptance_reason") == "trust_region_rejected_all_candidates"
                ),
                "objective_zero_holds": sum(
                    1 for row in trace if row.get("acceptance_reason") == "objective_zero_hold_state"
                ),
                "minimum_accepted_alpha": min(
                    [float(row["accepted_alpha"]) for row in trace if row.get("accepted_alpha") not in (None, 0, 0.0)],
                    default=None,
                ),
                "final_latent_delta_rms": result.final_step["latent_delta_rms"],
                "final_lpips": result.final_step["lpips_loss"],
            }
        )
    return summaries


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "e039-artifact-manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def run_e039(
    *,
    parent_dir: Path,
    e038_results_dir: Path,
    output_dir: Path,
    recipes: tuple[Recipe, ...] = DEFAULT_RECIPES,
    config: E039Config | None = None,
    expected_parent_commit: str | None = None,
    skip_qr_verify: bool = False,
    skip_quality: bool = False,
) -> dict[str, Any]:
    import torch

    from .quality import image_change_metrics, image_quality_metrics
    from .qr import generate_diffqrcoder_qr

    config = config or E039Config()
    if not torch.cuda.is_available():
        raise RuntimeError("E039 requires an available CUDA GPU")
    if config.gamma != 1000.0:
        raise ValueError("E039 keeps gamma fixed at 1000")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E039 output directory must be empty: {output_dir}")
    if len(recipes) != 10:
        raise ValueError("E039 preregistered grid contains exactly 10 recipes")

    e038_control_row, e038_control_image = _load_e038_control(e038_results_dir)

    expected_parent = {
        "qr_version": config.qr_version,
        "qr_mask_pattern": config.qr_mask_pattern,
        "qr_module_size": config.qr_module_size,
        "qr_padding_px": config.crop_padding_px,
        "diffqrcoder_revision": UPSTREAM_REVISION,
        "stage1_image_sha256": E034_OBSERVED_STAGE1_IMAGE_SHA256,
        "stage1_file_sha256": E034_OBSERVED_STAGE1_FILE_SHA256,
    }
    if expected_parent_commit:
        expected_parent["source_commit"] = expected_parent_commit
    parent = load_parent_artifact(parent_dir, device="cpu", expected=expected_parent)
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
    _atomic_json(
        output_dir / "plan.json",
        {
            "experiment": EXPERIMENT,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "recipes": [asdict(recipe) for recipe in recipes],
            "gamma_is_fixed": True,
            "e038_required_winner": E038_REQUIRED_WINNER,
            "selection_priority": [
                "visual_guard_pass",
                "qr_verify_exact_presets_desc",
                "original_decode_desc",
                "full_module_error_count_asc",
                "lpips_asc",
            ],
            "production_ready": False,
        },
    )
    _atomic_json(output_dir / "parent-verification.json", parent.metadata)
    _atomic_json(output_dir / "e038-control.json", e038_control_row)

    backend, pipeline = _load_pipeline()
    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)

    results: dict[str, RecipeResult] = {}
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            try:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                torch.cuda.reset_peak_memory_stats()
                _atomic_json(
                    output_dir / "runtime.json",
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0),
                        "offloaded_modules": list(offloaded),
                        "vae_original_dtype": str(original_vae_dtype),
                        "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                        "diffqrcoder_revision": str(backend.settings.diffqrcoder_revision),
                        "parent_source_commit": str(source["source_commit"]),
                    },
                )
                for recipe in recipes:
                    results[recipe.name] = _run_recipe(
                        pipeline=pipeline,
                        parent=parent,
                        blueprint=blueprint,
                        recipe=recipe,
                        config=config,
                        output_root=output_dir,
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

    first_result = next(iter(results.values()))
    parent_fp32_path = Path(first_result.output_dir) / "images/iteration-000.png"
    parent_fp32 = Image.open(parent_fp32_path).convert("RGB")
    parent_fp32.save(output_dir / "parent-fp32-redecoded.png", format="PNG", optimize=False, compress_level=9)

    all_images: dict[str, Image.Image] = {
        "parent_fp32": parent_fp32,
        "e038_hybrid_r150": e038_control_image,
    }
    for name, result in results.items():
        all_images[name] = Image.open(result.final_image_path).convert("RGB")

    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, payload, all_images)
    if skip_qr_verify:
        _atomic_json(output_dir / "qr-verify-evidence.json", {"skipped": True})

    quality_scores: dict[str, Any] = {}
    quality_provenance: dict[str, Any] = {"skipped": True}
    if not skip_quality:
        quality_scores, quality_provenance = _score_quality(all_images, prompt, backend.settings)
    _atomic_json(output_dir / "quality-scores.json", quality_scores)
    _atomic_json(output_dir / "quality-provenance.json", quality_provenance)

    parent_quality = quality_scores.get("parent_fp32") or {
        "clip_score": 0.0,
        "clip_aesthetic": 0.0,
        "hpsv2_1": None,
    }

    rows: list[dict[str, Any]] = []
    for recipe in recipes:
        result = results[recipe.name]
        final = result.final_step
        image = all_images[recipe.name]
        change = image_change_metrics(image, parent_fp32)
        quality = image_quality_metrics(image)
        qr_item = (qr_verify or {}).get(recipe.name) or {}
        exact = int(qr_item.get("conservative_exact_presets", 0))
        original_exact = _qr_original_exact(qr_item)
        qscore = quality_scores.get(recipe.name) or {}
        row = {
            "method": recipe.name,
            "source": "E039",
            "profile": recipe.profile,
            "max_iterations": recipe.max_iterations,
            "gamma": config.gamma,
            "radius": recipe.latent_radius_rms,
            "qr_verify_exact_presets": exact,
            "ssr": exact / 37.0,
            "original_exact": original_exact,
            "full_module_error_count": final["full_module_error_count"],
            "upstream_active_modules": final["upstream_active_modules"],
            "upstream_srl": final["upstream_srl"],
            "full_module_loss": final["full_module_loss"],
            "robust_loss": final["robust_loss"],
            "lpips": final["lpips_loss"],
            "latent_delta_rms": final["latent_delta_rms"],
            **change,
            **quality,
            "clip_score": qscore.get("clip_score"),
            "clip_aesthetic": qscore.get("clip_aesthetic"),
            "hpsv2_1": qscore.get("hpsv2_1"),
            "historical": False,
        }
        guard = _visual_guard(row, parent_quality, config) if not skip_quality else {
            "passed": (
                row["lpips"] <= config.max_lpips_for_ranking
                and row["mean_absolute_change"] <= config.max_mean_absolute_change
                and row["clipped_pixel_ratio_increase"] <= config.max_clipped_pixel_ratio_increase
                and row["rgb_clipped_channel_ratio_increase"] <= config.max_rgb_clipped_channel_ratio_increase
            ),
            "checks": {},
        }
        row["visual_guard_pass"] = guard["passed"]
        row["visual_guard_checks"] = guard["checks"]
        rows.append(row)

    blocker_summary = _blocker_summary(results)
    blocker_by_recipe = {row["recipe"]: row for row in blocker_summary}
    for row in rows:
        row.update({
            "dominant_blocker": blocker_by_recipe[row["method"]]["dominant_blocker"],
            "accepted_updates": blocker_by_recipe[row["method"]]["accepted_updates"],
            "rejected_all_iterations": blocker_by_recipe[row["method"]]["rejected_all_iterations"],
        })

    _atomic_json(output_dir / "blocker-summary.json", blocker_summary)
    _write_csv(output_dir / "blocker-summary.csv", blocker_summary)

    safe = [row for row in rows if row["visual_guard_pass"]]
    ranked = sorted(
        safe,
        key=lambda row: (
            -int(row["qr_verify_exact_presets"]),
            -int(bool(row["original_exact"])),
            int(row["full_module_error_count"]),
            float(row["lpips"]),
            float(row["latent_delta_rms"]),
        ),
    )
    winner = ranked[0]["method"] if ranked else None

    control_row = dict(e038_control_row)
    control_row["source"] = "E038"
    control_row["historical"] = True
    all_rows = [control_row] + rows
    _atomic_json(output_dir / "method-comparison.json", all_rows)
    with (output_dir / "method-comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({key for row in all_rows for key in row if key != "visual_guard_checks"})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    items: list[tuple[str, Image.Image, str]] = [
        ("Parent FP32", parent_fp32, "reference"),
        (
            "E038 winner hybrid r=.15 i4",
            e038_control_image,
            f"SSR={e038_control_row.get('qr_verify_exact_presets', '?')}/37  LPIPS={e038_control_row.get('lpips', float('nan')):.4f}",
        ),
    ]
    row_by_method = {row["method"]: row for row in rows}
    for recipe in recipes:
        row = row_by_method[recipe.name]
        items.append(
            (
                recipe.name.replace("e039_", "E039 "),
                all_images[recipe.name],
                (
                    f"SSR={row['qr_verify_exact_presets']}/37  MER={row['full_module_error_count']}/841  "
                    f"LPIPS={row['lpips']:.4f}  safe={row['visual_guard_pass']}"
                ),
            )
        )
    _comparison_sheet(output_dir / "e039-all-methods-contact-sheet.png", items, columns=4)

    verdict = {
        "experiment": EXPERIMENT,
        "gamma": config.gamma,
        "gamma_preserved": True,
        "recipe_count": len(recipes),
        "visual_safe_recipe_count": len(safe),
        "research_winner": winner,
        "winner_ssr_exact_presets": None if not ranked else ranked[0]["qr_verify_exact_presets"],
        "winner_ssr": None if not ranked else ranked[0]["ssr"],
        "winner_original_exact": None if not ranked else ranked[0]["original_exact"],
        "winner_dominant_blocker": None if not ranked else ranked[0].get("dominant_blocker"),
        "e038_control_ssr_exact_presets": e038_control_row.get("qr_verify_exact_presets"),
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": "REVIEW_LIMITER_AND_SCAN_AWARE_WINNER_BEFORE_GENERALIZATION",
    }
    _atomic_json(output_dir / "verdict.json", verdict)

    report = f"""# E039 — SR-MPGD limiter + scan-aware optimization

- Same frozen parent as E035/E036/E038.
- Gamma fixed to **1000** for every recipe.
- E038 required control: **{E038_REQUIRED_WINNER}**.
- New recipes: **{len(recipes)}**.
- E039 logs every backtracking candidate and every failed acceptance check.
- Research winner: **{winner or 'none'}**.
- Production ready: **no**.

The experiment first isolates update count on the exact E038 hybrid objective, then tests a
stronger multi-scale scan-aware objective. Ranking remains visual-safety first, then SSR,
direct/original decode, module errors and LPIPS.
"""
    _atomic_text(output_dir / "report.md", report)
    _atomic_json(output_dir / "e039-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--e038-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-commit", default=None)
    parser.add_argument("--skip-qr-verify", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    verdict = run_e039(
        parent_dir=args.parent_dir,
        e038_results_dir=args.e038_results_dir,
        output_dir=args.output_dir,
        expected_parent_commit=args.expected_parent_commit,
        skip_qr_verify=args.skip_qr_verify,
        skip_quality=args.skip_quality,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
