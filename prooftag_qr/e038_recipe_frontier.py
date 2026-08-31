"""E038 — SR-MPGD SSR/aesthetic frontier search.

E038 is a single-parent recipe search. It keeps gamma=1000 and the pinned public
DiffQRCoder scanning-robust loss, then varies only the trust-region radius and a
small set of preregistered QR objectives. The goal is to maximize conservative
QR-Verify SSR under explicit image-preservation guards, not to prove generalization.

Historical E035/E036 controls are loaded from their immutable result directories and
shown beside every E038 candidate. E033/E034 are kept as numeric historical baselines
because their archived parents are not assumed to be byte-identical to the current
frozen E035 parent.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

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

EXPERIMENT = "e038-srmpgd-ssr-aesthetic-frontier-v1"

ObjectiveKind = Literal["upstream", "full", "robust", "hybrid"]


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    latent_radius_rms: float
    objective_kind: ObjectiveKind
    lpips_budget: float = 0.050
    core_mae_budget: float = 0.050
    full_module_weight: float = 0.0
    robust_blur_weight: float = 0.0
    robust_downscale_weight: float = 0.0
    robust_brightness_weight: float = 0.0
    robust_contrast_weight: float = 0.0
    full_dark_threshold: float = 0.48
    full_light_threshold: float = 0.52


# Phase A: isolate trust-region radius with the exact upstream SRL.
RADIUS_RECIPES: tuple[Recipe, ...] = tuple(
    Recipe(
        name=f"e038_upstream_r{int(radius * 1000):03d}",
        latent_radius_rms=radius,
        objective_kind="upstream",
    )
    for radius in (0.075, 0.100, 0.125, 0.150, 0.200, 0.300)
)

# Phase B: add only one or two scientifically motivated QR terms at two moderate radii.
OBJECTIVE_RECIPES: tuple[Recipe, ...] = (
    Recipe(
        name="e038_full_r100",
        latent_radius_rms=0.100,
        objective_kind="full",
        full_module_weight=0.10,
    ),
    Recipe(
        name="e038_robust_r100",
        latent_radius_rms=0.100,
        objective_kind="robust",
        robust_blur_weight=0.15,
        robust_downscale_weight=0.15,
        robust_brightness_weight=0.05,
        robust_contrast_weight=0.05,
    ),
    Recipe(
        name="e038_hybrid_r100",
        latent_radius_rms=0.100,
        objective_kind="hybrid",
        full_module_weight=0.10,
        robust_blur_weight=0.15,
        robust_downscale_weight=0.15,
        robust_brightness_weight=0.05,
        robust_contrast_weight=0.05,
    ),
    Recipe(
        name="e038_hybrid_r150",
        latent_radius_rms=0.150,
        objective_kind="hybrid",
        full_module_weight=0.10,
        robust_blur_weight=0.15,
        robust_downscale_weight=0.15,
        robust_brightness_weight=0.05,
        robust_contrast_weight=0.05,
    ),
)

DEFAULT_RECIPES: tuple[Recipe, ...] = RADIUS_RECIPES + OBJECTIVE_RECIPES


@dataclass(frozen=True, slots=True)
class E038Config:
    max_iterations: int = 4
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

    # Research guards used for ranking; they do not alter candidate updates.
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
class E038Step:
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
    cuda: dict[str, int | None]


@dataclass(frozen=True, slots=True)
class RecipeResult:
    name: str
    recipe: dict[str, Any]
    output_dir: str
    final_image_path: str
    final_latent_path: str
    trace_path: str
    final_step: dict[str, Any]


def recipe_catalog() -> list[dict[str, Any]]:
    return [asdict(recipe) for recipe in DEFAULT_RECIPES]


def _rms(tensor: Any) -> Any:
    return tensor.square().mean().sqrt()


def _upstream_luminance(images: Any) -> Any:
    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError("images must be BCHW")
    if images.shape[1] == 1:
        return images[:, 0]
    coefficients = images.new_tensor((0.2999, 0.587, 0.1114)).view(1, 3, 1, 1)
    return (images * coefficients).sum(dim=1)


def _module_means(images: Any, layout: Any) -> Any:
    import torch

    gray = _upstream_luminance(images)
    batch = gray.shape[0]
    flat = gray.reshape(batch, -1)
    ids = layout.module_ids.unsqueeze(0).expand(batch, -1)
    sums = torch.zeros(
        (batch, layout.module_count),
        device=images.device,
        dtype=images.dtype,
    )
    counts = torch.zeros_like(sums)
    ones = torch.ones_like(flat)
    sums.scatter_add_(1, ids, flat)
    counts.scatter_add_(1, ids, ones)
    return sums / counts.clamp_min(1.0)


def full_module_margin_loss(
    images: Any,
    layout: Any,
    *,
    dark_threshold: float = 0.48,
    light_threshold: float = 0.52,
) -> Any:
    """Soft full-module margin complementary to upstream centre stopping."""

    import torch

    if not 0 < dark_threshold < light_threshold < 1:
        raise ValueError("invalid full-module thresholds")
    means = _module_means(images, layout)
    target_dark = layout.target_dark.unsqueeze(0).expand_as(means)
    violations = torch.where(
        target_dark,
        torch.relu(means - dark_threshold),
        torch.relu(light_threshold - means),
    )
    return violations.mean()


def _gaussian_blur_3x3(images: Any) -> Any:
    import torch.nn.functional as F

    kernel = images.new_tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ) / 16.0
    kernel = kernel.view(1, 1, 3, 3).repeat(images.shape[1], 1, 1, 1)
    return F.conv2d(images, kernel, padding=1, groups=images.shape[1])


def _downscale_restore(images: Any, factor: float = 0.75) -> Any:
    import torch.nn.functional as F

    height, width = images.shape[-2:]
    small = F.interpolate(
        images,
        scale_factor=factor,
        mode="bilinear",
        align_corners=False,
        recompute_scale_factor=True,
    )
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def robust_scan_loss(images: Any, blueprint: Any, layout: Any, recipe: Recipe) -> Any:
    """Differentiable scan perturbations evaluated with the faithful local upstream SRL."""

    total = images.new_tensor(0.0)
    if recipe.robust_blur_weight:
        value, _ = upstream_code_scanning_robust_loss(
            _gaussian_blur_3x3(images), blueprint, layout=layout
        )
        total = total + recipe.robust_blur_weight * value
    if recipe.robust_downscale_weight:
        value, _ = upstream_code_scanning_robust_loss(
            _downscale_restore(images), blueprint, layout=layout
        )
        total = total + recipe.robust_downscale_weight * value
    if recipe.robust_brightness_weight:
        low = (images * 0.90).clamp(0, 1)
        high = (images * 1.10).clamp(0, 1)
        low_loss, _ = upstream_code_scanning_robust_loss(low, blueprint, layout=layout)
        high_loss, _ = upstream_code_scanning_robust_loss(high, blueprint, layout=layout)
        total = total + recipe.robust_brightness_weight * (low_loss + high_loss) / 2
    if recipe.robust_contrast_weight:
        low = ((images - 0.5) * 0.90 + 0.5).clamp(0, 1)
        high = ((images - 0.5) * 1.10 + 0.5).clamp(0, 1)
        low_loss, _ = upstream_code_scanning_robust_loss(low, blueprint, layout=layout)
        high_loss, _ = upstream_code_scanning_robust_loss(high, blueprint, layout=layout)
        total = total + recipe.robust_contrast_weight * (low_loss + high_loss) / 2
    return total


def _qr_objective(
    unit: Any,
    *,
    recipe: Recipe,
    core_blueprint: Any,
    upstream_layout: Any,
    upstream_target: Any,
    official_upstream_srl: Any,
    config: E038Config,
    iteration: int,
    phase: str,
) -> tuple[Any, dict[str, float]]:
    local_upstream, _ = upstream_code_scanning_robust_loss(
        unit,
        core_blueprint,
        layout=upstream_layout,
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
    full = full_module_margin_loss(
        unit,
        upstream_layout,
        dark_threshold=recipe.full_dark_threshold,
        light_threshold=recipe.full_light_threshold,
    )
    robust = robust_scan_loss(unit, core_blueprint, upstream_layout, recipe)
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
    config: E038Config,
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


def _write_trace_csv(path: Path, steps: list[E038Step]) -> None:
    rows: list[dict[str, Any]] = []
    for step in steps:
        row = asdict(step)
        checks = row.pop("candidate_checks") or {}
        cuda = row.pop("cuda") or {}
        row.update({f"check_{key}": value for key, value in checks.items()})
        row.update({f"cuda_{key}": value for key, value in cuda.items()})
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_recipe(
    *,
    pipeline: Any,
    parent: LoadedParentArtifact,
    blueprint: Any,
    recipe: Recipe,
    config: E038Config,
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

    steps: list[E038Step] = []
    started = time.perf_counter()

    for iteration in range(config.max_iterations + 1):
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

        if iteration < config.max_iterations and qr_objective_value > 0.0:
            srl_core = decoded_core.detach().requires_grad_(True)
            srl_unit = (srl_core / 2 + 0.5).clamp(0, 1)
            qr_loss, _ = _qr_objective(
                srl_unit,
                recipe=recipe,
                core_blueprint=core_blueprint,
                upstream_layout=upstream_layout,
                upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl,
                config=config,
                iteration=iteration,
                phase="gradient",
            )
            qr_gradient = torch.autograd.grad(qr_loss, srl_core, only_inputs=True)[0]

            lpips_core = (
                decoded_core.detach()
                .to(device="cpu", dtype=lpips_dtype)
                .requires_grad_(True)
            )
            lpips_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(
                lpips_tensor,
                lpips_core,
                only_inputs=True,
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

            del srl_core, srl_unit, qr_loss, qr_gradient
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

                accepted_metrics: CandidateMetrics | None = None
                alpha = 1.0
                last_metrics: CandidateMetrics | None = None
                for _ in range(config.max_backtracks + 1):
                    if alpha < config.minimum_alpha:
                        break
                    trial = (working + alpha * direction).detach()
                    last_metrics = _candidate_metrics(
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
                    if last_metrics.accepted:
                        next_working = trial
                        accepted_metrics = last_metrics
                        accepted_alpha = alpha
                        accepted_step_rms = float(_rms(next_working - working).detach().cpu())
                        acceptance_reason = "accepted"
                        candidate_checks = last_metrics.checks
                        break
                    alpha *= 0.5

                if accepted_metrics is None:
                    acceptance_reason = "trust_region_rejected_all_candidates"
                    accepted_alpha = 0.0
                    accepted_step_rms = 0.0
                    candidate_checks = last_metrics.checks if last_metrics else None

                del gradient, raw_target, projected_target, direction
        elif iteration < config.max_iterations:
            acceptance_reason = "objective_zero_hold_state"
            accepted_alpha = 0.0
            accepted_step_rms = 0.0

        latent_delta_rms = float(_rms(working - initial).detach().cpu())
        step = E038Step(
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
            cuda=_cuda_snapshot(),
        )
        steps.append(step)
        working = next_working.detach()
        gc.collect()
        torch.cuda.empty_cache()

    final_image_path = images_root / f"iteration-{config.max_iterations:03d}.png"
    final_latent_path = branch_root / "final-latent.safetensors"
    save_file({"latent": working.detach().cpu().contiguous()}, str(final_latent_path))
    trace_path = branch_root / "trace.json"
    _atomic_json(trace_path, [asdict(step) for step in steps])
    _write_trace_csv(branch_root / "trace.csv", steps)
    result = RecipeResult(
        name=recipe.name,
        recipe=asdict(recipe),
        output_dir=str(branch_root),
        final_image_path=str(final_image_path),
        final_latent_path=str(final_latent_path),
        trace_path=str(trace_path),
        final_step=asdict(steps[-1]),
    )
    _atomic_json(branch_root / "recipe-result.json", asdict(result))
    return result


def _historical_images(e035_dir: Path, e036_dir: Path) -> dict[str, Image.Image]:
    candidates = {
        "parent_fp32": e036_dir / "parent-fp32-redecoded.png",
        "e035_paper": e035_dir / "e035_paper_srl_control/images/iteration-004.png",
        "e035_upstream_unbounded": e035_dir / "e035_upstream_code_srl/images/iteration-004.png",
        "e036_global_r050": e036_dir / "e036_gamma1000_global_trust/images/iteration-004.png",
        "e036_strict_r025": e036_dir / "e036_gamma1000_strict_trust/images/iteration-004.png",
        "e036_local_r050": e036_dir / "e036_gamma1000_local_preserve/images/iteration-004.png",
    }
    result: dict[str, Image.Image] = {}
    for name, path in candidates.items():
        if path.is_file():
            result[name] = Image.open(path).convert("RGB")
    return result


def _historical_rows(e035_dir: Path, e036_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    e035_qr_path = e035_dir / "qr-verify-evidence.json"
    e035_qr = json.loads(e035_qr_path.read_text(encoding="utf-8")) if e035_qr_path.is_file() else {}
    for name, path in (
        ("e035_paper", e035_dir / "e035_paper_srl_control/branch-result.json"),
        ("e035_upstream_unbounded", e035_dir / "e035_upstream_code_srl/branch-result.json"),
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        final = data["final_step"]
        qr_key = "e035_paper_srl_control" if name == "e035_paper" else "e035_upstream_code_srl"
        exact = int((e035_qr.get(qr_key) or {}).get("conservative_exact_presets", 0))
        rows.append(
            {
                "method": name,
                "source": "E035",
                "gamma": 1000.0,
                "radius": None,
                "objective_kind": (
                    "paper_v3" if name == "e035_paper" else "upstream_unbounded"
                ),
                "qr_verify_exact_presets": exact,
                "ssr": exact / 37.0,
                "full_module_error_count": final["diagnostics"]["full_module_error_count"],
                "upstream_active_modules": final["diagnostics"]["upstream_margin_active_count"],
                "lpips": final["lpips_loss"],
                "latent_delta_rms": final["latent_delta_rms"],
                "historical": True,
            }
        )
    summary_path = e036_dir / "branch-summary.json"
    if summary_path.is_file():
        name_map = {
            "e036_gamma1000_global_trust": "e036_global_r050",
            "e036_gamma1000_strict_trust": "e036_strict_r025",
            "e036_gamma1000_local_preserve": "e036_local_r050",
        }
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            rows.append(
                {
                    "method": name_map.get(row["branch"], row["branch"]),
                    "source": "E036",
                    "gamma": row["gamma"],
                    "radius": row["policy_latent_radius_rms"],
                    "objective_kind": "upstream_trust_region",
                    "qr_verify_exact_presets": row["qr_verify_exact_presets"],
                    "ssr": row["qr_verify_exact_presets"] / 37.0,
                    "full_module_error_count": row["full_module_error_count"],
                    "upstream_active_modules": row["upstream_active_modules"],
                    "lpips": row["lpips"],
                    "latent_delta_rms": row["latent_delta_rms"],
                    "historical": True,
                }
            )
    return rows


def _score_quality(images: dict[str, Image.Image], prompt: str, settings: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from .quality_scoring import quality_scorer_from_settings

    scorer = quality_scorer_from_settings(settings, device="cpu")
    results: dict[str, Any] = {}
    for name, image in images.items():
        score = scorer.score(image, prompt)
        results[name] = asdict(score)
    return results, scorer.provenance()


def _comparison_sheet(
    output_path: Path,
    items: list[tuple[str, Image.Image, str]],
    *,
    columns: int = 4,
) -> None:
    tile_w = 390
    tile_h = 450
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image, subtitle) in enumerate(items):
        row, col = divmod(index, columns)
        x0, y0 = col * tile_w, row * tile_h
        preview = image.convert("RGB").copy()
        preview.thumbnail((370, 350), Image.Resampling.LANCZOS)
        x = x0 + (tile_w - preview.width) // 2
        y = y0 + 78 + (350 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((x0 + 8, y0 + 8), label, fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 28), subtitle[:72], fill=(50, 50, 50))
        if len(subtitle) > 72:
            draw.text((x0 + 8, y0 + 46), subtitle[72:144], fill=(50, 50, 50))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False, compress_level=9)


def _qr_original_exact(item: dict[str, Any]) -> bool:
    """Best-effort extraction of the qr-verify original preset from canonical evidence."""

    for key in (
        "conservative_original_exact",
        "original_exact",
        "qr_verify_direct_exact",
        "original_strict_all",
    ):
        if key in item and item[key] is not None:
            return bool(item[key])
    for key in ("presets", "preset_results", "results"):
        values = item.get(key)
        if isinstance(values, list):
            matches = []
            for value in values:
                if not isinstance(value, dict):
                    continue
                name = str(value.get("preset") or value.get("name") or value.get("id") or "")
                if name != "original":
                    continue
                exact = value.get("exact_payload_match")
                if exact is None:
                    exact = value.get("exact")
                if exact is None:
                    exact = value.get("passed")
                if exact is not None:
                    matches.append(bool(exact))
            if matches:
                return all(matches)
    return False


def _visual_guard(
    row: dict[str, Any],
    parent_quality: dict[str, Any],
    config: E038Config,
) -> dict[str, Any]:
    hps = row.get("hpsv2_1")
    parent_hps = parent_quality.get("hpsv2_1")
    checks = {
        "lpips": float(row["lpips"]) <= config.max_lpips_for_ranking,
        "mean_absolute_change": (
            float(row["mean_absolute_change"]) <= config.max_mean_absolute_change
        ),
        "clipped_pixel_ratio_increase": (
            float(row["clipped_pixel_ratio_increase"])
            <= config.max_clipped_pixel_ratio_increase
        ),
        "rgb_clipped_channel_ratio_increase": (
            float(row["rgb_clipped_channel_ratio_increase"])
            <= config.max_rgb_clipped_channel_ratio_increase
        ),
        "saturation_mean_change": (
            abs(float(row["saturation_mean_increase"]))
            <= config.max_abs_saturation_mean_change
        ),
        "high_saturation_ratio_increase": (
            float(row["high_saturation_ratio_increase"])
            <= config.max_high_saturation_ratio_increase
        ),
        "clip_score": (
            float(row["clip_score"])
            >= float(parent_quality["clip_score"]) - config.max_clip_score_drop
        ),
        "clip_aesthetic": (
            float(row["clip_aesthetic"])
            >= float(parent_quality["clip_aesthetic"])
            - config.max_clip_aesthetic_drop
        ),
    }
    if hps is not None and parent_hps is not None:
        checks["hpsv2_1"] = float(hps) >= float(parent_hps) - config.max_hps_drop
    return {"passed": all(checks.values()), "checks": checks}


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "e038-artifact-manifest.json":
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def run_e038(
    *,
    parent_dir: Path,
    e035_results_dir: Path,
    e036_results_dir: Path,
    output_dir: Path,
    recipes: tuple[Recipe, ...] = DEFAULT_RECIPES,
    config: E038Config | None = None,
    expected_parent_commit: str | None = None,
    skip_qr_verify: bool = False,
    skip_quality: bool = False,
) -> dict[str, Any]:
    import torch

    from .quality import image_change_metrics, image_quality_metrics
    from .qr import generate_diffqrcoder_qr

    config = config or E038Config()
    if not torch.cuda.is_available():
        raise RuntimeError("E038 requires an available CUDA GPU")
    if config.gamma != 1000.0:
        raise ValueError("E038 keeps gamma fixed at 1000")
    if config.max_iterations != 4:
        raise ValueError("E038 is frozen to four updates")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E038 output directory must be empty: {output_dir}")

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
            "parent_contract_sha256": parent.metadata["contract_sha256"],
            "gamma_is_fixed": True,
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

    backend, pipeline = _load_pipeline()
    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(
        getattr(pipeline.vae, "is_gradient_checkpointing", False)
    )
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

    # Canonical FP32 parent comes from any E038 iteration zero.
    first_result = next(iter(results.values()))
    parent_fp32_path = Path(first_result.output_dir) / "images/iteration-000.png"
    parent_fp32 = Image.open(parent_fp32_path).convert("RGB")
    parent_fp32.save(
        output_dir / "parent-fp32-redecoded.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )

    historical_images = _historical_images(e035_results_dir, e036_results_dir)
    all_images: dict[str, Image.Image] = {"parent_fp32": parent_fp32, **historical_images}
    for name, result in results.items():
        all_images[name] = Image.open(result.final_image_path).convert("RGB")

    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, payload, all_images)
    if skip_qr_verify:
        _atomic_json(output_dir / "qr-verify-evidence.json", {"skipped": True})

    quality_scores: dict[str, Any] = {}
    quality_provenance: dict[str, Any] = {"skipped": True}
    if not skip_quality:
        quality_scores, quality_provenance = _score_quality(
            all_images,
            prompt,
            backend.settings,
        )
    _atomic_json(output_dir / "quality-scores.json", quality_scores)
    _atomic_json(output_dir / "quality-provenance.json", quality_provenance)

    historical_rows = _historical_rows(e035_results_dir, e036_results_dir)
    _atomic_json(output_dir / "historical-controls.json", historical_rows)

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
            "source": "E038",
            "gamma": config.gamma,
            "radius": recipe.latent_radius_rms,
            "objective_kind": recipe.objective_kind,
            "full_module_weight": recipe.full_module_weight,
            "robust_blur_weight": recipe.robust_blur_weight,
            "robust_downscale_weight": recipe.robust_downscale_weight,
            "robust_brightness_weight": recipe.robust_brightness_weight,
            "robust_contrast_weight": recipe.robust_contrast_weight,
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
                and row["clipped_pixel_ratio_increase"]
                <= config.max_clipped_pixel_ratio_increase
                and row["rgb_clipped_channel_ratio_increase"]
                <= config.max_rgb_clipped_channel_ratio_increase
            ),
            "checks": {},
        }
        row["visual_guard_pass"] = guard["passed"]
        row["visual_guard_checks"] = guard["checks"]
        rows.append(row)

    # Ranking is lexicographic and explicitly SSR-first after the visual gate.
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

    all_rows = historical_rows + rows
    _atomic_json(output_dir / "method-comparison.json", all_rows)
    with (output_dir / "method-comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = sorted({key for row in all_rows for key in row if key != "visual_guard_checks"})
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    items: list[tuple[str, Image.Image, str]] = []
    historical_label_map = {
        "parent_fp32": "Parent FP32",
        "e035_paper": "E035 paper",
        "e035_upstream_unbounded": "E035 upstream libre",
        "e036_global_r050": "E036 global r=.05",
        "e036_strict_r025": "E036 strict r=.025",
        "e036_local_r050": "E036 local r=.05",
    }
    row_by_method = {row["method"]: row for row in all_rows}
    for key in (
        "parent_fp32",
        "e035_paper",
        "e035_upstream_unbounded",
        "e036_global_r050",
        "e036_strict_r025",
        "e036_local_r050",
    ):
        image = all_images.get(key)
        if image is None:
            continue
        if key == "parent_fp32":
            subtitle = "reference"
        else:
            hrow = row_by_method.get(key) or {}
            exact = hrow.get("qr_verify_exact_presets", "?")
            lpips = hrow.get("lpips")
            subtitle = f"SSR={exact}/37  LPIPS={lpips:.4f}" if isinstance(lpips, (int, float)) else f"SSR={exact}/37"
        items.append((historical_label_map[key], image, subtitle))

    for row in sorted(rows, key=lambda value: (float(value["radius"]), value["method"])):
        items.append(
            (
                row["method"].replace("e038_", "E038 "),
                all_images[row["method"]],
                (
                    f"SSR={row['qr_verify_exact_presets']}/37  MER={row['full_module_error_count']}/841  "
                    f"LPIPS={row['lpips']:.4f}  safe={row['visual_guard_pass']}"
                ),
            )
        )
    _comparison_sheet(output_dir / "e038-all-methods-contact-sheet.png", items)

    verdict = {
        "experiment": EXPERIMENT,
        "gamma": config.gamma,
        "gamma_preserved": True,
        "recipe_count": len(recipes),
        "visual_safe_recipe_count": len(safe),
        "research_winner": winner,
        "winner_ssr_exact_presets": (
            None if not ranked else ranked[0]["qr_verify_exact_presets"]
        ),
        "winner_ssr": None if not ranked else ranked[0]["ssr"],
        "winner_original_exact": None if not ranked else ranked[0]["original_exact"],
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": (
            "REVIEW_WINNER_AND_NEIGHBOR_RECIPES_VISUALLY"
            if winner is not None
            else "NO_VISUALLY_SAFE_RECIPE_FOUND"
        ),
    }
    _atomic_json(output_dir / "verdict.json", verdict)

    report = f"""# E038 — SR-MPGD SSR/aesthetic frontier search

- Same frozen parent as E035/E036.
- Gamma fixed to **1000** for every recipe.
- New recipes: **{len(recipes)}**.
- Ranking: visual guard first, then QR-Verify SSR, direct/original decode, MER, LPIPS.
- Research winner: **{winner or 'none'}**.
- Production ready: **no**.

The contact sheet compares the parent, E035 paper/upstream, E036 trust-region controls and every
new E038 final. E033/E034 remain numeric historical references because their archived parent
provenance is not silently equated to the current frozen parent.
"""
    _atomic_text(output_dir / "report.md", report)

    _atomic_json(output_dir / "e038-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--e035-results-dir", type=Path, required=True)
    parser.add_argument("--e036-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-commit", default=None)
    parser.add_argument("--skip-qr-verify", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    verdict = run_e038(
        parent_dir=args.parent_dir,
        e035_results_dir=args.e035_results_dir,
        e036_results_dir=args.e036_results_dir,
        output_dir=args.output_dir,
        expected_parent_commit=args.expected_parent_commit,
        skip_qr_verify=args.skip_qr_verify,
        skip_quality=args.skip_quality,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
