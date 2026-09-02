"""E043 — scanner-cell SR-MPGD frontier derived directly from E042.

E042 localized the primary blocker as GRID_DETECTION_OR_INTRA_MODULE_TEXTURE:
seven of nine artistic states decode after target-assisted grid reconstruction,
while quiet-zone-only and generic binarization rescues are zero. E043 therefore
keeps the E041 prompt/parent and gamma=500 paired, fixes only the exact
DiffQRCoder quiet-zone raster geometry, and ablates scanner-cell losses:

A. paired E041 gamma=500 control (existing latents, VAE re-decode only);
B. whole-cell margin + intra-module variance;
C. B + sub-cell/grid consistency;
D. C + format-information and ECC-risk-weighted data margins.

The ECC term is explicitly a data-module risk proxy, not a differentiable
Reed-Solomon decoder. QR-Verify remains authoritative. Production and
generalization stay forbidden.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .e035_loss_fidelity import (
    _atomic_json,
    _atomic_text,
    _core_blueprint,
    _crop_core,
    _cuda_snapshot,
    _decode_latent_tensor,
    _gradient_scales,
    _image_sha256,
    _load_lpips,
    _load_official_upstream_srl,
    _load_pipeline,
    _offload_diffusion_modules,
)
from .e035_losses import (
    module_diagnostics,
    prepare_upstream_torch_layout,
    upstream_qrcode_tensor,
)
from .e035_parent_artifact import LoadedParentArtifact, sha256_file, tensor_sha256
from .e036_trust_region import project_latent_candidate
from .e038_recipe_frontier import _comparison_sheet
from .e039_limiter_scanaware import (
    E039Config,
    Recipe as E039Recipe,
    _qr_objective as e039_qr_objective,
    _rms,
)
from .e041_gamma_functional_frontier import (
    ERROR_CORRECTION,
    PAYLOAD,
    PROMPT,
    QR_MASK_PATTERN,
    QR_MODULE_SIZE,
    QR_PADDING_PX,
    QR_VERSION,
    _score_rows,
)
from .e042_decoder_failure_localization import (
    EXPERIMENT as E042_EXPERIMENT,
    QR_CANVAS_PX,
    QR_CORE_MODULES,
    QR_CORE_PX,
    _exact_adaptive_quiet_color,
    _margin_metrics,
    _region_error_metrics,
    _region_masks,
    _restore_exact_quiet_zone,
    _sample_modules,
)

EXPERIMENT = "e043-scanner-cell-frontier-v1"
E041_EXPERIMENT = "e041-gamma-functional-pattern-frontier-v1"
GAMMA = 500.0
LATENT_RADIUS_RMS = 0.200
MAX_ITERATIONS = 8
CHECKPOINTS_PER_RECIPE = MAX_ITERATIONS + 1
RECIPE_COUNT = 4
EXPECTED_CHECKPOINT_COUNT = RECIPE_COUNT * CHECKPOINTS_PER_RECIPE

CONTROL_RECIPE = "e043_A_control_e041_g500"
E041_CONTROL_SOURCE_RECIPE = "e041_gamma_0500_r200_i08"


@dataclass(frozen=True, slots=True)
class ScannerCellRecipe:
    name: str
    whole_cell_weight: float = 0.0
    variance_weight: float = 0.0
    grid_weight: float = 0.0
    format_weight: float = 0.0
    data_ecc_risk_weight: float = 0.0
    gamma: float = GAMMA
    latent_radius_rms: float = LATENT_RADIUS_RMS
    max_iterations: int = MAX_ITERATIONS


RECIPES: tuple[ScannerCellRecipe, ...] = (
    ScannerCellRecipe(CONTROL_RECIPE),
    ScannerCellRecipe(
        "e043_B_cellvar_g500",
        whole_cell_weight=0.20,
        variance_weight=0.06,
    ),
    ScannerCellRecipe(
        "e043_C_grid_g500",
        whole_cell_weight=0.20,
        variance_weight=0.06,
        grid_weight=0.15,
    ),
    ScannerCellRecipe(
        "e043_D_critical_g500",
        whole_cell_weight=0.20,
        variance_weight=0.06,
        grid_weight=0.15,
        format_weight=0.25,
        data_ecc_risk_weight=0.12,
    ),
)


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    objective: float
    lpips: float
    core_mae: float
    latent_delta_rms: float
    checks: dict[str, bool]


@dataclass(frozen=True, slots=True)
class Checkpoint:
    recipe: str
    iteration: int
    image_path: str
    latent_path: str
    trace_step: dict[str, Any]


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


def _load_required_verdicts(e041_results_dir: Path, e042_results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    e041_path = e041_results_dir / "verdict.json"
    e042_path = e042_results_dir / "verdict.json"
    if not e041_path.is_file():
        raise FileNotFoundError(f"E041 verdict missing: {e041_path}")
    if not e042_path.is_file():
        raise FileNotFoundError(f"E042 verdict missing: {e042_path}")
    e041 = json.loads(e041_path.read_text(encoding="utf-8"))
    e042 = json.loads(e042_path.read_text(encoding="utf-8"))
    if e041.get("experiment") != E041_EXPERIMENT or int(e041.get("phase_a_checkpoint_count", 0)) != 54:
        raise RuntimeError("E043 requires finalized E041 with 54 Phase-A checkpoints")
    if e042.get("experiment") != E042_EXPERIMENT:
        raise RuntimeError("E043 requires finalized E042")
    if e042.get("primary_blocker") != "GRID_DETECTION_OR_INTRA_MODULE_TEXTURE":
        raise RuntimeError(f"unexpected E042 blocker: {e042.get('primary_blocker')}")
    if int(e042.get("grid_reconstruction_rescue_count", 0)) < 1:
        raise RuntimeError("E043 requires E042 grid-reconstruction rescue evidence")
    if e041.get("production_ready") is not False or e042.get("production_ready") is not False:
        raise RuntimeError("historical experiments must remain non-production")
    if e041.get("generalization_authorized") is not False or e042.get("generalization_authorized") is not False:
        raise RuntimeError("historical experiments must remain non-generalized")
    return e041, e042


def _load_parent(e041_results_dir: Path) -> LoadedParentArtifact:
    from safetensors.torch import load_file

    root = e041_results_dir / "parent"
    image_path = root / "stage2.png"
    latent_path = root / "stage2-latent.safetensors"
    metadata_path = root / "parent-metadata.json"
    for path in (image_path, latent_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"E041 parent artifact missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("payload") != PAYLOAD or metadata.get("prompt") != PROMPT:
        raise RuntimeError("E041 parent payload/prompt mismatch")
    latent = load_file(str(latent_path), device="cpu")["latent"].detach().cpu().contiguous()
    return LoadedParentArtifact(
        root=root,
        image=Image.open(image_path).convert("RGB"),
        latent=latent,
        metadata={"source": metadata},
    )


def _core_target_matrix(blueprint: Any) -> np.ndarray:
    border = int(blueprint.border)
    matrix = blueprint.matrix[border:-border, border:-border] if border else blueprint.matrix
    matrix = np.asarray(matrix, dtype=bool)
    if matrix.shape != (QR_CORE_MODULES, QR_CORE_MODULES):
        raise ValueError(f"E043 expects 29x29 core, got {matrix.shape}")
    return matrix


def _decoded_to_exact_scan_ready(decoded: Any) -> Image.Image:
    unit = (decoded.detach().float() / 2 + 0.5).clamp(0, 1)
    array = unit[0].permute(1, 2, 0).cpu().numpy()
    raw = Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")
    if raw.size != (QR_CANVAS_PX, QR_CANVAS_PX):
        raise ValueError(f"E043 expects {QR_CANVAS_PX}x{QR_CANVAS_PX}, got {raw.size}")
    color = _exact_adaptive_quiet_color(raw, minimum_luminance=0.90)
    return _restore_exact_quiet_zone(raw, color)


def _module_cells(unit: Any) -> Any:
    if unit.ndim != 4 or unit.shape[1] != 3:
        raise ValueError("unit image must be BCHW RGB")
    if tuple(unit.shape[-2:]) != (QR_CORE_PX, QR_CORE_PX):
        raise ValueError(f"E043 core must be {QR_CORE_PX}x{QR_CORE_PX}")
    coefficients = unit.new_tensor((0.2999, 0.5870, 0.1114)).view(1, 3, 1, 1)
    gray = (unit * coefficients).sum(dim=1)
    batch = gray.shape[0]
    return (
        gray.reshape(batch, QR_CORE_MODULES, QR_MODULE_SIZE, QR_CORE_MODULES, QR_MODULE_SIZE)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


def _target_tensor(target_dark_np: np.ndarray, reference: Any) -> Any:
    import torch
    return torch.as_tensor(target_dark_np, device=reference.device, dtype=torch.bool).unsqueeze(0)


def _margin_violation(values: Any, target_dark: Any, dark_threshold: float, light_threshold: float) -> Any:
    import torch
    if not 0 < dark_threshold < light_threshold < 1:
        raise ValueError("invalid thresholds")
    target = target_dark
    while target.ndim < values.ndim:
        target = target.unsqueeze(-1)
    return torch.where(
        target,
        torch.relu(values - dark_threshold),
        torch.relu(light_threshold - values),
    )


def whole_cell_margin_loss(unit: Any, target_dark_np: np.ndarray) -> Any:
    """Penalize both average and worst-quarter pixels inside every module."""
    cells = _module_cells(unit)
    target = _target_tensor(target_dark_np, cells)
    violations = _margin_violation(cells, target, 0.48, 0.52)
    flat = violations.flatten(start_dim=-2)
    k = max(1, flat.shape[-1] // 4)
    worst = flat.topk(k, dim=-1, largest=True, sorted=False).values.mean(dim=-1)
    mean = flat.mean(dim=-1)
    return (0.35 * mean + 0.65 * worst).mean()


def intra_module_variance_penalty(unit: Any, target_dark_np: np.ndarray) -> Any:
    """Suppress excessive texture mainly in ambiguous/at-risk modules."""
    import torch
    cells = _module_cells(unit)
    means = cells.mean(dim=(-1, -2))
    stds = cells.std(dim=(-1, -2), unbiased=False)
    target = _target_tensor(target_dark_np, means)
    margin = torch.where(target, 0.5 - means, means - 0.5)
    risk = torch.sigmoid((0.10 - margin) / 0.03)
    excess = torch.relu(stds - 0.14).square()
    return (excess * (0.50 + risk)).mean()


def grid_consistency_loss(unit: Any, target_dark_np: np.ndarray) -> Any:
    """Apply the module target to quadrants and inner-edge strips, not a visible grid."""
    import torch
    cells = _module_cells(unit)
    target = _target_tensor(target_dark_np, cells[..., 0, 0])
    q = QR_MODULE_SIZE // 2
    edge = 4
    regions = [
        cells[..., :q, :q].mean(dim=(-1, -2)),
        cells[..., :q, q:].mean(dim=(-1, -2)),
        cells[..., q:, :q].mean(dim=(-1, -2)),
        cells[..., q:, q:].mean(dim=(-1, -2)),
        cells[..., :edge, :].mean(dim=(-1, -2)),
        cells[..., -edge:, :].mean(dim=(-1, -2)),
        cells[..., :, :edge].mean(dim=(-1, -2)),
        cells[..., :, -edge:].mean(dim=(-1, -2)),
    ]
    stacked = torch.stack(regions, dim=-1)
    violation = _margin_violation(stacked, target, 0.47, 0.53).mean()
    coherence = torch.relu(stacked.std(dim=-1, unbiased=False) - 0.08).square().mean()
    return violation + 0.5 * coherence


def critical_module_losses(unit: Any, target_dark_np: np.ndarray, masks_np: dict[str, np.ndarray]) -> tuple[Any, Any]:
    """Return format loss and ECC-risk-weighted data loss (proxy, not RS decoding)."""
    import torch
    cells = _module_cells(unit)
    means = cells.mean(dim=(-1, -2))
    target = _target_tensor(target_dark_np, means)
    violation = _margin_violation(means, target, 0.44, 0.56)
    format_mask = torch.as_tensor(masks_np["format"], device=means.device, dtype=torch.bool).unsqueeze(0)
    data_mask = torch.as_tensor(masks_np["data"], device=means.device, dtype=torch.bool).unsqueeze(0)
    format_loss = violation[format_mask.expand_as(violation)].mean()
    margin = torch.where(target, 0.5 - means, means - 0.5)
    risk_multiplier = 1.0 + 2.0 * torch.sigmoid((0.08 - margin) / 0.025)
    data_loss = (violation * risk_multiplier)[data_mask.expand_as(violation)].mean()
    return format_loss, data_loss


def _base_e039_recipe(recipe: ScannerCellRecipe) -> E039Recipe:
    return E039Recipe(
        name=recipe.name,
        profile="scanaware_v2",
        max_iterations=recipe.max_iterations,
        latent_radius_rms=recipe.latent_radius_rms,
        lpips_budget=0.050,
        core_mae_budget=0.050,
        full_module_weight=0.10,
    )


def _scanner_cell_objective(
    unit: Any,
    *,
    recipe: ScannerCellRecipe,
    core_blueprint: Any,
    upstream_layout: Any,
    upstream_target: Any,
    official_upstream_srl: Any,
    config: E039Config,
    iteration: int,
    phase: str,
    target_dark_np: np.ndarray,
    masks_np: dict[str, np.ndarray],
) -> tuple[Any, dict[str, float]]:
    base, base_parts = e039_qr_objective(
        unit,
        recipe=_base_e039_recipe(recipe),
        core_blueprint=core_blueprint,
        upstream_layout=upstream_layout,
        upstream_target=upstream_target,
        official_upstream_srl=official_upstream_srl,
        config=config,
        iteration=iteration,
        phase=phase,
    )
    whole = whole_cell_margin_loss(unit, target_dark_np) if recipe.whole_cell_weight else unit.new_tensor(0.0)
    variance = intra_module_variance_penalty(unit, target_dark_np) if recipe.variance_weight else unit.new_tensor(0.0)
    grid = grid_consistency_loss(unit, target_dark_np) if recipe.grid_weight else unit.new_tensor(0.0)
    fmt, data = (
        critical_module_losses(unit, target_dark_np, masks_np)
        if recipe.format_weight or recipe.data_ecc_risk_weight
        else (unit.new_tensor(0.0), unit.new_tensor(0.0))
    )
    total = (
        base
        + recipe.whole_cell_weight * whole
        + recipe.variance_weight * variance
        + recipe.grid_weight * grid
        + recipe.format_weight * fmt
        + recipe.data_ecc_risk_weight * data
    )
    return total, {
        **base_parts,
        "whole_cell_loss": float(whole.detach().cpu()),
        "variance_loss": float(variance.detach().cpu()),
        "grid_loss": float(grid.detach().cpu()),
        "format_loss": float(fmt.detach().cpu()),
        "data_ecc_risk_loss": float(data.detach().cpu()),
    }


def _core_mae(current: Any, reference: Any) -> float:
    return float(((current - reference).abs().mean() / 2).detach().cpu())


def _candidate_metrics(
    *, pipeline: Any, candidate: Any, initial: Any, reference_core: Any,
    reference_lpips: Any, lpips_model: Any, lpips_dtype: Any,
    core_blueprint: Any, upstream_layout: Any, upstream_target: Any,
    official_upstream_srl: Any, recipe: ScannerCellRecipe,
    current_objective: float, config: E039Config,
    target_dark_np: np.ndarray, masks_np: dict[str, np.ndarray],
) -> CandidateMetrics:
    import torch
    with torch.no_grad():
        decoded = _decode_latent_tensor(pipeline, candidate).float()
        core = _crop_core(decoded, config.crop_padding_px).detach()
        unit = (core / 2 + 0.5).clamp(0, 1)
        qr_tensor, _ = _scanner_cell_objective(
            unit, recipe=recipe, core_blueprint=core_blueprint,
            upstream_layout=upstream_layout, upstream_target=upstream_target,
            official_upstream_srl=official_upstream_srl, config=config,
            iteration=-1, phase="candidate", target_dark_np=target_dark_np,
            masks_np=masks_np,
        )
        lpips = float(lpips_model(core.to(device="cpu", dtype=lpips_dtype), reference_lpips).mean().detach().cpu())
        objective = float(qr_tensor.detach().cpu()) + config.lpips_weight * lpips
        core_mae = _core_mae(core, reference_core)
        latent_delta = float(_rms(candidate - initial).detach().cpu())
    checks = {
        "latent_radius": latent_delta <= recipe.latent_radius_rms + 1e-9,
        "lpips_budget": lpips <= 0.050 + 1e-9,
        "core_mae_budget": core_mae <= 0.050 + 1e-9,
        "objective_nonincrease": objective <= current_objective + config.objective_nonincrease_tolerance,
    }
    return CandidateMetrics(objective, lpips, core_mae, latent_delta, checks)


def _run_new_trajectory(
    *, pipeline: Any, parent: LoadedParentArtifact, blueprint: Any,
    recipe: ScannerCellRecipe, config: E039Config, output_root: Path,
) -> list[Checkpoint]:
    import torch
    from safetensors.torch import save_file
    from .guidance import prepare_torch_layout

    if recipe.name == CONTROL_RECIPE:
        raise ValueError("control trajectory must be reused from E041, not optimized")
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
    core_h, core_w = reference_core.shape[-2:]
    if (core_h, core_w) != (QR_CORE_PX, QR_CORE_PX):
        raise ValueError(f"E043 expects core {QR_CORE_PX}, got {(core_h, core_w)}")
    core_blueprint = _core_blueprint(blueprint, core_w, core_h)
    paper_layout = prepare_torch_layout(core_blueprint, core_h, core_w, device=device, dtype=torch.float32, center_fraction=1/3)
    upstream_layout = prepare_upstream_torch_layout(core_blueprint, core_h, core_w, device=device, dtype=torch.float32)
    upstream_target = upstream_qrcode_tensor(core_blueprint, core_h, core_w, device=device, dtype=torch.float32)
    official_upstream_srl = _load_official_upstream_srl(config.qr_module_size, device=device)
    target_dark_np = _core_target_matrix(blueprint)
    masks_np = _region_masks(blueprint)

    lpips_model = _load_lpips(pipeline, net=config.lpips_net)
    lpips_parameter = next(iter(lpips_model.parameters()), None)
    lpips_dtype = lpips_parameter.dtype if lpips_parameter is not None else torch.float32
    reference_lpips = reference_core.to(device="cpu", dtype=lpips_dtype).detach()

    checkpoints: list[Checkpoint] = []
    trace_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for iteration in range(recipe.max_iterations + 1):
        working = working.detach()
        with torch.no_grad():
            decoded = reference_decoded if iteration == 0 else _decode_latent_tensor(pipeline, working).float()
            core = _crop_core(decoded, config.crop_padding_px).detach()
            unit = (core / 2 + 0.5).clamp(0, 1)
            qr_tensor, parts = _scanner_cell_objective(
                unit, recipe=recipe, core_blueprint=core_blueprint,
                upstream_layout=upstream_layout, upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl, config=config,
                iteration=iteration, phase="evaluation", target_dark_np=target_dark_np,
                masks_np=masks_np,
            )
            lpips_loss = float(lpips_model(core.to(device="cpu", dtype=lpips_dtype), reference_lpips).mean().detach().cpu())
            qr_value = float(qr_tensor.detach().cpu())
            objective = qr_value + config.lpips_weight * lpips_loss
            diagnostics = module_diagnostics(unit, core_blueprint, paper_layout=paper_layout, upstream_layout=upstream_layout)

        image = _decoded_to_exact_scan_ready(decoded)
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

        if iteration < recipe.max_iterations and qr_value > 0.0:
            objective_core = core.detach().requires_grad_(True)
            objective_unit = (objective_core / 2 + 0.5).clamp(0, 1)
            qr_loss, _ = _scanner_cell_objective(
                objective_unit, recipe=recipe, core_blueprint=core_blueprint,
                upstream_layout=upstream_layout, upstream_target=upstream_target,
                official_upstream_srl=official_upstream_srl, config=config,
                iteration=iteration, phase="gradient", target_dark_np=target_dark_np,
                masks_np=masks_np,
            )
            qr_gradient = torch.autograd.grad(qr_loss, objective_core, only_inputs=True)[0]
            lpips_core = core.detach().to(device="cpu", dtype=lpips_dtype).requires_grad_(True)
            lpips_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(lpips_tensor, lpips_core, only_inputs=True)[0]
            objective_gradient = qr_gradient.clone()
            objective_gradient.add_(lpips_gradient_cpu.to(device=device, dtype=objective_gradient.dtype), alpha=config.lpips_weight)

            gradient = None
            if torch.isfinite(objective_gradient).all():
                for scale in _gradient_scales(config.gradient_scale):
                    candidate = working.detach().requires_grad_(True)
                    candidate_decoded = _decode_latent_tensor(pipeline, candidate).float()
                    candidate_core = _crop_core(candidate_decoded, config.crop_padding_px)
                    candidate_gradient = torch.autograd.grad(
                        candidate_core, candidate,
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
                last_metrics: CandidateMetrics | None = None
                for backtrack_index in range(config.max_backtracks + 1):
                    if alpha < config.minimum_alpha:
                        break
                    trial = (working + alpha * direction).detach()
                    metrics = _candidate_metrics(
                        pipeline=pipeline, candidate=trial, initial=initial,
                        reference_core=reference_core, reference_lpips=reference_lpips,
                        lpips_model=lpips_model, lpips_dtype=lpips_dtype,
                        core_blueprint=core_blueprint, upstream_layout=upstream_layout,
                        upstream_target=upstream_target, official_upstream_srl=official_upstream_srl,
                        recipe=recipe, current_objective=objective, config=config,
                        target_dark_np=target_dark_np, masks_np=masks_np,
                    )
                    last_metrics = metrics
                    failed = [key for key, passed in metrics.checks.items() if not passed]
                    rejection_rows.append({
                        "recipe": recipe.name, "iteration": iteration,
                        "backtrack_index": backtrack_index, "alpha": alpha,
                        "accepted": all(metrics.checks.values()),
                        "rejection_reasons": ";".join(failed),
                        "objective": metrics.objective, "lpips": metrics.lpips,
                        "core_mae": metrics.core_mae,
                        "latent_delta_rms": metrics.latent_delta_rms,
                        **{f"check_{key}": passed for key, passed in metrics.checks.items()},
                    })
                    if all(metrics.checks.values()):
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

        latent_delta_rms = float(_rms(working - initial).detach().cpu())
        step = {
            "recipe": recipe.name, "iteration": iteration,
            "elapsed_s": time.perf_counter() - started,
            "image_sha256": _image_sha256(image),
            "latent_sha256": tensor_sha256(working),
            **parts,
            "lpips_loss": lpips_loss, "objective": objective,
            "upstream_active_modules": int(diagnostics["upstream_margin_active_count"]),
            "full_module_error_count": int(diagnostics["full_module_error_count"]),
            "full_module_error_rate": float(diagnostics["full_module_error_rate"]),
            "gamma": config.gamma, "latent_gradient_rms": latent_gradient_rms,
            "raw_step_rms": raw_step_rms, "projected_step_rms": projected_step_rms,
            "accepted_step_rms": accepted_step_rms, "accepted_alpha": accepted_alpha,
            "latent_delta_rms": latent_delta_rms,
            "effective_gradient_scale": effective_gradient_scale,
            "acceptance_reason": acceptance_reason,
            "candidate_checks": candidate_checks,
            "rejected_trial_count": rejected_this_iteration,
            "cuda": _cuda_snapshot(),
        }
        trace_rows.append(step)
        checkpoints.append(Checkpoint(recipe.name, iteration, str(image_path), str(latent_path), step))
        working = next_working.detach()
        gc.collect()
        torch.cuda.empty_cache()

    _atomic_json(root / "trace.json", trace_rows)
    _atomic_json(root / "rejection-log.json", rejection_rows)
    _atomic_json(root / "recipe.json", asdict(recipe))
    flat_trace = []
    for row in trace_rows:
        current = {k: v for k, v in row.items() if k not in {"candidate_checks", "cuda"}}
        current.update({f"check_{k}": v for k, v in (row.get("candidate_checks") or {}).items()})
        current.update({f"cuda_{k}": v for k, v in (row.get("cuda") or {}).items()})
        flat_trace.append(current)
    _write_csv(root / "trace.csv", flat_trace)
    _write_csv(root / "rejection-log.csv", rejection_rows)
    return checkpoints


def _reuse_control(*, pipeline: Any, e041_results_dir: Path, output_root: Path) -> list[Checkpoint]:
    """Re-decode existing E041 gamma=500 latents with exact 736/78 geometry."""
    import torch
    from safetensors.torch import load_file

    source_root = e041_results_dir / "phase-a-trajectories" / E041_CONTROL_SOURCE_RECIPE
    trace_path = source_root / "trace.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"E041 gamma500 trace missing: {trace_path}")
    source_trace = {int(row["iteration"]): row for row in json.loads(trace_path.read_text(encoding="utf-8"))}
    root = output_root / CONTROL_RECIPE
    images_root = root / "images"
    latents_root = root / "latents"
    images_root.mkdir(parents=True, exist_ok=True)
    latents_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    checkpoints: list[Checkpoint] = []
    for iteration in range(CHECKPOINTS_PER_RECIPE):
        source_latent = source_root / "latents" / f"iteration-{iteration:03d}.safetensors"
        if not source_latent.is_file():
            raise FileNotFoundError(f"E041 gamma500 latent missing: {source_latent}")
        latent_cpu = load_file(str(source_latent), device="cpu")["latent"].detach().cpu().contiguous()
        latent = latent_cpu.to(device="cuda", dtype=torch.float32)
        with torch.no_grad():
            decoded = _decode_latent_tensor(pipeline, latent).float()
        image = _decoded_to_exact_scan_ready(decoded)
        image_path = images_root / f"iteration-{iteration:03d}.png"
        latent_path = latents_root / f"iteration-{iteration:03d}.safetensors"
        image.save(image_path, format="PNG", optimize=False, compress_level=9)
        shutil.copy2(source_latent, latent_path)
        source_step = dict(source_trace.get(iteration) or {})
        step = {
            "recipe": CONTROL_RECIPE, "iteration": iteration,
            "source_recipe": E041_CONTROL_SOURCE_RECIPE,
            "source_latent_sha256": tensor_sha256(latent_cpu),
            "image_sha256": _image_sha256(image), "gamma": GAMMA,
            "latent_delta_rms": source_step.get("latent_delta_rms"),
            "accepted_alpha": source_step.get("accepted_alpha"),
            "accepted_step_rms": source_step.get("accepted_step_rms"),
            "raw_step_rms": source_step.get("raw_step_rms"),
            "projected_step_rms": source_step.get("projected_step_rms"),
            "reused_e041_latent": True,
            "optimization_performed_by_e043": False,
        }
        rows.append(step)
        checkpoints.append(Checkpoint(CONTROL_RECIPE, iteration, str(image_path), str(latent_path), step))
        del latent, latent_cpu, decoded
        gc.collect()
        torch.cuda.empty_cache()
    _atomic_json(root / "trace.json", rows)
    _write_csv(root / "trace.csv", rows)
    _atomic_json(root / "recipe.json", asdict(RECIPES[0]) | {
        "paired_control": True, "source_recipe": E041_CONTROL_SOURCE_RECIPE,
        "new_srmpgd_updates": 0,
    })
    return checkpoints


def _scanner_diagnostics(image: Image.Image, blueprint: Any) -> dict[str, Any]:
    target = _core_target_matrix(blueprint)
    masks = _region_masks(blueprint)
    means, centers, stds = _sample_modules(image, center_fraction=0.40)
    mean_pred = means < 0.5
    center_pred = centers < 0.5
    output = {
        "intra_module_std_mean": float(stds.mean()),
        "intra_module_std_p90": float(np.quantile(stds, 0.90)),
        "intra_module_std_max": float(stds.max()),
    }
    output.update({f"mean_{k}": v for k, v in _region_error_metrics(mean_pred, target, masks).items()})
    output.update({f"center_{k}": v for k, v in _region_error_metrics(center_pred, target, masks).items()})
    output.update({f"mean_{k}": v for k, v in _margin_metrics(means, target, masks).items()})
    return output


def _rank_key(row: dict[str, Any], recipe_order: dict[str, int]) -> tuple[Any, ...]:
    return (
        -int(row.get("qr_verify_exact_presets", 0)),
        -int(bool(row.get("original_exact"))),
        int(row.get("mean_format_error_count", 999)),
        int(row.get("mean_data_error_count", 999)),
        int(row.get("mean_total_error_count", 999)),
        float(row.get("intra_module_std_p90", 9.0)),
        float(row.get("lpips", 9.0)),
        int(row.get("iteration", 999)),
        recipe_order.get(str(row.get("recipe")), 999),
    )


def _manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "e043-artifact-manifest.json":
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return rows


def run_e043(*, output_dir: Path, e041_results_dir: Path, e042_results_dir: Path, source_commit: str) -> dict[str, Any]:
    import torch
    from .qr import generate_diffqrcoder_qr

    if not torch.cuda.is_available():
        raise RuntimeError("E043 requires CUDA")
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("source_commit must be a lowercase 40-character Git SHA")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E043 output must be empty: {output_dir}")
    if len(RECIPES) != RECIPE_COUNT or RECIPES[0].name != CONTROL_RECIPE:
        raise RuntimeError("E043 recipe contract changed unexpectedly")
    if any(recipe.gamma != GAMMA for recipe in RECIPES):
        raise RuntimeError("E043 must keep all paired recipes at gamma=500")

    e041_verdict, e042_verdict = _load_required_verdicts(e041_results_dir, e042_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "e041-verdict.json", e041_verdict)
    _atomic_json(output_dir / "e042-verdict.json", e042_verdict)
    _atomic_json(output_dir / "plan.json", {
        "experiment": EXPERIMENT, "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit, "payload": PAYLOAD, "prompt": PROMPT,
        "paired_prompt_parent_with_e041": True,
        "e042_primary_blocker": e042_verdict["primary_blocker"],
        "gamma": GAMMA, "latent_radius_rms": LATENT_RADIUS_RMS,
        "max_iterations": MAX_ITERATIONS, "recipes": [asdict(recipe) for recipe in RECIPES],
        "expected_checkpoint_count": EXPECTED_CHECKPOINT_COUNT,
        "control_reuses_e041_latents": True,
        "new_srmpgd_update_budget": 3 * MAX_ITERATIONS,
        "quiet_zone_geometry": {
            "canvas_px": QR_CANVAS_PX, "padding_px": QR_PADDING_PX,
            "core_px": QR_CORE_PX, "core_modules": QR_CORE_MODULES,
            "module_px": QR_MODULE_SIZE, "legacy_core_overwrite_forbidden": True,
        },
        "ecc_awareness": "risk-weighted data-module margin proxy; not differentiable Reed-Solomon",
        "production_ready": False, "generalization_authorized": False,
    })

    parent = _load_parent(e041_results_dir)
    blueprint = generate_diffqrcoder_qr(
        PAYLOAD, ERROR_CORRECTION, version=QR_VERSION,
        mask_pattern=QR_MASK_PATTERN, module_size=QR_MODULE_SIZE,
    )
    backend, pipeline = _load_pipeline()
    config = E039Config(
        gamma=GAMMA, crop_padding_px=QR_PADDING_PX,
        qr_version=QR_VERSION, qr_mask_pattern=QR_MASK_PATTERN,
        qr_module_size=QR_MODULE_SIZE, quiet_zone_mode="adaptive_light",
        functional_pattern_tone_factor=0.0,
    )

    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
    all_checkpoints: list[Checkpoint] = []
    try:
        with _offload_diffusion_modules(pipeline) as offloaded:
            if not checkpointing_was_enabled and callable(enable_checkpointing):
                enable_checkpointing()
            pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
            _atomic_json(output_dir / "runtime.json", {
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "offloaded_modules": list(offloaded),
                "vae_original_dtype": str(original_vae_dtype),
                "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                "gamma": GAMMA,
            })
            all_checkpoints.extend(_reuse_control(
                pipeline=pipeline, e041_results_dir=e041_results_dir,
                output_root=output_dir / "trajectories",
            ))
            for recipe in RECIPES[1:]:
                all_checkpoints.extend(_run_new_trajectory(
                    pipeline=pipeline, parent=parent, blueprint=blueprint,
                    recipe=recipe, config=config,
                    output_root=output_dir / "trajectories",
                ))
                gc.collect(); torch.cuda.empty_cache()
    finally:
        pipeline.vae.to(dtype=original_vae_dtype)
        if not checkpointing_was_enabled and callable(disable_checkpointing):
            disable_checkpointing()
        gc.collect(); torch.cuda.empty_cache()

    if len(all_checkpoints) != EXPECTED_CHECKPOINT_COUNT:
        raise RuntimeError(f"E043 produced {len(all_checkpoints)} checkpoints, expected {EXPECTED_CHECKPOINT_COUNT}")

    backend._pipeline = None
    del pipeline
    gc.collect(); torch.cuda.empty_cache()

    images: dict[str, Image.Image] = {}
    metadata: dict[str, dict[str, Any]] = {}
    recipe_order = {recipe.name: index for index, recipe in enumerate(RECIPES)}
    for item in all_checkpoints:
        key = f"{item.recipe}__i{item.iteration:02d}"
        images[key] = Image.open(item.image_path).convert("RGB")
        metadata[key] = {
            "phase": "E043", "recipe": item.recipe, "iteration": item.iteration,
            "gamma": GAMMA, "radius": LATENT_RADIUS_RMS,
            "image_path": item.image_path, "latent_path": item.latent_path,
            "is_parent_reference": item.recipe == CONTROL_RECIPE and item.iteration == 0,
            "control_reused_e041_latent": bool(item.trace_step.get("reused_e041_latent")),
            "optimization_performed_by_e043": bool(item.trace_step.get("optimization_performed_by_e043", item.recipe != CONTROL_RECIPE)),
        }

    parent_key = f"{CONTROL_RECIPE}__i00"
    parent_image = images[parent_key]
    rows, surrogate_status = _score_rows(
        images=images, metadata=metadata, output_dir=output_dir / "scoring",
        backend=backend, blueprint=blueprint, parent_image=parent_image,
        trace_lpips=None,
    )
    for row in rows:
        row.update(_scanner_diagnostics(images[str(row["variant"])], blueprint))
    _atomic_json(output_dir / "scanner-cell-comparison.json", rows)
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["visual_guard_checks"] = json.dumps(flat.get("visual_guard_checks") or {}, ensure_ascii=False, sort_keys=True)
        csv_rows.append(flat)
    _write_csv(output_dir / "scanner-cell-comparison.csv", csv_rows)

    safe = [row for row in rows if bool(row.get("visual_guard_pass"))]
    if not safe:
        raise RuntimeError("E043 produced no visually-safe checkpoint, including parent")
    winner = sorted(safe, key=lambda row: _rank_key(row, recipe_order))[0]
    raw_best = sorted(rows, key=lambda row: _rank_key(row, recipe_order))[0]
    control_rows = [row for row in rows if row["recipe"] == CONTROL_RECIPE]
    control_safe_rows = [row for row in control_rows if bool(row.get("visual_guard_pass"))]
    if not control_safe_rows:
        raise RuntimeError("E043 paired control has no visually-safe checkpoint")
    control_best = sorted(control_safe_rows, key=lambda row: _rank_key(row, recipe_order))[0]
    control_raw_best = sorted(control_rows, key=lambda row: _rank_key(row, recipe_order))[0]

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    reference = np.full((QR_CANVAS_PX, QR_CANVAS_PX), 255, dtype=np.uint8)
    target = _core_target_matrix(blueprint)
    for r in range(QR_CORE_MODULES):
        for c in range(QR_CORE_MODULES):
            if target[r, c]:
                y0 = QR_PADDING_PX + r * QR_MODULE_SIZE
                x0 = QR_PADDING_PX + c * QR_MODULE_SIZE
                reference[y0:y0+QR_MODULE_SIZE, x0:x0+QR_MODULE_SIZE] = 0
    Image.fromarray(reference, mode="L").convert("RGB").save(pipeline_dir / "01-qr-reference-exact736.png")
    shutil.copy2(e041_results_dir / "parent/stage1.png", pipeline_dir / "02-stage1.png")
    shutil.copy2(e041_results_dir / "parent/stage2.png", pipeline_dir / "03-stage2.png")
    images[parent_key].save(pipeline_dir / "04-stage2-exact-qz.png")
    for index, recipe in enumerate(RECIPES, start=5):
        recipe_rows = [row for row in rows if row["recipe"] == recipe.name]
        recipe_safe = [row for row in recipe_rows if bool(row.get("visual_guard_pass"))]
        best = sorted(recipe_safe or recipe_rows, key=lambda row: _rank_key(row, recipe_order))[0]
        images[str(best["variant"])].save(
            pipeline_dir / f"{index:02d}-{recipe.name}.png",
            format="PNG", optimize=False, compress_level=9,
        )

    final_image = pipeline_dir / "99-FINAL-QR.png"
    final_latent = pipeline_dir / "99-FINAL-latent.safetensors"
    shutil.copy2(Path(str(winner["image_path"])), final_image)
    shutil.copy2(Path(str(winner["latent_path"])), final_latent)
    _atomic_json(pipeline_dir / "99-FINAL-metadata.json", {
        "winner_variant": winner["variant"], "recipe": winner["recipe"],
        "iteration": winner["iteration"], "gamma": GAMMA,
        "radius": LATENT_RADIUS_RMS,
        "ssr_exact_presets": winner["qr_verify_exact_presets"],
        "original_exact": winner["original_exact"],
        "visual_guard_pass": winner["visual_guard_pass"],
        "exact_quiet_zone_geometry": True,
    })

    sheet = [
        ("Exact QR", Image.open(pipeline_dir / "01-qr-reference-exact736.png").convert("RGB"), "736 / qz=78 / core=580"),
        ("Stage1", Image.open(pipeline_dir / "02-stage1.png").convert("RGB"), "paired E041 prompt"),
        ("Stage2", Image.open(pipeline_dir / "03-stage2.png").convert("RGB"), "fresh E041 SRPG parent"),
        ("Stage2 exact QZ", parent_image, "legacy core overwrite removed"),
    ]
    for recipe in RECIPES:
        recipe_rows = [row for row in rows if row["recipe"] == recipe.name]
        recipe_safe = [row for row in recipe_rows if bool(row.get("visual_guard_pass"))]
        best = sorted(recipe_safe or recipe_rows, key=lambda row: _rank_key(row, recipe_order))[0]
        sheet.append((recipe.name, images[str(best["variant"])], f"i{best['iteration']} SSR={best['qr_verify_exact_presets']}/37 safe={best['visual_guard_pass']}"))
    sheet.append(("FINAL E043", Image.open(final_image).convert("RGB"), f"{winner['recipe']} i{winner['iteration']} SSR={winner['qr_verify_exact_presets']}/37"))
    _comparison_sheet(pipeline_dir / "full-pipeline-contact-sheet.png", sheet, columns=3)

    control_ssr = int(control_best["qr_verify_exact_presets"])
    winner_ssr = int(winner["qr_verify_exact_presets"])
    real_decode_gain = winner_ssr > control_ssr or (bool(winner["original_exact"]) and not bool(control_best["original_exact"]))
    verdict = {
        "experiment": EXPERIMENT, "source_commit": source_commit,
        "paired_prompt_parent_with_e041": True,
        "e042_primary_blocker": e042_verdict["primary_blocker"],
        "gamma": GAMMA, "latent_radius_rms": LATENT_RADIUS_RMS,
        "recipe_count": len(RECIPES), "checkpoint_count": len(rows),
        "control_checkpoint_count": len(control_rows),
        "new_optimized_recipe_count": len(RECIPES)-1,
        "new_srmpgd_update_budget": 3 * MAX_ITERATIONS,
        "control_best_ssr_exact_presets": control_ssr,
        "control_raw_best_ssr_exact_presets": int(control_raw_best["qr_verify_exact_presets"]),
        "winner_variant": winner["variant"], "winner_recipe": winner["recipe"],
        "winner_iteration": int(winner["iteration"]),
        "winner_ssr_exact_presets": winner_ssr, "winner_ssr": float(winner["ssr"]),
        "winner_original_exact": bool(winner["original_exact"]),
        "winner_visual_guard_pass": bool(winner["visual_guard_pass"]),
        "winner_visual_guard_checks": winner["visual_guard_checks"],
        "winner_mean_module_errors": int(winner["mean_total_error_count"]),
        "winner_mean_format_errors": int(winner["mean_format_error_count"]),
        "winner_mean_data_errors": int(winner["mean_data_error_count"]),
        "winner_intra_module_std_p90": float(winner["intra_module_std_p90"]),
        "raw_best_variant": raw_best["variant"],
        "raw_best_ssr_exact_presets": int(raw_best["qr_verify_exact_presets"]),
        "real_decode_gain_over_paired_control": bool(real_decode_gain),
        "exact_diffqrcoder_quiet_zone_geometry": True,
        "legacy_quiet_zone_core_overwrite": False,
        "ecc_awareness_kind": "risk-weighted data-module margin proxy; not differentiable Reed-Solomon",
        "e016_surrogate_research_usable": bool(surrogate_status.get("research_usable")),
        "production_ready": False, "generalization_authorized": False,
        "next_action": "EXPAND_SCANNER_CELL_RECIPE_LOCALLY_BEFORE_GENERALIZATION" if real_decode_gain else "REFINE_SCANNER_CELL_LOSS_FROM_E043_FAILURES_BEFORE_GENERALIZATION",
    }
    _atomic_json(output_dir / "verdict.json", verdict)
    _atomic_text(output_dir / "report.md", "\n".join([
        "# E043 — scanner-cell frontier", "",
        f"- E042 blocker: **{e042_verdict['primary_blocker']}**",
        "- prompt/parent: paired with E041", f"- gamma: **{GAMMA:g}**",
        f"- radius: **{LATENT_RADIUS_RMS}**",
        f"- checkpoints: **{len(rows)}** (9 control re-decodes + 27 B/C/D)",
        f"- winner: **{winner['variant']}**", f"- SSR: **{winner_ssr}/37**",
        f"- original exact: **{winner['original_exact']}**",
        f"- visual guard: **{winner['visual_guard_pass']}**",
        f"- paired control SSR: **{control_ssr}/37**",
        f"- real decode gain: **{real_decode_gain}**", "",
        "ECC-awareness is a data-module risk proxy, not a differentiable Reed-Solomon decoder.",
        "Production/generalization remain forbidden.",
    ]))
    _atomic_json(output_dir / "e043-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--e041-results-dir", type=Path, required=True)
    parser.add_argument("--e042-results-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    verdict = run_e043(output_dir=args.output_dir, e041_results_dir=args.e041_results_dir, e042_results_dir=args.e042_results_dir, source_commit=args.source_commit)
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
