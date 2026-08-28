"""E036 — gamma-preserving perceptual trust-region SR-MPGD.

E036 keeps the raw SR-MPGD proposal scale frozen to gamma=1000 and changes only
how the proposal is accepted. The official pinned DiffQRCoder scanning-robust loss
is used for every branch. A raw step is first proposed with ``-1000 * grad`` and is
then projected/backtracked into a preregistered trust region around the immutable
E035 parent latent.

The experiment is intentionally single-parent and additive. It never regenerates
Stage 1 or Stage 2 and never marks a result production-ready.
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
    E035Config,
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

EXPERIMENT = "e036-gamma1000-perceptual-trust-region-v1"

BRANCH_GLOBAL = "e036_gamma1000_global_trust"
BRANCH_STRICT = "e036_gamma1000_strict_trust"
BRANCH_LOCAL = "e036_gamma1000_local_preserve"

BranchName = Literal[
    "e036_gamma1000_global_trust",
    "e036_gamma1000_strict_trust",
    "e036_gamma1000_local_preserve",
]


@dataclass(frozen=True, slots=True)
class BranchPolicy:
    name: BranchName
    latent_radius_rms: float
    lpips_budget: float
    core_mae_budget: float
    outside_active_mae_budget: float | None = None
    active_dilation_modules: int = 1


DEFAULT_POLICIES: tuple[BranchPolicy, ...] = (
    BranchPolicy(
        name=BRANCH_GLOBAL,
        latent_radius_rms=0.050,
        lpips_budget=0.050,
        core_mae_budget=0.050,
    ),
    BranchPolicy(
        name=BRANCH_STRICT,
        latent_radius_rms=0.025,
        lpips_budget=0.020,
        core_mae_budget=0.030,
    ),
    BranchPolicy(
        name=BRANCH_LOCAL,
        latent_radius_rms=0.050,
        lpips_budget=0.050,
        core_mae_budget=0.050,
        outside_active_mae_budget=0.010,
        active_dilation_modules=1,
    ),
)


@dataclass(frozen=True, slots=True)
class E036Config:
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
    srl_nonincrease_tolerance: float = 2e-6


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    upstream_srl: float
    lpips: float
    core_mae: float
    outside_active_mae: float | None
    latent_delta_rms: float
    accepted: bool
    checks: dict[str, bool]


@dataclass(frozen=True, slots=True)
class E036Step:
    branch: str
    iteration: int
    elapsed_s: float
    image_sha256: str
    latent_sha256: str
    upstream_srl: float
    lpips_loss: float
    core_mae: float
    outside_active_mae: float | None
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
class BranchResult:
    name: str
    policy: dict[str, Any]
    output_dir: str
    final_image_path: str
    final_latent_path: str
    trace_path: str
    final_step: dict[str, Any]


def _rms(tensor: Any) -> Any:
    return tensor.square().mean().sqrt()


def project_latent_candidate(candidate: Any, center: Any, max_rms: float) -> Any:
    """Project a latent candidate onto an RMS ball around ``center``."""

    if max_rms <= 0:
        raise ValueError("max_rms must be positive")
    delta = candidate - center
    delta_rms = _rms(delta)
    if float(delta_rms.detach().cpu()) <= max_rms:
        return candidate
    scale = max_rms / float(delta_rms.detach().cpu())
    return center + delta * scale


def dilate_active_modules(active_modules: Any, *, radius: int) -> Any:
    """Dilate a BxN boolean module mask on its square QR module grid."""

    import torch
    import torch.nn.functional as F

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if active_modules.ndim != 2:
        raise ValueError("active_modules must have shape [batch, modules]")
    module_count = int(active_modules.shape[1])
    side = int(round(math.sqrt(module_count)))
    if side * side != module_count:
        raise ValueError("active module mask must describe a square QR grid")
    grid = active_modules.to(dtype=torch.float32).reshape(-1, 1, side, side)
    if radius:
        kernel = radius * 2 + 1
        grid = F.max_pool2d(grid, kernel_size=kernel, stride=1, padding=radius)
    return grid.reshape(active_modules.shape[0], module_count) > 0


def expand_active_module_mask(
    active_modules: Any,
    module_ids: Any,
    *,
    height: int,
    width: int,
    dilation_modules: int,
) -> Any:
    """Expand an active module mask to a Bx1xHxW pixel mask."""

    dilated = dilate_active_modules(active_modules, radius=dilation_modules)
    batch = dilated.shape[0]
    pixels = dilated[:, module_ids].reshape(batch, 1, height, width)
    return pixels


def _normalised_core_mae(current: Any, reference: Any) -> float:
    # VAE tensors are in [-1, 1], so divide absolute error by two for [0, 1] scale.
    return float(((current - reference).abs().mean() / 2).detach().cpu())


def _outside_active_mae(current: Any, reference: Any, active_pixels: Any | None) -> float | None:
    if active_pixels is None:
        return None
    import torch

    outside = (~active_pixels.bool()).expand_as(current)
    if not bool(outside.any()):
        return 0.0
    values = (current - reference).abs()[outside] / 2
    return float(torch.mean(values).detach().cpu())


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
    active_pixels: Any | None,
    policy: BranchPolicy,
    current_srl: float,
    config: E036Config,
) -> CandidateMetrics:
    import torch

    with torch.no_grad():
        decoded = _decode_latent_tensor(pipeline, candidate).float()
        core = _crop_core(decoded, config.crop_padding_px).detach()
        unit = (core / 2 + 0.5).clamp(0, 1)
        local_loss, _ = upstream_code_scanning_robust_loss(
            unit,
            core_blueprint,
            layout=upstream_layout,
        )
        official_loss = official_upstream_srl(unit, upstream_target)
        upstream_srl, _, _ = _assert_upstream_reference_match(
            local=local_loss,
            official=official_loss,
            config=config,
            branch=policy.name,
            iteration=-1,
            phase="candidate",
        )
        lpips = float(
            lpips_model(
                core.to(device="cpu", dtype=lpips_dtype),
                reference_lpips,
            ).mean().detach().cpu()
        )
        core_mae = _normalised_core_mae(core, reference_core)
        outside = _outside_active_mae(core, reference_core, active_pixels)
        latent_delta = float(_rms(candidate - initial).detach().cpu())

    checks = {
        "latent_radius": latent_delta <= policy.latent_radius_rms + 1e-9,
        "lpips_budget": lpips <= policy.lpips_budget + 1e-9,
        "core_mae_budget": core_mae <= policy.core_mae_budget + 1e-9,
        "srl_nonincrease": upstream_srl <= current_srl + config.srl_nonincrease_tolerance,
    }
    if policy.outside_active_mae_budget is not None:
        checks["outside_active_mae_budget"] = (
            outside is not None
            and outside <= policy.outside_active_mae_budget + 1e-9
        )
    return CandidateMetrics(
        upstream_srl=upstream_srl,
        lpips=lpips,
        core_mae=core_mae,
        outside_active_mae=outside,
        latent_delta_rms=latent_delta,
        accepted=all(checks.values()),
        checks=checks,
    )


def _write_trace_csv(path: Path, steps: list[E036Step]) -> None:
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


def _run_branch(
    *,
    pipeline: Any,
    parent: LoadedParentArtifact,
    blueprint: Any,
    policy: BranchPolicy,
    config: E036Config,
    output_root: Path,
) -> BranchResult:
    import torch
    from safetensors.torch import save_file

    branch_root = output_root / policy.name
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

    steps: list[E036Step] = []
    started = time.perf_counter()

    for iteration in range(config.max_iterations + 1):
        working = working.detach()
        with torch.no_grad():
            decoded = reference_decoded if iteration == 0 else _decode_latent_tensor(pipeline, working).float()
            decoded_core = _crop_core(decoded, config.crop_padding_px).detach()
            decoded_unit = (decoded_core / 2 + 0.5).clamp(0, 1)
            local_loss, local_diag = upstream_code_scanning_robust_loss(
                decoded_unit,
                core_blueprint,
                layout=upstream_layout,
            )
            official_loss = official_upstream_srl(decoded_unit, upstream_target)
            upstream_srl, _, _ = _assert_upstream_reference_match(
                local=local_loss,
                official=official_loss,
                config=config,
                branch=policy.name,
                iteration=iteration,
                phase="evaluation",
            )
            active_modules = local_diag["active_mask"].detach()
            active_pixels = expand_active_module_mask(
                active_modules,
                upstream_layout.module_ids,
                height=core_height,
                width=core_width,
                dilation_modules=policy.active_dilation_modules,
            )
            lpips_loss = float(
                lpips_model(
                    decoded_core.to(device="cpu", dtype=lpips_dtype),
                    reference_lpips,
                ).mean().detach().cpu()
            )
            core_mae = _normalised_core_mae(decoded_core, reference_core)
            outside_mae = _outside_active_mae(decoded_core, reference_core, active_pixels)
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

        if iteration < config.max_iterations and upstream_srl > 0.0:
            srl_core = decoded_core.detach().requires_grad_(True)
            srl_unit = (srl_core / 2 + 0.5).clamp(0, 1)
            local_selected_loss, _ = upstream_code_scanning_robust_loss(
                srl_unit,
                core_blueprint,
                layout=upstream_layout,
            )
            selected_loss = official_upstream_srl(srl_unit, upstream_target)
            _assert_upstream_reference_match(
                local=local_selected_loss,
                official=selected_loss,
                config=config,
                branch=policy.name,
                iteration=iteration,
                phase="gradient",
            )
            selected_srl_gradient = torch.autograd.grad(
                selected_loss,
                srl_core,
                only_inputs=True,
            )[0]

            lpips_core = decoded_core.detach().to(device="cpu", dtype=lpips_dtype).requires_grad_(True)
            lpips_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(
                lpips_tensor,
                lpips_core,
                only_inputs=True,
            )[0]

            objective_gradient = selected_srl_gradient.clone()
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

            del srl_core, srl_unit, local_selected_loss, selected_loss
            del selected_srl_gradient, lpips_core, lpips_tensor, lpips_gradient_cpu, objective_gradient

            if gradient is None:
                acceptance_reason = "no_finite_latent_gradient"
            else:
                latent_gradient_rms = float(_rms(gradient).detach().cpu())
                raw_target = working - config.gamma * gradient
                raw_step_rms = float(_rms(raw_target - working).detach().cpu())
                projected_target = project_latent_candidate(
                    raw_target,
                    initial,
                    policy.latent_radius_rms,
                )
                projected_step_rms = float(_rms(projected_target - working).detach().cpu())
                direction = projected_target - working

                accepted_metrics: CandidateMetrics | None = None
                alpha = 1.0
                for _ in range(config.max_backtracks + 1):
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
                        active_pixels=active_pixels,
                        policy=policy,
                        current_srl=upstream_srl,
                        config=config,
                    )
                    if metrics.accepted:
                        next_working = trial
                        accepted_metrics = metrics
                        accepted_alpha = alpha
                        accepted_step_rms = float(_rms(next_working - working).detach().cpu())
                        acceptance_reason = "accepted"
                        candidate_checks = metrics.checks
                        break
                    alpha *= 0.5

                if accepted_metrics is None:
                    acceptance_reason = "trust_region_rejected_all_candidates"
                    accepted_alpha = 0.0
                    accepted_step_rms = 0.0
                    candidate_checks = metrics.checks if "metrics" in locals() else None

                del gradient, raw_target, projected_target, direction

        elif iteration < config.max_iterations:
            acceptance_reason = "upstream_srl_zero_hold_state"
            accepted_alpha = 0.0
            accepted_step_rms = 0.0

        latent_delta_rms = float(_rms(working - initial).detach().cpu())
        step = E036Step(
            branch=policy.name,
            iteration=iteration,
            elapsed_s=time.perf_counter() - started,
            image_sha256=_image_sha256(image),
            latent_sha256=tensor_sha256(working),
            upstream_srl=upstream_srl,
            lpips_loss=lpips_loss,
            core_mae=core_mae,
            outside_active_mae=outside_mae,
            upstream_active_modules=int(local_diag["active_modules"].detach().cpu()),
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
    result = BranchResult(
        name=policy.name,
        policy=asdict(policy),
        output_dir=str(branch_root),
        final_image_path=str(final_image_path),
        final_latent_path=str(final_latent_path),
        trace_path=str(trace_path),
        final_step=asdict(steps[-1]),
    )
    _atomic_json(branch_root / "branch-result.json", asdict(result))
    return result


def _comparison_contact_sheet(
    output_path: Path,
    items: list[tuple[str, Image.Image, str]],
) -> None:
    columns = 3
    tile_w = 420
    tile_h = 470
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image, subtitle) in enumerate(items):
        row, col = divmod(index, columns)
        x0 = col * tile_w
        y0 = row * tile_h
        preview = image.convert("RGB").copy()
        preview.thumbnail((400, 390), Image.Resampling.LANCZOS)
        x = x0 + (tile_w - preview.width) // 2
        y = y0 + 58 + (390 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((x0 + 10, y0 + 10), label, fill=(0, 0, 0))
        draw.text((x0 + 10, y0 + 30), subtitle, fill=(60, 60, 60))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False, compress_level=9)


def _load_optional_e035_images(e035_results_dir: Path) -> dict[str, Image.Image]:
    result: dict[str, Image.Image] = {}
    candidates = {
        "e035_paper": e035_results_dir / "e035_paper_srl_control/images/iteration-004.png",
        "e035_upstream": e035_results_dir / "e035_upstream_code_srl/images/iteration-004.png",
    }
    for name, path in candidates.items():
        if path.is_file():
            result[name] = Image.open(path).convert("RGB")
    return result


def _extract_exact_presets(evidence: dict[str, Any], name: str) -> int:
    item = evidence.get(name) or {}
    return int(item.get("conservative_exact_presets", 0))


def run_e036(
    *,
    parent_dir: Path,
    output_dir: Path,
    e035_results_dir: Path,
    policies: tuple[BranchPolicy, ...] = DEFAULT_POLICIES,
    config: E036Config | None = None,
    expected_parent_commit: str | None = None,
    skip_qr_verify: bool = False,
) -> dict[str, Any]:
    import torch
    from .qr import generate_diffqrcoder_qr

    config = config or E036Config()
    if not torch.cuda.is_available():
        raise RuntimeError("E036 requires an available CUDA GPU")
    if config.gamma != 1000.0:
        raise ValueError("E036 is frozen to gamma=1000")
    if config.max_iterations != 4:
        raise ValueError("E036 is frozen to four recorded updates")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"E036 output directory must be empty: {output_dir}")

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
            "policies": [asdict(policy) for policy in policies],
            "parent_contract_sha256": parent.metadata["contract_sha256"],
            "gamma_is_fixed": True,
            "production_ready": False,
            "automatic_expansion_authorized": False,
        },
    )
    _atomic_json(output_dir / "parent-verification.json", parent.metadata)

    backend, pipeline = _load_pipeline()
    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)

    branch_results: dict[str, BranchResult] = {}
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
                for policy in policies:
                    branch_results[policy.name] = _run_branch(
                        pipeline=pipeline,
                        parent=parent,
                        blueprint=blueprint,
                        policy=policy,
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

    first_result = next(iter(branch_results.values()))
    parent_fp32_path = Path(first_result.output_dir) / "images/iteration-000.png"
    parent_fp32 = Image.open(parent_fp32_path).convert("RGB")
    parent_fp32.save(output_dir / "parent-fp32-redecoded.png", format="PNG", optimize=False, compress_level=9)

    reference_images: dict[str, Image.Image] = {
        "parent_source": parent.image,
        "parent_fp32": parent_fp32,
    }
    reference_images.update(_load_optional_e035_images(e035_results_dir))
    for name, result in branch_results.items():
        reference_images[name] = Image.open(result.final_image_path).convert("RGB")

    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, payload, reference_images)
    if skip_qr_verify:
        _atomic_json(output_dir / "qr-verify-evidence.json", {"skipped": True})

    rows: list[dict[str, Any]] = []
    for name, result in branch_results.items():
        final = result.final_step
        rows.append(
            {
                "branch": name,
                "gamma": config.gamma,
                "qr_verify_exact_presets": (
                    0 if qr_verify is None else _extract_exact_presets(qr_verify, name)
                ),
                "upstream_srl": final["upstream_srl"],
                "upstream_active_modules": final["upstream_active_modules"],
                "full_module_error_count": final["full_module_error_count"],
                "full_module_error_rate": final["full_module_error_rate"],
                "lpips": final["lpips_loss"],
                "core_mae": final["core_mae"],
                "outside_active_mae": final["outside_active_mae"],
                "latent_delta_rms": final["latent_delta_rms"],
                "policy_latent_radius_rms": result.policy["latent_radius_rms"],
                "policy_lpips_budget": result.policy["lpips_budget"],
                "policy_core_mae_budget": result.policy["core_mae_budget"],
                "policy_outside_active_mae_budget": result.policy["outside_active_mae_budget"],
            }
        )
    summary_csv = output_dir / "branch-summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(output_dir / "branch-summary.json", rows)

    # Select a research winner only among branches that stayed inside their own final budgets.
    def final_is_safe(row: dict[str, Any]) -> bool:
        outside_budget = row["policy_outside_active_mae_budget"]
        return (
            row["latent_delta_rms"] <= row["policy_latent_radius_rms"] + 1e-9
            and row["lpips"] <= row["policy_lpips_budget"] + 1e-9
            and row["core_mae"] <= row["policy_core_mae_budget"] + 1e-9
            and (
                outside_budget is None
                or (
                    row["outside_active_mae"] is not None
                    and row["outside_active_mae"] <= outside_budget + 1e-9
                )
            )
        )

    safe_rows = [row for row in rows if final_is_safe(row)]
    ranked = sorted(
        safe_rows,
        key=lambda row: (
            -int(row["qr_verify_exact_presets"]),
            int(row["full_module_error_count"]),
            float(row["lpips"]),
            float(row["latent_delta_rms"]),
        ),
    )
    winner = ranked[0]["branch"] if ranked else None
    best_exact = ranked[0]["qr_verify_exact_presets"] if ranked else 0
    verdict = {
        "experiment": EXPERIMENT,
        "gamma": config.gamma,
        "gamma_preserved": True,
        "production_ready": False,
        "automatic_expansion_authorized": False,
        "advisor_training_authorized": False,
        "research_winner": winner,
        "best_conservative_exact_presets": best_exact,
        "decision": (
            "PREPARE_MINI_HOLDOUT_WITH_WINNER"
            if winner is not None and best_exact >= 1
            else "TUNE_TRUST_REGION_OR_REVISIT_UPDATE_GEOMETRY"
        ),
        "branches": {name: asdict(result) for name, result in branch_results.items()},
    }
    _atomic_json(output_dir / "verdict.json", verdict)

    items: list[tuple[str, Image.Image, str]] = [
        ("Parent FP32", parent_fp32, "reference D(z0)"),
    ]
    if "e035_paper" in reference_images:
        items.append(("E035 paper", reference_images["e035_paper"], "gamma=1000 / paper loss"))
    if "e035_upstream" in reference_images:
        exact = 0 if qr_verify is None else _extract_exact_presets(qr_verify, "e035_upstream")
        items.append(("E035 upstream", reference_images["e035_upstream"], f"unbounded / QR exact={exact}"))
    summary_by_name = {row["branch"]: row for row in rows}
    for name, result in branch_results.items():
        row = summary_by_name[name]
        subtitle = (
            f"exact={row['qr_verify_exact_presets']}  MER={row['full_module_error_rate']:.4f}  "
            f"LPIPS={row['lpips']:.4f}"
        )
        items.append((name.replace("e036_gamma1000_", "E036 "), reference_images[name], subtitle))
    _comparison_contact_sheet(output_dir / "e036-final-contact-sheet.png", items)

    report = f"""# E036 — gamma=1000 perceptual trust-region SR-MPGD

- Gamma is frozen to **1000** for every raw proposal.
- Official upstream DiffQRCoder SRL revision: `{UPSTREAM_REVISION}`.
- Parent contract: `{parent.metadata['contract_sha256']}`.
- Research winner: **{winner or 'none'}**.
- Best conservative QR-Verify exact presets: **{best_exact}**.
- Production ready: **no**.

The experiment compares projection/acceptance policies rather than reducing gamma.
The contact sheet contains the FP32 parent, available E035 controls, and all E036 finals.
"""
    _atomic_text(output_dir / "report.md", report)
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--e035-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-parent-commit", default=None)
    parser.add_argument("--skip-qr-verify", action="store_true")
    args = parser.parse_args()
    verdict = run_e036(
        parent_dir=args.parent_dir,
        e035_results_dir=args.e035_results_dir,
        output_dir=args.output_dir,
        expected_parent_commit=args.expected_parent_commit,
        skip_qr_verify=args.skip_qr_verify,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
