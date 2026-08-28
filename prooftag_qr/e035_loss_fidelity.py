"""Standalone GPU runner for E035 — SR-MPGD loss fidelity gate.

The runner deliberately bypasses the multi-prompt Lab campaign. It loads one immutable
Stage-2 parent artefact, starts two FP32 branches from the exact same latent, performs
four fixed Eq. 14 updates, evaluates both SRL profiles at every iteration, and produces
machine-readable evidence. It never invokes Stage 1 or Stage 2 generation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from .e035_losses import (
    UPSTREAM_REVISION,
    combined_gradient_gate,
    evaluate_loss_profiles,
    module_diagnostics,
    module_error_maps,
    prepare_upstream_torch_layout,
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

E034_BASE_COMMIT = "933a947ee226e9e6b36145d35e41d53c90f50484"
EXPERIMENT = "e035-srmpgd-loss-fidelity-gate-v1"
BRANCH_PAPER = "e035_paper_srl_control"
BRANCH_UPSTREAM = "e035_upstream_code_srl"
BranchName = Literal[
    "e035_paper_srl_control",
    "e035_upstream_code_srl",
]


@dataclass(frozen=True, slots=True)
class E035Config:
    max_iterations: int = 4
    step_size: float = 1000.0
    gradient_scale: float = 32768.0
    min_gradient_rms: float = 1e-12
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


@dataclass(frozen=True, slots=True)
class E035Step:
    branch: str
    iteration: int
    elapsed_s: float
    image_sha256: str
    latent_sha256: str
    selected_srl_profile: str
    selected_srl: float
    paper_srl: float
    upstream_srl: float
    lpips_loss: float
    objective: float
    selected_srl_image_gradient_rms: float | None
    lpips_image_gradient_rms: float | None
    weighted_lpips_image_gradient_rms: float | None
    objective_image_gradient_rms: float | None
    latent_gradient_rms: float | None
    requested_step_rms: float | None
    applied_step_rms: float | None
    latent_delta_rms: float
    effective_gradient_scale: float | None
    gradient_gate_passed: bool | None
    gradient_gate: dict[str, Any] | None
    diagnostics: dict[str, Any]
    visual_change: dict[str, float]
    cuda: dict[str, int | None]


@dataclass(frozen=True, slots=True)
class E035BranchResult:
    name: str
    selected_srl_profile: str
    output_dir: str
    final_image_path: str
    final_latent_path: str
    trace_path: str
    selected_iteration: int
    final_step: dict[str, Any]


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        if value and not value.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _image_sha256(image: Image.Image) -> str:
    from .quality import image_sha256

    return image_sha256(image)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _cuda_snapshot() -> dict[str, int | None]:
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "driver_free_bytes": None,
            "driver_total_bytes": None,
        }
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def _gradient_scales(maximum: float) -> tuple[float, ...]:
    values: list[float] = []
    current = float(maximum)
    while current > 1.0:
        values.append(current)
        current = max(1.0, current / 4.0)
    if not values or values[-1] != 1.0:
        values.append(1.0)
    return tuple(values)


def _load_lpips(pipeline: Any, *, net: str) -> Any:
    import lpips

    cache_name = f"_prooftag_e035_lpips_{net}"
    cached = getattr(pipeline, cache_name, None)
    if cached is None:
        cached = lpips.LPIPS(net=net, verbose=False)
        cached.requires_grad_(False).eval()
        setattr(pipeline, cache_name, cached)
    return cached.to(device="cpu")


def _decode_latent_tensor(pipeline: Any, latent: Any) -> Any:
    vae = pipeline.vae
    return vae.decode(
        latent / vae.config.scaling_factor,
        return_dict=False,
    )[0]


def _decoded_to_scan_ready_image(
    pipeline: Any,
    decoded: Any,
    blueprint: Any,
    config: E035Config,
) -> Image.Image:
    from .qr import prepare_scan_ready_image

    image = pipeline.image_processor.postprocess(
        decoded.detach(),
        output_type="pil",
        do_denormalize=[True],
    )[0].convert("RGB")
    return prepare_scan_ready_image(
        image,
        blueprint,
        quiet_zone_mode=config.quiet_zone_mode,
        quiet_zone_minimum_luminance=config.quiet_zone_minimum_luminance,
        functional_pattern_tone_factor=config.functional_pattern_tone_factor,
    )


def _crop_core(tensor: Any, padding: int) -> Any:
    if padding < 0:
        raise ValueError("crop padding cannot be negative")
    if padding == 0:
        return tensor
    if tensor.shape[-2] <= 2 * padding or tensor.shape[-1] <= 2 * padding:
        raise ValueError("crop padding removes the complete decoded image")
    return tensor[..., padding:-padding, padding:-padding]


def _core_blueprint(blueprint: Any, width: int, height: int) -> Any:
    from .qr import QRBlueprint

    border = int(blueprint.border)
    matrix = (
        blueprint.matrix[border:-border, border:-border].copy()
        if border
        else blueprint.matrix.copy()
    )
    return QRBlueprint(
        image=Image.new("RGB", (width, height), "white"),
        matrix=matrix,
        version=blueprint.version,
        border=0,
    )


def _load_official_upstream_srl(module_size: int, *, device: Any) -> Any:
    """Load the loss class from the pinned DiffQRCoder checkout in the GPU image."""

    import torch

    try:
        from diffqrcoder.losses.scanning_robust_loss import ScanningRobustLoss
    except Exception as exc:  # pragma: no cover - exercised inside the CUDA image
        raise RuntimeError(
            "the pinned DiffQRCoder ScanningRobustLoss is unavailable; "
            "E035 refuses to approximate the upstream branch"
        ) from exc
    reference = ScanningRobustLoss(module_size=module_size)
    reference.requires_grad_(False).eval().to(
        device=device,
        dtype=torch.float32,
    )
    return reference


def _assert_upstream_reference_match(
    *,
    local: Any,
    official: Any,
    config: E035Config,
    branch: str,
    iteration: int,
    phase: str,
) -> tuple[float, float, float]:
    """Fail closed when the local diagnostic port diverges from upstream code."""

    import torch

    local_value = float(local.detach().cpu())
    official_value = float(official.detach().cpu())
    absolute_error = abs(local_value - official_value)
    denominator = max(abs(official_value), 1e-12)
    relative_error = absolute_error / denominator
    if not torch.allclose(
        local.detach(),
        official.detach(),
        atol=config.upstream_reference_atol,
        rtol=config.upstream_reference_rtol,
    ):
        raise RuntimeError(
            "local upstream SRL port diverged from the pinned official class: "
            f"branch={branch}, iteration={iteration}, phase={phase}, "
            f"local={local_value:.12g}, official={official_value:.12g}, "
            f"abs_error={absolute_error:.12g}, rel_error={relative_error:.12g}"
        )
    return official_value, absolute_error, relative_error


@contextmanager
def _offload_diffusion_modules(pipeline: Any):
    """Keep only VAE on CUDA while E035 performs post-processing."""

    import torch

    moved: list[tuple[str, Any, Any]] = []
    for name in (
        "unet",
        "controlnet",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "srpg",
    ):
        module = getattr(pipeline, name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        parameter = next(iter(module.parameters()), None)
        device = getattr(parameter, "device", None) if parameter is not None else None
        if getattr(device, "type", None) != "cuda":
            continue
        moved.append((name, module, device))
        module.to(device="cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        yield tuple(name for name, _, _ in moved)
    finally:
        for _, module, device in reversed(moved):
            module.to(device=device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _save_error_map(
    path: Path,
    errors: np.ndarray,
    functional: np.ndarray,
    *,
    scale: int = 16,
) -> None:
    if errors.shape != functional.shape:
        raise ValueError("error and functional maps must have the same shape")
    canvas = np.full((*errors.shape, 3), 245, dtype=np.uint8)
    canvas[errors & ~functional] = (230, 145, 35)
    canvas[errors & functional] = (185, 35, 35)
    canvas[~errors & functional] = (190, 215, 235)
    image = Image.fromarray(canvas, mode="RGB").resize(
        (errors.shape[1] * scale, errors.shape[0] * scale),
        Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    for row in range(errors.shape[0] + 1):
        y = row * scale
        draw.line((0, y, image.width, y), fill=(120, 120, 120), width=1)
    for col in range(errors.shape[1] + 1):
        x = col * scale
        draw.line((x, 0, x, image.height), fill=(120, 120, 120), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _write_trace_csv(path: Path, steps: Iterable[E035Step]) -> None:
    rows: list[dict[str, Any]] = []
    for step in steps:
        row = {
            key: value
            for key, value in asdict(step).items()
            if key not in {"diagnostics", "visual_change", "cuda", "gradient_gate"}
        }
        row.update({f"diag_{key}": value for key, value in step.diagnostics.items()})
        row.update({f"visual_{key}": value for key, value in step.visual_change.items()})
        row.update({f"cuda_{key}": value for key, value in step.cuda.items()})
        if step.gradient_gate:
            row.update(
                {f"gate_{key}": value for key, value in step.gradient_gate.items()}
            )
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_branch(
    *,
    pipeline: Any,
    parent: LoadedParentArtifact,
    blueprint: Any,
    config: E035Config,
    branch: BranchName,
    output_root: Path,
) -> E035BranchResult:
    import torch
    from safetensors.torch import save_file

    from .guidance import prepare_torch_layout
    from .quality import image_change_metrics
    from .qr import functional_pattern_mask

    selected_profile = (
        "paper_v3" if branch == BRANCH_PAPER else "upstream_code_e24ea73"
    )
    branch_root = output_root / branch
    images_root = branch_root / "images"
    maps_root = branch_root / "diagnostic-maps"
    branch_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    # The immutable source tensor was already byte-verified by load_parent_artifact.
    # Eq. 14 uses an FP32 working copy, exactly as E034's FP32 branch did; changing
    # dtype changes the tensor hash by design, so the source and working hashes are
    # recorded separately rather than compared.
    initial = parent.latent.detach().to(device=device, dtype=torch.float32).clone()
    working = initial.clone()

    with torch.no_grad():
        reference_decoded = _decode_latent_tensor(pipeline, working).float().detach()
    reference_core = _crop_core(reference_decoded, config.crop_padding_px).detach()
    core_height, core_width = reference_core.shape[-2:]
    core_blueprint = _core_blueprint(blueprint, core_width, core_height)
    if core_width != core_blueprint.matrix.shape[1] * config.qr_module_size:
        raise ValueError("E035 core width does not match 29 modules of 20 pixels")
    if core_height != core_blueprint.matrix.shape[0] * config.qr_module_size:
        raise ValueError("E035 core height does not match 29 modules of 20 pixels")

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
    functional = np.asarray(functional_pattern_mask(core_blueprint), dtype=bool)
    lpips_model = _load_lpips(pipeline, net=config.lpips_net)
    lpips_parameter = next(iter(lpips_model.parameters()), None)
    lpips_dtype = lpips_parameter.dtype if lpips_parameter is not None else torch.float32
    reference_lpips = reference_core.to(device="cpu", dtype=lpips_dtype).detach()

    started = time.perf_counter()
    steps: list[E035Step] = []
    for iteration in range(config.max_iterations + 1):
        working = working.detach()
        with torch.no_grad():
            decoded = (
                reference_decoded
                if iteration == 0
                else _decode_latent_tensor(pipeline, working).float()
            )
        decoded_core = _crop_core(decoded.float(), config.crop_padding_px).detach()
        decoded_unit = (decoded_core / 2 + 0.5).clamp(0, 1)

        # The scientific state at i0 is the FP32 VAE re-decode D(z0), not the
        # source PNG produced earlier in FP16. Keeping the two rasters separate makes
        # any precision/reconstruction delta explicit instead of hiding it.
        image = _decoded_to_scan_ready_image(pipeline, decoded, blueprint, config)
        image_path = images_root / f"iteration-{iteration:03d}.png"
        image.save(image_path, format="PNG", optimize=False, compress_level=9)

        with torch.no_grad():
            profiles = evaluate_loss_profiles(
                decoded_unit,
                core_blueprint,
                paper_layout=paper_layout,
                upstream_layout=upstream_layout,
            )
            paper_srl = float(profiles["paper_v3"][0].detach().cpu())
            local_upstream_tensor = profiles["upstream_code_e24ea73"][0]
            official_upstream_tensor = official_upstream_srl(
                decoded_unit,
                upstream_target,
            )
            (
                upstream_srl,
                upstream_reference_abs_error,
                upstream_reference_rel_error,
            ) = _assert_upstream_reference_match(
                local=local_upstream_tensor,
                official=official_upstream_tensor,
                config=config,
                branch=branch,
                iteration=iteration,
                phase="evaluation",
            )
            lpips_loss = float(
                lpips_model(
                    decoded_core.to(device="cpu", dtype=lpips_dtype),
                    reference_lpips,
                )
                .mean()
                .detach()
                .cpu()
            )
        selected_srl = paper_srl if selected_profile == "paper_v3" else upstream_srl
        objective = selected_srl + config.lpips_weight * lpips_loss
        diagnostics = module_diagnostics(
            decoded_unit,
            core_blueprint,
            paper_layout=paper_layout,
            upstream_layout=upstream_layout,
        )
        diagnostics.update(
            {
                "paper_active_modules": int(
                    profiles["paper_v3"][1]["active_modules"].detach().cpu()
                ),
                "upstream_active_modules": int(
                    profiles["upstream_code_e24ea73"][1]["active_modules"]
                    .detach()
                    .cpu()
                ),
                "upstream_reference_class": (
                    "diffqrcoder.losses.scanning_robust_loss.ScanningRobustLoss"
                ),
                "upstream_reference_revision": UPSTREAM_REVISION,
                "upstream_reference_official_loss": upstream_srl,
                "upstream_reference_local_loss": float(
                    local_upstream_tensor.detach().cpu()
                ),
                "upstream_reference_absolute_error": upstream_reference_abs_error,
                "upstream_reference_relative_error": upstream_reference_rel_error,
                "upstream_reference_match": True,
            }
        )
        maps = module_error_maps(
            decoded_unit,
            core_blueprint,
            paper_layout=paper_layout,
            upstream_layout=upstream_layout,
        )
        for map_name, values in maps.items():
            _save_error_map(
                maps_root / f"iteration-{iteration:03d}-{map_name}.png",
                values,
                functional,
            )

        selected_srl_image_gradient_rms = None
        lpips_image_gradient_rms = None
        weighted_lpips_image_gradient_rms = None
        objective_image_gradient_rms = None
        latent_gradient_rms = None
        requested_step_rms = None
        applied_step_rms = None
        effective_gradient_scale = None
        gradient_gate = None
        next_working = None

        if iteration < config.max_iterations:
            srl_core = decoded_core.detach().requires_grad_(True)
            srl_unit = (srl_core / 2 + 0.5).clamp(0, 1)
            srl_profiles = evaluate_loss_profiles(
                srl_unit,
                core_blueprint,
                paper_layout=paper_layout,
                upstream_layout=upstream_layout,
            )
            if selected_profile == "paper_v3":
                selected_loss = srl_profiles[selected_profile][0]
            else:
                local_selected_loss = srl_profiles[selected_profile][0]
                selected_loss = official_upstream_srl(srl_unit, upstream_target)
                _assert_upstream_reference_match(
                    local=local_selected_loss,
                    official=selected_loss,
                    config=config,
                    branch=branch,
                    iteration=iteration,
                    phase="gradient",
                )
            selected_srl_gradient = torch.autograd.grad(
                selected_loss,
                srl_core,
                only_inputs=True,
                allow_unused=True,
            )[0]
            selected_srl_image_gradient_rms = (
                0.0
                if selected_srl_gradient is None
                else float(
                    selected_srl_gradient.square().mean().sqrt().detach().cpu()
                )
            )
            del srl_profiles, selected_loss, srl_unit, srl_core

            lpips_core = (
                decoded_core.detach()
                .to(device="cpu", dtype=lpips_dtype)
                .requires_grad_(True)
            )
            lpips_value_tensor = lpips_model(lpips_core, reference_lpips).mean()
            lpips_gradient_cpu = torch.autograd.grad(
                lpips_value_tensor,
                lpips_core,
                only_inputs=True,
                allow_unused=True,
            )[0]
            lpips_image_gradient_rms = (
                0.0
                if lpips_gradient_cpu is None
                else float(lpips_gradient_cpu.square().mean().sqrt().detach().cpu())
            )
            weighted_lpips_image_gradient_rms = (
                config.lpips_weight * lpips_image_gradient_rms
            )
            del lpips_value_tensor, lpips_core

            objective_gradient = torch.zeros_like(decoded_core)
            if selected_srl_gradient is not None:
                objective_gradient.add_(selected_srl_gradient)
            if lpips_gradient_cpu is not None and config.lpips_weight:
                objective_gradient.add_(
                    lpips_gradient_cpu.to(device=device, dtype=objective_gradient.dtype),
                    alpha=config.lpips_weight,
                )
            objective_image_gradient_rms = float(
                objective_gradient.square().mean().sqrt().detach().cpu()
            )
            del selected_srl_gradient, lpips_gradient_cpu

            gradient = None
            if torch.isfinite(objective_gradient).all():
                for scale in _gradient_scales(config.gradient_scale):
                    candidate = working.detach().requires_grad_(True)
                    candidate_decoded = _decode_latent_tensor(pipeline, candidate).float()
                    candidate_core = _crop_core(
                        candidate_decoded,
                        config.crop_padding_px,
                    )
                    candidate_gradient = torch.autograd.grad(
                        candidate_core,
                        candidate,
                        grad_outputs=(
                            objective_gradient.to(dtype=candidate_core.dtype) * scale
                        ),
                        only_inputs=True,
                    )[0] / scale
                    del candidate_core, candidate_decoded, candidate
                    if torch.isfinite(candidate_gradient).all():
                        gradient = candidate_gradient
                        effective_gradient_scale = scale
                        break
                    del candidate_gradient
            del objective_gradient
            if gradient is None:
                raise RuntimeError(
                    f"{branch} produced no finite latent gradient at iteration {iteration}"
                )
            latent_gradient_rms = float(
                gradient.square().mean().sqrt().detach().cpu()
            )
            requested_step_rms = config.step_size * latent_gradient_rms
            next_working = working - config.step_size * gradient
            applied_step_rms = float(
                (next_working.detach() - working).square().mean().sqrt().cpu()
            )
            gradient_gate = combined_gradient_gate(
                selected_srl=selected_srl,
                selected_srl_image_gradient_rms=selected_srl_image_gradient_rms,
                objective_image_gradient_rms=objective_image_gradient_rms,
                latent_gradient_rms=latent_gradient_rms,
                applied_step_rms=applied_step_rms,
                gradient_tolerance=config.min_gradient_rms,
                loss_zero_tolerance=config.min_gradient_rms,
            )
            if not gradient_gate["passed"]:
                raise RuntimeError(
                    f"{branch} corrected gradient gate failed at iteration {iteration}: "
                    f"{gradient_gate}"
                )
            del gradient

        visual_change = {
            key: float(value)
            for key, value in image_change_metrics(image, parent.image).items()
        }
        latent_delta_rms = float(
            (working.detach() - initial).square().mean().sqrt().cpu()
        )
        step = E035Step(
            branch=branch,
            iteration=iteration,
            elapsed_s=time.perf_counter() - started,
            image_sha256=_image_sha256(image),
            latent_sha256=tensor_sha256(working),
            selected_srl_profile=selected_profile,
            selected_srl=selected_srl,
            paper_srl=paper_srl,
            upstream_srl=upstream_srl,
            lpips_loss=lpips_loss,
            objective=objective,
            selected_srl_image_gradient_rms=selected_srl_image_gradient_rms,
            lpips_image_gradient_rms=lpips_image_gradient_rms,
            weighted_lpips_image_gradient_rms=weighted_lpips_image_gradient_rms,
            objective_image_gradient_rms=objective_image_gradient_rms,
            latent_gradient_rms=latent_gradient_rms,
            requested_step_rms=requested_step_rms,
            applied_step_rms=applied_step_rms,
            latent_delta_rms=latent_delta_rms,
            effective_gradient_scale=effective_gradient_scale,
            gradient_gate_passed=(
                None if gradient_gate is None else bool(gradient_gate["passed"])
            ),
            gradient_gate=gradient_gate,
            diagnostics=diagnostics,
            visual_change=visual_change,
            cuda=_cuda_snapshot(),
        )
        steps.append(step)
        if next_working is not None:
            working = next_working.detach()

    final_image_path = images_root / f"iteration-{config.max_iterations:03d}.png"
    final_latent_path = branch_root / "final-latent.safetensors"
    save_file(
        {"latent": working.detach().cpu().contiguous()},
        str(final_latent_path),
        metadata={
            "experiment": EXPERIMENT,
            "branch": branch,
            "parent_tensor_sha256": parent.metadata["files"]["latent"][
                "tensor_sha256"
            ],
        },
    )
    trace_path = branch_root / "trace.json"
    _atomic_json(trace_path, [asdict(step) for step in steps])
    _write_trace_csv(branch_root / "trace.csv", steps)
    _atomic_json(
        branch_root / "branch-result.json",
        {
            "experiment": EXPERIMENT,
            "branch": branch,
            "selected_srl_profile": selected_profile,
            "selected_iteration": config.max_iterations,
            "parent_latent_sha256": parent.metadata["files"]["latent"][
                "tensor_sha256"
            ],
            "final_latent_tensor_sha256": tensor_sha256(working),
            "final_latent_file_sha256": sha256_file(final_latent_path),
            "final_image_sha256": steps[-1].image_sha256,
            "final_step": asdict(steps[-1]),
        },
    )
    del official_upstream_srl, upstream_target, reference_lpips, lpips_model
    return E035BranchResult(
        name=branch,
        selected_srl_profile=selected_profile,
        output_dir=str(branch_root),
        final_image_path=str(final_image_path),
        final_latent_path=str(final_latent_path),
        trace_path=str(trace_path),
        selected_iteration=config.max_iterations,
        final_step=asdict(steps[-1]),
    )


def _score_qr_verify(
    output_root: Path,
    payload: str,
    images: dict[str, Image.Image],
) -> dict[str, Any]:
    from .validation import (
        ConservativeQRVerifyScorer,
        canonical_conservative_qr_verify_evidence,
    )

    scorer = ConservativeQRVerifyScorer(
        repetitions=3,
        cache_dir=output_root / "qr-verify-cache",
    )
    results: dict[str, Any] = {}
    try:
        for name, image in images.items():
            score = scorer.score(image, payload)
            results[name] = canonical_conservative_qr_verify_evidence(score)
    finally:
        close = getattr(scorer.decoder, "close", None)
        if callable(close):
            close()
    _atomic_json(output_root / "qr-verify-evidence.json", results)
    return results


def _visual_guard(step: dict[str, Any]) -> dict[str, Any]:
    changes = step["visual_change"]
    checks = {
        "mean_absolute_change": changes.get("mean_absolute_change", 1.0) <= 0.04,
        "clipped_pixel_ratio_increase": (
            changes.get("clipped_pixel_ratio_increase", 1.0) <= 0.01
        ),
        "saturation_mean_increase": changes.get("saturation_mean_increase", 1.0)
        <= 0.04,
        "high_saturation_ratio_increase": (
            changes.get("high_saturation_ratio_increase", 1.0) <= 0.05
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _decision(
    parent_diagnostics: dict[str, Any],
    branches: dict[str, E035BranchResult],
    qr_verify: dict[str, Any] | None,
) -> dict[str, Any]:
    upstream = branches.get(BRANCH_UPSTREAM)
    paper = branches.get(BRANCH_PAPER)
    if upstream is None:
        return {
            "decision": "INCOMPLETE_UPSTREAM_BRANCH_MISSING",
            "production_ready": False,
        }
    upstream_final = upstream.final_step
    upstream_diag = upstream_final["diagnostics"]
    visual = _visual_guard(upstream_final)
    upstream_qr = (qr_verify or {}).get(BRANCH_UPSTREAM) or {}
    exact = int(upstream_qr.get("conservative_exact_presets", 0))
    margin_improved = (
        upstream_diag["upstream_margin_active_count"]
        < parent_diagnostics["upstream_margin_active_count"]
    )
    full_mer_improved = (
        upstream_diag["full_module_error_count"]
        < parent_diagnostics["full_module_error_count"]
    )
    if exact >= 1 and visual["passed"]:
        decision = "GO_MINI_HOLDOUT_8_TO_12_PROMPTS"
    elif margin_improved or full_mer_improved:
        decision = "E036_HYBRID_CENTER_PLUS_FULL_MODULE_LOSS"
    else:
        decision = "STOP_AND_DIAGNOSE_DETECTOR_GEOMETRY"
    return {
        "decision": decision,
        "production_ready": False,
        "automatic_expansion_authorized": False,
        "mini_holdout_authorized": decision.startswith("GO_MINI_HOLDOUT"),
        "advisor_training_authorized": False,
        "upstream_conservative_exact_presets": exact,
        "upstream_visual_guard": visual,
        "upstream_margin_improved": margin_improved,
        "upstream_full_module_error_improved": full_mer_improved,
        "paper_branch_available": paper is not None,
    }


def _contact_sheet(
    output_path: Path,
    parent: Image.Image,
    paper: Image.Image | None,
    upstream: Image.Image | None,
) -> None:
    items = [("Parent immuable", parent)]
    if paper is not None:
        items.append(("E035 paper_v3 i4", paper))
    if upstream is not None:
        items.append(("E035 upstream e24ea73 i4", upstream))
    width = 420
    tile_height = 470
    sheet = Image.new("RGB", (width * len(items), tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        preview = image.convert("RGB").copy()
        preview.thumbnail((400, 400), Image.Resampling.LANCZOS)
        x = index * width + (width - preview.width) // 2
        y = 45 + (400 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((index * width + 12, 12), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=False, compress_level=9)


def _load_pipeline() -> tuple[Any, Any]:
    from .config import Settings
    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .lab import DIFFQRCODER_MODEL_SETTINGS

    base = Settings()
    settings = Settings.model_validate(
        {
            **base.model_dump(),
            **DIFFQRCODER_MODEL_SETTINGS,
            "device": "cuda",
            "diffqrcoder_upstream_enabled": True,
        }
    )
    backend = UpstreamDiffQRCoderBackend(settings)
    return backend, backend._load()


def run_e035(
    *,
    parent_dir: Path,
    output_dir: Path,
    branches: tuple[BranchName, ...] = (BRANCH_PAPER, BRANCH_UPSTREAM),
    config: E035Config | None = None,
    skip_qr_verify: bool = False,
    expected_parent_commit: str | None = None,
) -> dict[str, Any]:
    import torch

    from .qr import generate_diffqrcoder_qr

    if config is None:
        config = E035Config()
    if not torch.cuda.is_available():
        raise RuntimeError("E035 requires an available CUDA GPU")
    if config.max_iterations != 4:
        raise ValueError("E035 is frozen to exactly four updates")
    if config.step_size != 1000.0 or config.lpips_weight != 0.01:
        raise ValueError("E035 gamma and LPIPS weight are frozen to 1000 and 0.01")
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
    parent = load_parent_artifact(
        parent_dir,
        device="cpu",
        expected=expected_parent,
    )
    source = parent.metadata["source"]
    source_method = str(source.get("source_method_id") or "")
    if source_method == "e033_public_demo_srpg_exact_e034_export":
        if source.get("parent_origin") != "exact_e034_stage2_export":
            raise ValueError(
                "an exact E034 parent must declare "
                "parent_origin=exact_e034_stage2_export"
            )
    elif source_method == "e033_public_demo_srpg_from_fixed_e034_stage1":
        if source.get("parent_origin") != "stage2_replayed_from_exact_e034_stage1":
            raise ValueError("fallback parent origin is missing or inconsistent")
        if source.get("stage1_image_sha256") != (
            "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
        ):
            raise ValueError("fallback parent does not reference the fixed E034 Stage-1")
        if source.get("generation", {}).get("stage1_regenerated") is not False:
            raise ValueError("fallback parent must declare stage1_regenerated=false")
    else:
        raise ValueError(f"unsupported E035 parent source_method_id: {source_method!r}")
    payload = str(source["payload"])
    blueprint = generate_diffqrcoder_qr(
        payload,
        str(source["error_correction"]),
        version=config.qr_version,
        mask_pattern=config.qr_mask_pattern,
        module_size=config.qr_module_size,
    )
    if parent.image.size != (736, 736):
        raise ValueError(
            f"E035 expects the public 736x736 Stage-2 raster, got {parent.image.size}"
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"E035 output directory must be empty to preserve immutability: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_dir / "plan.json",
        {
            "experiment": EXPERIMENT,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "branches": list(branches),
            "parent_contract_sha256": parent.metadata["contract_sha256"],
            "parent_latent_tensor_sha256": parent.metadata["files"]["latent"][
                "tensor_sha256"
            ],
            "parent_image_file_sha256": parent.metadata["files"]["image"]["sha256"],
            "automatic_expansion_authorized": False,
            "production_ready": False,
        },
    )
    _atomic_json(output_dir / "parent-verification.json", parent.metadata)

    backend, pipeline = _load_pipeline()
    loaded_scaling_factor = float(pipeline.vae.config.scaling_factor)
    source_scaling_factor = float(source["vae_scaling_factor"])
    if not math.isclose(
        loaded_scaling_factor,
        source_scaling_factor,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "parent VAE scaling factor does not match the loaded pipeline: "
            f"{source_scaling_factor} != {loaded_scaling_factor}"
        )
    for field in ("base_model_id", "base_model_revision", "diffqrcoder_revision"):
        loaded_value = str(getattr(backend.settings, field))
        source_value = str(source[field])
        if loaded_value != source_value:
            raise ValueError(
                f"parent {field} does not match the loaded pipeline: "
                f"{source_value!r} != {loaded_value!r}"
            )
    original_vae_dtype = next(pipeline.vae.parameters()).dtype
    checkpointing_was_enabled = bool(
        getattr(pipeline.vae, "is_gradient_checkpointing", False)
    )
    enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
    disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
    branch_results: dict[str, E035BranchResult] = {}
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
                        "vae_scaling_factor": loaded_scaling_factor,
                        "base_model_id": str(backend.settings.base_model_id),
                        "base_model_revision": str(backend.settings.base_model_revision),
                        "diffqrcoder_revision": str(backend.settings.diffqrcoder_revision),
                        "parent_source_commit": str(source["source_commit"]),
                    },
                )
                for branch in branches:
                    branch_results[branch] = _run_branch(
                        pipeline=pipeline,
                        parent=parent,
                        blueprint=blueprint,
                        config=config,
                        branch=branch,
                        output_root=output_dir,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                # Restore the compact VAE *before* diffusion modules return to CUDA.
                # Reversing this order can exceed a 20 GiB RTX during context exit.
                pipeline.vae.to(dtype=original_vae_dtype)
                if not checkpointing_was_enabled and callable(disable_checkpointing):
                    disable_checkpointing()
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        # Idempotent safety net for partial failures during promotion/restoration.
        if next(pipeline.vae.parameters()).dtype != original_vae_dtype:
            pipeline.vae.to(dtype=original_vae_dtype)
        gc.collect()
        torch.cuda.empty_cache()

    if not branch_results:
        raise RuntimeError("E035 produced no branch result")
    # Iteration zero is the exact common parent latent decode for every branch. Reuse
    # that recorded evidence instead of decoding again after diffusion weights return
    # to CUDA, which would create an avoidable memory peak.
    first_trace_path = Path(next(iter(branch_results.values())).trace_path)
    first_trace = json.loads(first_trace_path.read_text(encoding="utf-8"))
    parent_diagnostics = dict(first_trace[0]["diagnostics"])
    _atomic_json(output_dir / "parent-module-diagnostics.json", parent_diagnostics)

    initial_branch_states: dict[str, dict[str, str]] = {}
    for name, result in branch_results.items():
        trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
        if not trace or trace[0]["iteration"] != 0:
            raise RuntimeError(f"{name} has no canonical iteration-zero state")
        initial_branch_states[name] = {
            "latent_sha256": str(trace[0]["latent_sha256"]),
            "image_sha256": str(trace[0]["image_sha256"]),
        }
    initial_latent_hashes = {
        value["latent_sha256"] for value in initial_branch_states.values()
    }
    initial_image_hashes = {
        value["image_sha256"] for value in initial_branch_states.values()
    }
    source_parent_raster_sha256 = _image_sha256(parent.image)
    fp32_parent_redecoded_path = Path(next(iter(branch_results.values())).output_dir) / (
        "images/iteration-000.png"
    )
    fp32_parent_redecoded = Image.open(fp32_parent_redecoded_path).convert("RGB")
    fp32_parent_redecoded_sha256 = _image_sha256(fp32_parent_redecoded)
    fp32_parent_redecoded.save(
        output_dir / "parent-fp32-redecoded.png",
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    pairing = {
        "passed": len(initial_latent_hashes) == 1 and len(initial_image_hashes) == 1,
        "branch_initial_states": initial_branch_states,
        "source_parent_tensor_sha256": parent.metadata["files"]["latent"][
            "tensor_sha256"
        ],
        "source_parent_image_raster_sha256": source_parent_raster_sha256,
        "fp32_parent_redecoded_sha256": fp32_parent_redecoded_sha256,
        "note": (
            "The source parent may be FP16 while Eq. 14 uses identical FP32 working "
            "copies. Pairing is proven on the post-conversion iteration-zero latent "
            "and on the independently saved FP32 D(z0) raster for every branch."
        ),
    }
    if not pairing["passed"]:
        raise RuntimeError(f"E035 branch pairing failed: {pairing}")
    _atomic_json(output_dir / "branch-pairing.json", pairing)

    final_images = {
        name: Image.open(result.final_image_path).convert("RGB")
        for name, result in branch_results.items()
    }
    qr_images = {
        "parent_source": parent.image,
        "parent_fp32_redecoded": fp32_parent_redecoded,
        **final_images,
    }
    qr_verify = None if skip_qr_verify else _score_qr_verify(output_dir, payload, qr_images)
    if skip_qr_verify:
        _atomic_json(
            output_dir / "qr-verify-evidence.json",
            {"skipped": True, "reason": "--skip-qr-verify explicitly supplied"},
        )

    verdict = _decision(parent_diagnostics, branch_results, qr_verify)
    verdict.update(
        {
            "experiment": EXPERIMENT,
            "parent_contract_sha256": parent.metadata["contract_sha256"],
            "branches": {
                name: asdict(result) for name, result in branch_results.items()
            },
            "qr_verify_skipped": skip_qr_verify,
            "branch_pairing": pairing,
        }
    )
    _atomic_json(output_dir / "verdict.json", verdict)
    _contact_sheet(
        output_dir / "e035-final-contact-sheet.png",
        parent.image,
        final_images.get(BRANCH_PAPER),
        final_images.get(BRANCH_UPSTREAM),
    )
    report = f"""# E035 — SR-MPGD loss fidelity gate

- Parent contract: `{parent.metadata['contract_sha256']}`
- Parent latent: `{parent.metadata['files']['latent']['tensor_sha256']}`
- Branches: {', '.join(branch_results)}
- QR-Verify: {'skipped explicitly' if skip_qr_verify else '37 presets × 3 repetitions'}
- Decision: **{verdict['decision']}**
- Production ready: **non**
- Automatic advisor training: **non**

The experiment changes only the SRL profile. Both branches start from the exact same
verified Stage-2 latent, use a FP32 VAE, gamma 1000, neutral gradient scaling 32768,
LPIPS-VGG on CPU with weight 0.01, and report the fixed fourth update.
"""
    _atomic_text(output_dir / "report.md", report)
    del backend
    return verdict


def _parse_branches(value: str) -> tuple[BranchName, ...]:
    if value == "both":
        return (BRANCH_PAPER, BRANCH_UPSTREAM)
    if value == "paper":
        return (BRANCH_PAPER,)
    if value == "upstream":
        return (BRANCH_UPSTREAM,)
    raise ValueError(value)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--branches",
        choices=("both", "paper", "upstream"),
        default="both",
    )
    parser.add_argument("--skip-qr-verify", action="store_true")
    parser.add_argument(
        "--expected-parent-commit",
        default=None,
        help=(
            "Fail closed unless the immutable parent source commit matches this SHA. "
            "The deployment script reads and supplies the commit from the verified contract."
        ),
    )
    args = parser.parse_args()
    verdict = run_e035(
        parent_dir=args.parent_dir,
        output_dir=args.output_dir,
        branches=_parse_branches(args.branches),
        skip_qr_verify=args.skip_qr_verify,
        expected_parent_commit=args.expected_parent_commit,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
