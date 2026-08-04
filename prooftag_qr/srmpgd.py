from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .guidance import (
    crop_tensor_to_qr_core,
    prepare_torch_layout,
    qr_core_geometry,
    scanning_robust_loss,
)
from .qr import (
    QRBlueprint,
    diffqrcoder_module_error_rate,
    module_error_rate,
    prepare_scan_ready_image,
)
from .quality import image_change_metrics


@dataclass(frozen=True, slots=True)
class SRMPGDConfig:
    """Paper-aligned Scanning-Robust Manifold Projected Gradient Descent.

    The WACV 2025 paper specifies ``gamma=1000`` and an LPIPS coefficient of ``0.01``.
    It does not publish the iteration count, so the implementation evaluates and persists every
    state up to ``max_iterations`` and stops as soon as the external validation gate is strict.
    """

    max_iterations: int = 4
    step_size: float = 100.0
    lpips_weight: float = 0.10
    lpips_net: str = "vgg"
    crop_padding_px: int = -1
    dark_threshold: float = 0.5
    light_threshold: float = 0.5
    center_fraction: float = 1 / 3
    max_initial_module_error_rate: float = 0.10
    max_step_rms: float = 0.02
    max_total_delta_rms: float = 0.06
    min_relative_module_improvement: float = 0.01
    max_lpips_loss: float = 0.15
    max_mean_absolute_change: float = 0.06
    max_saturation_mean_increase: float = 0.04
    max_high_saturation_ratio_increase: float = 0.05
    max_rgb_clipped_channel_ratio_increase: float = 0.01
    robust_blur_weight: float = 0.0
    robust_blur_kernel: int = 3
    robust_downscale_weight: float = 0.0
    robust_downscale_factor: float = 0.75
    robust_brightness_weight: float = 0.0
    robust_brightness_low: float = 0.80
    robust_brightness_high: float = 1.20
    robust_contrast_weight: float = 0.0
    robust_contrast_factor: float = 0.75
    quiet_zone_mode: str = "adaptive_light"
    quiet_zone_minimum_luminance: float = 0.90
    functional_pattern_tone_factor: float = 0.0


@dataclass(frozen=True, slots=True)
class SRMPGDStep:
    iteration: int
    elapsed_s: float
    scanning_robust_loss: float
    lpips_loss: float
    objective: float
    surrogate_module_error_rate: float
    actual_module_error_rate: float
    passed: int
    total: int
    pass_rate: float
    strict_all: bool
    worst_decoder_pass_rate: float
    worst_scenario_pass_rate: float
    gradient_rms: float | None
    next_step_rms: float | None
    applied_step_rms: float | None
    step_scale: float | None
    latent_delta_rms: float
    relative_module_improvement: float
    mean_absolute_change: float
    saturation_mean_increase: float
    high_saturation_ratio_increase: float
    rgb_clipped_channel_ratio_increase: float
    aesthetic_guard_passed: bool
    qr_gain_sufficient: bool
    eligible_for_selection: bool
    base_scanning_loss: float
    blur_scanning_loss: float | None
    downscale_scanning_loss: float | None
    brightness_scanning_loss: float | None
    contrast_scanning_loss: float | None


@dataclass(frozen=True, slots=True)
class SRMPGDResult:
    image: Image.Image
    latent: Any
    steps: tuple[SRMPGDStep, ...]
    selected_iteration: int
    stop_reason: str
    duration_s: float
    initial_module_error_rate: float
    final_module_error_rate: float


ValidationCallback = Callable[[Image.Image, int], Mapping[str, Any]]
PreviewCallback = Callable[[Image.Image, SRMPGDStep], None]
ScanningLoss = Callable[[Any, Any], Any]


def _validate_config(config: SRMPGDConfig) -> None:
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if config.step_size <= 0:
        raise ValueError("step_size must be positive")
    if config.lpips_weight < 0:
        raise ValueError("lpips_weight cannot be negative")
    if config.lpips_net not in {"alex", "vgg", "squeeze"}:
        raise ValueError("lpips_net must be alex, vgg or squeeze")
    if config.crop_padding_px < -1:
        raise ValueError("crop_padding_px must be -1 (automatic) or non-negative")
    if not 0 < config.dark_threshold <= config.light_threshold < 1:
        raise ValueError("thresholds must satisfy 0 < dark <= light < 1")
    if not 0 < config.center_fraction <= 1:
        raise ValueError("center_fraction must be between 0 (exclusive) and 1")
    if not 0 <= config.max_initial_module_error_rate <= 1:
        raise ValueError("max_initial_module_error_rate must be between 0 and 1")
    if config.max_step_rms <= 0:
        raise ValueError("max_step_rms must be positive")
    if config.max_total_delta_rms <= 0:
        raise ValueError("max_total_delta_rms must be positive")
    if config.max_step_rms > config.max_total_delta_rms:
        raise ValueError("max_step_rms cannot exceed max_total_delta_rms")
    if not 0 <= config.min_relative_module_improvement <= 1:
        raise ValueError("min_relative_module_improvement must be between 0 and 1")
    if config.max_lpips_loss < 0:
        raise ValueError("max_lpips_loss cannot be negative")
    for name in (
        "max_mean_absolute_change",
        "max_saturation_mean_increase",
        "max_high_saturation_ratio_increase",
        "max_rgb_clipped_channel_ratio_increase",
    ):
        if not 0 <= getattr(config, name) <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name in (
        "robust_blur_weight",
        "robust_downscale_weight",
        "robust_brightness_weight",
        "robust_contrast_weight",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if config.robust_blur_kernel < 1 or config.robust_blur_kernel % 2 == 0:
        raise ValueError("robust_blur_kernel must be a positive odd integer")
    if not 0 < config.robust_downscale_factor <= 1:
        raise ValueError("robust_downscale_factor must be between 0 and 1")
    if not 0 < config.robust_brightness_low <= 1:
        raise ValueError("robust_brightness_low must be between 0 and 1")
    if not 1 <= config.robust_brightness_high <= 2:
        raise ValueError("robust_brightness_high must be between 1 and 2")
    if not 0 < config.robust_contrast_factor <= 1:
        raise ValueError("robust_contrast_factor must be between 0 and 1")
    if config.quiet_zone_mode not in {"none", "white", "adaptive_light"}:
        raise ValueError("quiet_zone_mode must be none, white or adaptive_light")
    if not 0.0 < config.quiet_zone_minimum_luminance <= 1.0:
        raise ValueError("quiet_zone_minimum_luminance must be between 0 and 1")
    if not 0.0 <= config.functional_pattern_tone_factor <= 1.0:
        raise ValueError("functional_pattern_tone_factor must be between 0 and 1")


def _load_lpips(pipeline: Any, *, device: Any, net: str) -> Any:
    cache_name = f"_prooftag_srmpgd_lpips_{net}"
    cached = getattr(pipeline, cache_name, None)
    if cached is not None:
        return cached.to(device=device)
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Install lpips==0.1.4 to run paper-aligned SR-MPGD") from exc
    model = lpips.LPIPS(net=net, verbose=False)
    model.requires_grad_(False).eval().to(device=device)
    setattr(pipeline, cache_name, model)
    return model


def _crop_tensor(tensor: Any, padding: int) -> Any:
    if padding == 0:
        return tensor
    if tensor.shape[-2] <= 2 * padding or tensor.shape[-1] <= 2 * padding:
        raise ValueError("crop padding removes the complete decoded image")
    return tensor[..., padding:-padding, padding:-padding]


def _blueprint_tensor(
    blueprint: QRBlueprint,
    *,
    height: int,
    width: int,
    device: Any,
    strip_border: bool,
) -> Any:
    import numpy as np
    import torch

    border = blueprint.border
    matrix = (
        blueprint.matrix[border:-border, border:-border]
        if strip_border and border
        else blueprint.matrix
    )
    binary = np.where(matrix, 0, 255).astype(np.uint8)
    resized = Image.fromarray(binary, mode="L").resize(
        (width, height), Image.Resampling.NEAREST
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.as_tensor(array, device=device).unsqueeze(0).unsqueeze(0)


def _core_blueprint(
    blueprint: QRBlueprint,
    *,
    height: int,
    width: int,
    strip_border: bool,
) -> QRBlueprint:
    border = blueprint.border
    matrix = (
        blueprint.matrix[border:-border, border:-border].copy()
        if strip_border and border
        else blueprint.matrix.copy()
    )
    return QRBlueprint(
        image=Image.new("RGB", (width, height), "white"),
        matrix=matrix,
        version=blueprint.version,
        border=0 if strip_border else border,
    )


def _decode_latent(
    pipeline: Any,
    latent: Any,
    *,
    blueprint: QRBlueprint,
    config: SRMPGDConfig,
) -> tuple[Any, Image.Image]:
    import torch

    vae = pipeline.vae
    scaling_factor = vae.config.scaling_factor
    if latent.device.type == "cuda":
        with torch.autocast("cuda", dtype=next(vae.parameters()).dtype):
            decoded = vae.decode(latent / scaling_factor, return_dict=False)[0]
    else:
        decoded = vae.decode(latent / scaling_factor, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(
        decoded.detach(), output_type="pil", do_denormalize=[True]
    )[0].convert("RGB")
    return decoded, prepare_scan_ready_image(
        image,
        blueprint,
        quiet_zone_mode=config.quiet_zone_mode,
        quiet_zone_minimum_luminance=config.quiet_zone_minimum_luminance,
        functional_pattern_tone_factor=config.functional_pattern_tone_factor,
    )


def _module_error_for_canvas(
    image: Image.Image,
    blueprint: QRBlueprint,
    *,
    crop_padding_px: int,
) -> float:
    """Measure modules on the same core crop used by the differentiable objective."""
    if crop_padding_px == 0:
        return module_error_rate(image, blueprint)
    core_modules = blueprint.matrix.shape[0] - 2 * blueprint.border
    core_size = image.width - 2 * crop_padding_px
    if core_modules <= 0 or core_size % core_modules:
        raise ValueError("QR core does not have an integer module geometry")
    return diffqrcoder_module_error_rate(
        image,
        blueprint,
        padding_px=crop_padding_px,
        module_size=core_size // core_modules,
    )


def _validation_values(values: Mapping[str, Any] | None) -> dict[str, Any]:
    values = values or {}
    passed = int(values.get("passed", 0))
    total = int(values.get("total", 0))
    return {
        "passed": passed,
        "total": total,
        "pass_rate": float(values.get("pass_rate", passed / total if total else 0.0)),
        "strict_all": bool(values.get("strict_all", total > 0 and passed == total)),
        "worst_decoder_pass_rate": float(values.get("worst_decoder_pass_rate", 0.0)),
        "worst_scenario_pass_rate": float(values.get("worst_scenario_pass_rate", 0.0)),
    }


def _rank_step(step: SRMPGDStep) -> tuple[Any, ...]:
    return (
        step.strict_all,
        step.pass_rate,
        step.worst_decoder_pass_rate,
        step.worst_scenario_pass_rate,
        -step.lpips_loss,
        -step.mean_absolute_change,
        -step.actual_module_error_rate,
        -step.scanning_robust_loss,
        -step.iteration,
    )


def _robust_scanning_loss(
    images: Any,
    target: Any,
    scanning_loss: ScanningLoss,
    config: SRMPGDConfig,
) -> tuple[Any, dict[str, Any | None]]:
    """Average the public DiffQRCoder SRL over differentiable scan degradations."""
    import torch.nn.functional as functional

    base = scanning_loss(images, target)
    if base.ndim != 0:
        base = base.mean()
    total = base
    total_weight = 1.0
    components: dict[str, Any | None] = {
        "base": base,
        "blur": None,
        "downscale": None,
        "brightness": None,
        "contrast": None,
    }

    if config.robust_blur_weight:
        blurred = functional.avg_pool2d(
            images,
            kernel_size=config.robust_blur_kernel,
            stride=1,
            padding=config.robust_blur_kernel // 2,
        )
        blur = scanning_loss(blurred, target)
        if blur.ndim != 0:
            blur = blur.mean()
        components["blur"] = blur
        total = total + config.robust_blur_weight * blur
        total_weight += config.robust_blur_weight

    if config.robust_downscale_weight:
        reduced = functional.interpolate(
            images,
            scale_factor=config.robust_downscale_factor,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        restored = functional.interpolate(
            reduced,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        downscale = scanning_loss(restored, target)
        if downscale.ndim != 0:
            downscale = downscale.mean()
        components["downscale"] = downscale
        total = total + config.robust_downscale_weight * downscale
        total_weight += config.robust_downscale_weight

    if config.robust_brightness_weight:
        brightness = images.new_tensor(0.0)
        for factor in (config.robust_brightness_low, config.robust_brightness_high):
            value = scanning_loss((images * factor).clamp(0, 1), target)
            if value.ndim != 0:
                value = value.mean()
            brightness = brightness + value / 2
        components["brightness"] = brightness
        total = total + config.robust_brightness_weight * brightness
        total_weight += config.robust_brightness_weight

    if config.robust_contrast_weight:
        contrasted = (
            (images - 0.5) * config.robust_contrast_factor + 0.5
        ).clamp(0, 1)
        contrast = scanning_loss(contrasted, target)
        if contrast.ndim != 0:
            contrast = contrast.mean()
        components["contrast"] = contrast
        total = total + config.robust_contrast_weight * contrast
        total_weight += config.robust_contrast_weight

    return total / total_weight, components


def run_srmpgd(
    pipeline: Any,
    initial_latent: Any,
    blueprint: QRBlueprint,
    config: SRMPGDConfig,
    *,
    scanning_loss: ScanningLoss | None = None,
    validation_callback: ValidationCallback | None = None,
    preview_callback: PreviewCallback | None = None,
) -> SRMPGDResult:
    """Optimize the exact clean Stage-2 latent with Eq. 13-14 from DiffQRCoder.

    No image is encoded in this function: ``initial_latent`` must be the clean latent returned by
    Stage-2. The SRL target is the original QR blueprint, whereas LPIPS is measured against the
    detached image decoded from that same initial latent. Stage-2 SRPG weights are intentionally
    absent because Eq. 13 defines a separate ``SRL + 0.01 * LPIPS`` objective.
    """
    import torch

    _validate_config(config)
    if not torch.is_tensor(initial_latent) or initial_latent.ndim != 4:
        raise ValueError("initial_latent must be a BCHW torch tensor")
    if initial_latent.shape[0] != 1:
        raise ValueError("SR-MPGD currently requires batch size one")

    vae = pipeline.vae
    vae.requires_grad_(False).eval()
    device = initial_latent.device
    initial = initial_latent.detach().to(dtype=torch.float32).clone()
    working = initial.clone()
    lpips_model = _load_lpips(pipeline, device=device, net=config.lpips_net)
    started = time.perf_counter()

    with torch.no_grad():
        reference_decoded, reference_image = _decode_latent(
            pipeline,
            working,
            blueprint=blueprint,
            config=config,
        )
        if config.crop_padding_px == -1:
            geometry = qr_core_geometry(
                blueprint,
                reference_decoded.shape[-2],
                reference_decoded.shape[-1],
            )
            reference_core = crop_tensor_to_qr_core(
                reference_decoded.float(),
                geometry,
            ).detach()
            core_blueprint = geometry.blueprint
            resolved_crop_padding_px = geometry.left
        else:
            reference_core = _crop_tensor(
                reference_decoded.float(),
                config.crop_padding_px,
            ).detach()
            core_blueprint = _core_blueprint(
                blueprint,
                height=reference_core.shape[-2],
                width=reference_core.shape[-1],
                strip_border=config.crop_padding_px > 0,
            )
            resolved_crop_padding_px = config.crop_padding_px
    core_height, core_width = reference_core.shape[-2:]
    target_core = _blueprint_tensor(
        core_blueprint,
        height=core_height,
        width=core_width,
        device=device,
        strip_border=False,
    )
    if target_core.shape[-2:] != (core_height, core_width):
        raise ValueError("QR target and decoded latent core dimensions do not match")
    layout = prepare_torch_layout(
        core_blueprint,
        core_height,
        core_width,
        device=device,
        dtype=torch.float32,
        center_fraction=config.center_fraction,
    )

    states: list[tuple[SRMPGDStep, Any, Image.Image]] = []
    stop_reason = "max_iterations"
    initial_module_error_rate = _module_error_for_canvas(
        reference_image,
        blueprint,
        crop_padding_px=resolved_crop_padding_px,
    )
    refinement_applicable = (
        initial_module_error_rate <= config.max_initial_module_error_rate
    )
    baseline_pass_rate = 0.0
    for iteration in range(config.max_iterations + 1):
        iteration_stop_reason = None
        with torch.enable_grad():
            working = working.detach().requires_grad_(True)
            decoded, image = _decode_latent(
                pipeline,
                working,
                blueprint=blueprint,
                config=config,
            )
            decoded_core = (
                crop_tensor_to_qr_core(decoded.float(), geometry)
                if config.crop_padding_px == -1
                else _crop_tensor(decoded.float(), config.crop_padding_px)
            )
            decoded_unit = (decoded_core / 2 + 0.5).clamp(0, 1)
            diagnostic_srl, diagnostics = scanning_robust_loss(
                decoded_unit,
                core_blueprint,
                functional_weight=1.0,
                center_fraction=config.center_fraction,
                dark_threshold=config.dark_threshold,
                light_threshold=config.light_threshold,
                layout=layout,
            )
            if scanning_loss is not None:
                srl, robust_components = _robust_scanning_loss(
                    decoded_unit,
                    target_core,
                    scanning_loss,
                    config,
                )
            else:
                srl = diagnostic_srl
                robust_components = {
                    "base": diagnostic_srl,
                    "blur": None,
                    "downscale": None,
                    "brightness": None,
                    "contrast": None,
                }
            lpips_parameter = next(iter(lpips_model.parameters()), None)
            lpips_dtype = (
                lpips_parameter.dtype if lpips_parameter is not None else decoded_core.dtype
            )
            lpips_loss = lpips_model(
                decoded_core.to(dtype=lpips_dtype),
                reference_core.to(dtype=lpips_dtype),
            ).mean()
            objective = srl + config.lpips_weight * lpips_loss
            validation = _validation_values(
                validation_callback(image, iteration) if validation_callback else None
            )
            if iteration == 0:
                baseline_pass_rate = validation["pass_rate"]

            actual_module_error_rate = _module_error_for_canvas(
                image,
                blueprint,
                crop_padding_px=resolved_crop_padding_px,
            )
            relative_module_improvement = (
                (initial_module_error_rate - actual_module_error_rate)
                / max(initial_module_error_rate, 1e-8)
            )
            changes = image_change_metrics(image, reference_image)
            latent_delta_rms = float(
                (working.detach() - initial).square().mean().sqrt().cpu()
            )
            aesthetic_guard_passed = iteration == 0 or (
                float(lpips_loss.detach().cpu()) <= config.max_lpips_loss
                and latent_delta_rms <= config.max_total_delta_rms + 1e-8
                and changes["mean_absolute_change"]
                <= config.max_mean_absolute_change
                and changes["saturation_mean_increase"]
                <= config.max_saturation_mean_increase
                and changes["high_saturation_ratio_increase"]
                <= config.max_high_saturation_ratio_increase
                and changes["rgb_clipped_channel_ratio_increase"]
                <= config.max_rgb_clipped_channel_ratio_increase
            )
            qr_gain_sufficient = iteration == 0 or (
                validation["strict_all"]
                or validation["pass_rate"] > baseline_pass_rate
                or relative_module_improvement
                >= config.min_relative_module_improvement
            )
            eligible_for_selection = (
                aesthetic_guard_passed and qr_gain_sufficient
            )

            gradient = None
            gradient_rms = None
            next_step_rms = None
            applied_step_rms = None
            step_scale = None
            next_working = None
            if iteration > 0 and not aesthetic_guard_passed:
                iteration_stop_reason = (
                    f"aesthetic_guard_failed_at_iteration_{iteration}"
                )
            elif not refinement_applicable:
                iteration_stop_reason = "initial_module_error_rate_above_limit"
            elif not validation["strict_all"] and iteration < config.max_iterations:
                gradient = torch.autograd.grad(objective, working, only_inputs=True)[0]
                if not torch.isfinite(gradient).all():
                    iteration_stop_reason = (
                        f"non_finite_gradient_at_iteration_{iteration}"
                    )
                    gradient = None
                else:
                    gradient_rms = float(gradient.square().mean().sqrt().detach().cpu())
                    next_step_rms = config.step_size * gradient_rms
                    step_scale = min(
                        1.0,
                        config.max_step_rms / max(next_step_rms, 1e-12),
                    )
                    proposed = working - config.step_size * step_scale * gradient
                    delta = proposed.detach() - initial
                    total_delta_rms = float(delta.square().mean().sqrt().cpu())
                    if total_delta_rms > config.max_total_delta_rms:
                        delta = delta * (
                            config.max_total_delta_rms / total_delta_rms
                        )
                        proposed = initial + delta
                    next_working = proposed
                    applied_step_rms = float(
                        (next_working.detach() - working.detach())
                        .square()
                        .mean()
                        .sqrt()
                        .cpu()
                    )

            step = SRMPGDStep(
                iteration=iteration,
                elapsed_s=time.perf_counter() - started,
                scanning_robust_loss=float(srl.detach().cpu()),
                lpips_loss=float(lpips_loss.detach().cpu()),
                objective=float(objective.detach().cpu()),
                surrogate_module_error_rate=float(
                    diagnostics["module_error_rate"].detach().cpu()
                ),
                actual_module_error_rate=actual_module_error_rate,
                gradient_rms=gradient_rms,
                next_step_rms=next_step_rms,
                applied_step_rms=applied_step_rms,
                step_scale=step_scale,
                latent_delta_rms=latent_delta_rms,
                relative_module_improvement=relative_module_improvement,
                mean_absolute_change=changes["mean_absolute_change"],
                saturation_mean_increase=changes["saturation_mean_increase"],
                high_saturation_ratio_increase=changes[
                    "high_saturation_ratio_increase"
                ],
                rgb_clipped_channel_ratio_increase=changes[
                    "rgb_clipped_channel_ratio_increase"
                ],
                aesthetic_guard_passed=aesthetic_guard_passed,
                qr_gain_sufficient=qr_gain_sufficient,
                eligible_for_selection=eligible_for_selection,
                base_scanning_loss=float(
                    robust_components["base"].detach().cpu()
                ),
                blur_scanning_loss=(
                    float(robust_components["blur"].detach().cpu())
                    if robust_components["blur"] is not None
                    else None
                ),
                downscale_scanning_loss=(
                    float(robust_components["downscale"].detach().cpu())
                    if robust_components["downscale"] is not None
                    else None
                ),
                brightness_scanning_loss=(
                    float(robust_components["brightness"].detach().cpu())
                    if robust_components["brightness"] is not None
                    else None
                ),
                contrast_scanning_loss=(
                    float(robust_components["contrast"].detach().cpu())
                    if robust_components["contrast"] is not None
                    else None
                ),
                **validation,
            )
            states.append((step, working.detach().clone(), image.copy()))
            if preview_callback is not None:
                preview_callback(image, step)
            if step.strict_all and step.eligible_for_selection:
                stop_reason = "strict_validation_passed"
                break
            if iteration_stop_reason is not None:
                # A failed refinement must not abort a multi-hour search. State i is still a
                # valid decoded candidate and is ranked normally; the stop reason makes the
                # numerical failure explicit instead of silently replacing NaNs with zeros.
                stop_reason = iteration_stop_reason
                break
            if next_working is not None:
                working = next_working

    eligible_states = [
        item for item in states if item[0].eligible_for_selection
    ]
    selected_step, selected_latent, selected_image = max(
        eligible_states, key=lambda item: _rank_step(item[0])
    )
    return SRMPGDResult(
        image=selected_image,
        latent=selected_latent,
        steps=tuple(item[0] for item in states),
        selected_iteration=selected_step.iteration,
        stop_reason=stop_reason,
        duration_s=time.perf_counter() - started,
        initial_module_error_rate=_module_error_for_canvas(
            reference_image, blueprint, crop_padding_px=resolved_crop_padding_px
        ),
        final_module_error_rate=selected_step.actual_module_error_rate,
    )
