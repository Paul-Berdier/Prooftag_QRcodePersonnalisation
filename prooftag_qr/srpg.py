from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .guidance import prepare_torch_layout, scanning_robust_loss
from .qr import QRBlueprint, module_error_rate
from .quality import image_change_metrics


@dataclass(frozen=True, slots=True)
class SRPGConfig:
    steps: int = 40
    strength: float = 1.0
    controlnet_scale: float = 1.35
    qr_weight: float = 500.0
    perceptual_weight: float = 3.0
    functional_weight: float = 4.0
    center_fraction: float = 1 / 3
    dark_threshold: float = 0.5
    light_threshold: float = 0.5
    robust_blur_weight: float = 0.0
    robust_blur_kernel: int = 3
    robust_downscale_weight: float = 0.0
    robust_downscale_factor: float = 0.75
    robust_brightness_weight: float = 0.0
    robust_brightness_low: float = 0.75
    robust_brightness_high: float = 1.25
    robust_contrast_weight: float = 0.0
    robust_contrast_factor: float = 0.70
    target_module_error_rate: float = 0.0
    max_noise_delta_rms: float = 2.0
    eta: float = 0.0
    max_mean_absolute_change: float = 0.20
    min_relative_module_improvement: float = 0.10
    save_step_previews: bool = False
    preview_interval: int = 5


@dataclass(frozen=True, slots=True)
class SRPGStep:
    index: int
    timestep: int
    module_error_rate: float
    scanning_robust_loss: float
    perceptual_loss: float
    gradient_rms: float
    noise_delta_rms: float
    gradient_clipped: bool
    guidance_applied: bool


@dataclass(frozen=True, slots=True)
class SRPGPreview:
    index: int
    timestep: int
    predicted_clean_image: Image.Image
    active_module_map: Image.Image


@dataclass(frozen=True, slots=True)
class SRPGResult:
    image: Image.Image
    steps: tuple[SRPGStep, ...]
    previews: tuple[SRPGPreview, ...]
    initial_module_error_rate: float
    final_module_error_rate: float
    changed_pixel_ratio: float
    mean_absolute_change: float
    peak_gpu_memory_allocated_mib: float | None
    accepted: bool
    rejection_reason: str | None


def _predict_original_sample(
    scheduler: Any,
    model_output: Any,
    timestep: Any,
    sample: Any,
) -> Any:
    """Differentiable DDIM x0 prediction matching Diffusers 0.31."""
    alphas = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas[timestep]
    beta_prod_t = 1 - alpha_prod_t
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        predicted = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    elif prediction_type == "sample":
        predicted = model_output
    elif prediction_type == "v_prediction":
        predicted = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")
    if scheduler.config.thresholding:
        predicted = scheduler._threshold_sample(predicted)
    elif scheduler.config.clip_sample:
        predicted = predicted.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )
    return predicted


def _load_lpips(pipeline: Any, device: Any) -> Any:
    cached = getattr(pipeline, "_prooftag_lpips", None)
    if cached is not None:
        return cached
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Install the pinned 'lpips' GPU dependency for SRPG") from exc
    model = lpips.LPIPS(net="alex", verbose=False)
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    pipeline._prooftag_lpips = model
    return model


def _validate_config(config: SRPGConfig) -> None:
    if config.steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0 < config.strength <= 1:
        raise ValueError("strength must be between 0 (exclusive) and 1")
    if config.controlnet_scale <= 0:
        raise ValueError("controlnet_scale must be positive")
    if config.qr_weight <= 0 or config.perceptual_weight < 0:
        raise ValueError("invalid SRPG loss weights")
    if config.functional_weight < 1:
        raise ValueError("functional_weight must be at least 1")
    if not 0 < config.center_fraction <= 1:
        raise ValueError("center_fraction must be between 0 (exclusive) and 1")
    if not 0 < config.dark_threshold <= config.light_threshold < 1:
        raise ValueError("SRPG thresholds must satisfy 0 < dark <= light < 1")
    if (
        min(
            config.robust_blur_weight,
            config.robust_downscale_weight,
            config.robust_brightness_weight,
            config.robust_contrast_weight,
        )
        < 0
    ):
        raise ValueError("SRPG robustness weights cannot be negative")
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
    if not 0 <= config.target_module_error_rate <= 1:
        raise ValueError("target_module_error_rate must be between 0 and 1")
    if config.max_noise_delta_rms <= 0:
        raise ValueError("max_noise_delta_rms must be positive")
    if not 0 <= config.eta <= 1:
        raise ValueError("eta must be between 0 and 1")
    if not 0 < config.max_mean_absolute_change <= 1:
        raise ValueError("max_mean_absolute_change must be between 0 (exclusive) and 1")
    if not 0 <= config.min_relative_module_improvement <= 1:
        raise ValueError("min_relative_module_improvement must be between 0 and 1")
    if config.preview_interval < 1:
        raise ValueError("preview_interval must be at least 1")


def run_srpg_controlnet_img2img(
    pipeline: Any,
    candidate: Image.Image,
    blueprint: QRBlueprint,
    *,
    prompt: str,
    negative_prompt: str | None,
    guidance_scale: float,
    generator: Any,
    config: SRPGConfig,
    control_image: Image.Image | None = None,
    preview_callback: Callable[[SRPGPreview, SRPGStep], None] | None = None,
) -> SRPGResult:
    """Run Stage-2 ControlNet with SRPG inside every DDIM denoising step.

    This follows DiffQRCoder equations 7-11: predict x0, decode it, differentiate
    SRL + LPIPS with respect to z_t, add the conditional gradient to epsilon, then
    perform the DDIM step. Batch size one and one ControlNet are intentional E005
    constraints so the behavior remains auditable on a 20 GiB GPU.
    """
    import torch

    _validate_config(config)
    if pipeline.scheduler.__class__.__name__ != "DDIMScheduler":
        raise RuntimeError("SRPG requires DDIMScheduler")
    device = pipeline._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0
    components = (pipeline.vae, pipeline.unet, pipeline.controlnet, pipeline.text_encoder)
    for component in components:
        if component is not None:
            component.requires_grad_(False)
            component.eval()
    for component in (pipeline.unet, pipeline.controlnet):
        enable_checkpointing = getattr(component, "enable_gradient_checkpointing", None)
        if enable_checkpointing is not None:
            enable_checkpointing()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        device,
        1,
        do_classifier_free_guidance,
        negative_prompt,
    )
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    dtype = prompt_embeds.dtype

    source_image = pipeline.image_processor.preprocess(
        candidate.convert("RGB"), height=blueprint.image.height, width=blueprint.image.width
    ).to(device=device, dtype=torch.float32)
    reference_lpips = source_image.to(device=device, dtype=torch.float32)
    prepared_control_image = pipeline.prepare_control_image(
        image=control_image if control_image is not None else blueprint.image,
        width=blueprint.image.width,
        height=blueprint.image.height,
        batch_size=1,
        num_images_per_prompt=1,
        device=device,
        dtype=pipeline.controlnet.dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
        guess_mode=False,
    )
    pipeline.scheduler.set_timesteps(config.steps, device=device)
    timesteps, _ = pipeline.get_timesteps(config.steps, config.strength, device)
    latent_timestep = timesteps[:1]
    latents = pipeline.prepare_latents(
        source_image,
        latent_timestep,
        1,
        1,
        dtype,
        device,
        generator,
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, config.eta)
    layout = prepare_torch_layout(
        blueprint,
        blueprint.image.height,
        blueprint.image.width,
        device=device,
        dtype=dtype,
        center_fraction=config.center_fraction,
    )
    perceptual_model = _load_lpips(pipeline, device)
    scaling_factor = pipeline.vae.config.scaling_factor
    traces: list[SRPGStep] = []
    previews: list[SRPGPreview] = []
    preview_indices = (
        set(range(0, len(timesteps), config.preview_interval)) | {len(timesteps) - 1}
        if config.save_step_previews
        else set()
    )

    with torch.enable_grad():
        for index, timestep in enumerate(timesteps):
            latents = latents.detach().requires_grad_(True)
            latent_model_input = (
                torch.cat([latents, latents]) if do_classifier_free_guidance else latents
            )
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
            down_samples, mid_sample = pipeline.controlnet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=prepared_control_image,
                conditioning_scale=config.controlnet_scale,
                guess_mode=False,
                return_dict=False,
            )
            noise_prediction = pipeline.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_unconditional, noise_text = noise_prediction.chunk(2)
                noise_prediction = noise_unconditional + guidance_scale * (
                    noise_text - noise_unconditional
                )

            predicted_clean_latent = _predict_original_sample(
                pipeline.scheduler,
                noise_prediction,
                timestep,
                latents,
            )
            decoded = pipeline.vae.decode(
                predicted_clean_latent / scaling_factor, return_dict=False
            )[0]
            decoded_unit = (decoded / 2 + 0.5).clamp(0, 1)
            scanning_loss, diagnostics = scanning_robust_loss(
                decoded_unit,
                blueprint,
                functional_weight=config.functional_weight,
                center_fraction=config.center_fraction,
                dark_threshold=config.dark_threshold,
                light_threshold=config.light_threshold,
                layout=layout,
            )
            robust_weight = 1.0
            if config.robust_blur_weight:
                import torch.nn.functional as functional

                blurred = functional.avg_pool2d(
                    decoded_unit,
                    kernel_size=config.robust_blur_kernel,
                    stride=1,
                    padding=config.robust_blur_kernel // 2,
                )
                loss, _ = scanning_robust_loss(
                    blurred,
                    blueprint,
                    functional_weight=config.functional_weight,
                    dark_threshold=config.dark_threshold,
                    light_threshold=config.light_threshold,
                    layout=layout,
                )
                scanning_loss = scanning_loss + config.robust_blur_weight * loss
                robust_weight += config.robust_blur_weight
            if config.robust_downscale_weight:
                import torch.nn.functional as functional

                reduced = functional.interpolate(
                    decoded_unit,
                    scale_factor=config.robust_downscale_factor,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                restored = functional.interpolate(
                    reduced,
                    size=decoded_unit.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss, _ = scanning_robust_loss(
                    restored,
                    blueprint,
                    functional_weight=config.functional_weight,
                    dark_threshold=config.dark_threshold,
                    light_threshold=config.light_threshold,
                    layout=layout,
                )
                scanning_loss = scanning_loss + config.robust_downscale_weight * loss
                robust_weight += config.robust_downscale_weight
            if config.robust_brightness_weight:
                brightness_loss = decoded_unit.new_tensor(0.0)
                for factor in (
                    config.robust_brightness_low,
                    config.robust_brightness_high,
                ):
                    transformed = (decoded_unit * factor).clamp(0, 1)
                    loss, _ = scanning_robust_loss(
                        transformed,
                        blueprint,
                        functional_weight=config.functional_weight,
                        dark_threshold=config.dark_threshold,
                        light_threshold=config.light_threshold,
                        layout=layout,
                    )
                    brightness_loss = brightness_loss + loss / 2
                scanning_loss = scanning_loss + config.robust_brightness_weight * brightness_loss
                robust_weight += config.robust_brightness_weight
            if config.robust_contrast_weight:
                contrasted = (
                    (decoded_unit - 0.5) * config.robust_contrast_factor + 0.5
                ).clamp(0, 1)
                loss, _ = scanning_robust_loss(
                    contrasted,
                    blueprint,
                    functional_weight=config.functional_weight,
                    dark_threshold=config.dark_threshold,
                    light_threshold=config.light_threshold,
                    layout=layout,
                )
                scanning_loss = scanning_loss + config.robust_contrast_weight * loss
                robust_weight += config.robust_contrast_weight
            scanning_loss = scanning_loss / robust_weight
            perceptual_loss = perceptual_model(decoded.float(), reference_lpips).mean()
            module_error = float(diagnostics["module_error_rate"].detach().float().item())
            preview = None
            if index in preview_indices:
                preview_image = pipeline.image_processor.postprocess(
                    decoded.detach(),
                    output_type="pil",
                    do_denormalize=[True],
                )[0].convert("RGB")
                active_modules = (
                    diagnostics["active_mask"][0]
                    .detach()
                    .reshape(blueprint.matrix.shape)
                    .to(torch.uint8)
                    .mul(255)
                    .cpu()
                    .numpy()
                )
                active_map = Image.fromarray(active_modules).resize(
                    blueprint.image.size,
                    Image.Resampling.NEAREST,
                )
                preview = SRPGPreview(
                    index=index,
                    timestep=int(timestep.item()),
                    predicted_clean_image=preview_image,
                    active_module_map=active_map,
                )
                previews.append(preview)
            guidance_applied = module_error > config.target_module_error_rate
            gradient_rms = 0.0
            noise_delta_rms = 0.0
            gradient_clipped = False
            guided_noise_prediction = noise_prediction
            if guidance_applied:
                objective = (
                    config.qr_weight * scanning_loss + config.perceptual_weight * perceptual_loss
                )
                gradient = torch.autograd.grad(objective, latents, only_inputs=True)[0]
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"non-finite SRPG gradient at step {index}")
                gradient_rms_tensor = gradient.float().square().mean().sqrt()
                gradient_rms = float(gradient_rms_tensor.item())
                alphas = pipeline.scheduler.alphas_cumprod.to(
                    device=latents.device, dtype=latents.dtype
                )
                noise_delta = (1 - alphas[timestep]).sqrt() * gradient
                noise_delta_rms_tensor = noise_delta.float().square().mean().sqrt()
                if noise_delta_rms_tensor > config.max_noise_delta_rms:
                    noise_delta = noise_delta * (
                        config.max_noise_delta_rms / noise_delta_rms_tensor.clamp_min(1e-8)
                    )
                    gradient_clipped = True
                    noise_delta_rms_tensor = noise_delta.float().square().mean().sqrt()
                noise_delta_rms = float(noise_delta_rms_tensor.item())
                guided_noise_prediction = noise_prediction + noise_delta.to(noise_prediction.dtype)

            step = SRPGStep(
                index=index,
                timestep=int(timestep.item()),
                module_error_rate=module_error,
                scanning_robust_loss=float(scanning_loss.detach().float().item()),
                perceptual_loss=float(perceptual_loss.detach().float().item()),
                gradient_rms=gradient_rms,
                noise_delta_rms=noise_delta_rms,
                gradient_clipped=gradient_clipped,
                guidance_applied=guidance_applied,
            )
            traces.append(step)
            if preview is not None and preview_callback is not None:
                preview_callback(preview, step)
            latents = pipeline.scheduler.step(
                guided_noise_prediction.detach(),
                timestep,
                latents.detach(),
                **extra_step_kwargs,
                return_dict=False,
            )[0]

    with torch.no_grad():
        decoded_final = pipeline.vae.decode(latents / scaling_factor, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(
            decoded_final, output_type="pil", do_denormalize=[True]
        )[0].convert("RGB")

    initial_error = module_error_rate(candidate, blueprint)
    final_error = module_error_rate(image, blueprint)
    change = image_change_metrics(image, candidate)
    peak_gpu_memory_allocated_mib = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    qr_improvement_ok = final_error < initial_error * (1.0 - config.min_relative_module_improvement)
    preservation_ok = change["mean_absolute_change"] <= config.max_mean_absolute_change
    rejection_reason = None
    if not qr_improvement_ok:
        rejection_reason = "actual_module_error_not_improved"
    elif not preservation_ok:
        rejection_reason = "mean_absolute_change_limit"
    return SRPGResult(
        image=image,
        steps=tuple(traces),
        previews=tuple(previews),
        initial_module_error_rate=initial_error,
        final_module_error_rate=final_error,
        changed_pixel_ratio=change["changed_pixel_ratio"],
        mean_absolute_change=change["mean_absolute_change"],
        peak_gpu_memory_allocated_mib=peak_gpu_memory_allocated_mib,
        accepted=qr_improvement_ok and preservation_ok,
        rejection_reason=rejection_reason,
    )
