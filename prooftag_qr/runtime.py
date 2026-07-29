from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


def runtime_info(settings: Settings | None = None) -> dict:
    packages = {}
    for package in (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "lpips",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    result = {"packages": packages, "cuda_available": False}
    if settings is not None:
        result["generation_config"] = {
            "base_model_id": settings.base_model_id,
            "controlnet_model_id": settings.controlnet_model_id,
            "controlnet_model_subfolder": settings.controlnet_model_subfolder,
            "controlnet_conditioning_profile": settings.controlnet_conditioning_profile,
            "controlnet_pipeline_mode": settings.controlnet_pipeline_mode,
            "validation_min_pass_rate": settings.validation_min_pass_rate,
            "max_attempts": settings.max_attempts,
            "regenerate_before_global_repair": settings.regenerate_before_global_repair,
            "guided_rediffusion_enabled": settings.guided_rediffusion_enabled,
            "guided_rediffusion_steps": settings.guided_rediffusion_steps,
            "guided_rediffusion_strength": settings.guided_rediffusion_strength,
            "guided_rediffusion_controlnet_scale": (settings.guided_rediffusion_controlnet_scale),
            "guided_rediffusion_guide_center_scale": (
                settings.guided_rediffusion_guide_center_scale
            ),
            "guided_rediffusion_guide_confidence_margin": (
                settings.guided_rediffusion_guide_confidence_margin
            ),
            "guided_rediffusion_mask_dilation_px": (settings.guided_rediffusion_mask_dilation_px),
            "guided_rediffusion_mask_feather_px": (settings.guided_rediffusion_mask_feather_px),
            "guided_rediffusion_max_mean_absolute_change": (
                settings.guided_rediffusion_max_mean_absolute_change
            ),
            "guided_rediffusion_min_relative_module_improvement": (
                settings.guided_rediffusion_min_relative_module_improvement
            ),
            "guided_rediffusion_seed_offset": settings.guided_rediffusion_seed_offset,
            "srpg_enabled": settings.srpg_enabled,
            "srpg_steps": settings.srpg_steps,
            "srpg_strength": settings.srpg_strength,
            "srpg_controlnet_scale": settings.srpg_controlnet_scale,
            "srpg_qr_weight": settings.srpg_qr_weight,
            "srpg_perceptual_weight": settings.srpg_perceptual_weight,
            "srpg_functional_weight": settings.srpg_functional_weight,
            "srpg_center_fraction": settings.srpg_center_fraction,
            "srpg_dark_threshold": settings.srpg_dark_threshold,
            "srpg_light_threshold": settings.srpg_light_threshold,
            "srpg_robust_blur_weight": settings.srpg_robust_blur_weight,
            "srpg_robust_blur_kernel": settings.srpg_robust_blur_kernel,
            "srpg_robust_downscale_weight": settings.srpg_robust_downscale_weight,
            "srpg_robust_downscale_factor": settings.srpg_robust_downscale_factor,
            "srpg_robust_brightness_weight": settings.srpg_robust_brightness_weight,
            "srpg_robust_brightness_low": settings.srpg_robust_brightness_low,
            "srpg_robust_brightness_high": settings.srpg_robust_brightness_high,
            "srpg_robust_contrast_weight": settings.srpg_robust_contrast_weight,
            "srpg_robust_contrast_factor": settings.srpg_robust_contrast_factor,
            "srpg_target_module_error_rate": settings.srpg_target_module_error_rate,
            "srpg_max_noise_delta_rms": settings.srpg_max_noise_delta_rms,
            "srpg_eta": settings.srpg_eta,
            "srpg_max_mean_absolute_change": settings.srpg_max_mean_absolute_change,
            "srpg_min_relative_module_improvement": (settings.srpg_min_relative_module_improvement),
            "srpg_seed_offset": settings.srpg_seed_offset,
            "srpg_save_step_previews": settings.srpg_save_step_previews,
            "srpg_preview_interval": settings.srpg_preview_interval,
            "srpg_quiet_zone_mode": settings.srpg_quiet_zone_mode,
            "srpg_quiet_zone_minimum_luminance": (
                settings.srpg_quiet_zone_minimum_luminance
            ),
            "srpg_functional_pattern_tone_factor": (
                settings.srpg_functional_pattern_tone_factor
            ),
            "srmpgd_enabled": settings.srmpgd_enabled,
            "srmpgd_max_iterations": settings.srmpgd_max_iterations,
            "srmpgd_step_size": settings.srmpgd_step_size,
            "srmpgd_lpips_weight": settings.srmpgd_lpips_weight,
            "srmpgd_lpips_net": settings.srmpgd_lpips_net,
            "srmpgd_crop_padding_px": settings.srmpgd_crop_padding_px,
            "srmpgd_dark_threshold": settings.srmpgd_dark_threshold,
            "srmpgd_light_threshold": settings.srmpgd_light_threshold,
            "srmpgd_center_fraction": settings.srmpgd_center_fraction,
            "srmpgd_max_initial_module_error_rate": (
                settings.srmpgd_max_initial_module_error_rate
            ),
            "latent_refinement_enabled": settings.latent_refinement_enabled,
            "latent_refinement_iterations": settings.latent_refinement_iterations,
            "latent_refinement_learning_rate": settings.latent_refinement_learning_rate,
            "latent_refinement_qr_weight": settings.latent_refinement_qr_weight,
            "latent_refinement_preservation_weight": (
                settings.latent_refinement_preservation_weight
            ),
            "latent_refinement_functional_weight": (settings.latent_refinement_functional_weight),
            "latent_refinement_target_module_error_rate": (
                settings.latent_refinement_target_module_error_rate
            ),
            "latent_refinement_max_latent_delta": (settings.latent_refinement_max_latent_delta),
            "latent_refinement_max_mean_absolute_change": (
                settings.latent_refinement_max_mean_absolute_change
            ),
            "latent_refinement_min_relative_module_improvement": (
                settings.latent_refinement_min_relative_module_improvement
            ),
        }
    try:
        import torch
    except ImportError:
        return result

    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_runtime"] = torch.version.cuda
    if result["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device": torch.cuda.get_device_name(0),
                "device_memory_bytes": properties.total_memory,
            }
        )
    return result
