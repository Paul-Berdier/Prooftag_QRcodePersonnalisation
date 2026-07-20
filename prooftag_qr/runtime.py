from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


def runtime_info(settings: Settings | None = None) -> dict:
    packages = {}
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    result = {"packages": packages, "cuda_available": False}
    if settings is not None:
        result["generation_config"] = {
            "base_model_id": settings.base_model_id,
            "controlnet_model_id": settings.controlnet_model_id,
            "controlnet_pipeline_mode": settings.controlnet_pipeline_mode,
            "validation_min_pass_rate": settings.validation_min_pass_rate,
            "max_attempts": settings.max_attempts,
            "regenerate_before_global_repair": settings.regenerate_before_global_repair,
            "latent_refinement_enabled": settings.latent_refinement_enabled,
            "latent_refinement_iterations": settings.latent_refinement_iterations,
            "latent_refinement_learning_rate": settings.latent_refinement_learning_rate,
            "latent_refinement_qr_weight": settings.latent_refinement_qr_weight,
            "latent_refinement_preservation_weight": (
                settings.latent_refinement_preservation_weight
            ),
            "latent_refinement_functional_weight": (
                settings.latent_refinement_functional_weight
            ),
            "latent_refinement_target_module_error_rate": (
                settings.latent_refinement_target_module_error_rate
            ),
            "latent_refinement_max_latent_delta": (
                settings.latent_refinement_max_latent_delta
            ),
            "latent_refinement_max_mean_absolute_change": (
                settings.latent_refinement_max_mean_absolute_change
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
