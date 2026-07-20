from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

from PIL import Image

from . import metrics
from .config import Settings
from .guidance import LatentRefinementConfig, refine_candidate_latent
from .qr import QRBlueprint, repair_qr_modules
from .schemas import GenerationRequest

logger = logging.getLogger(__name__)

GLOBAL_REPAIR_VARIANTS = frozenset(
    {
        "centers_45",
        "centers_60",
        "centers_72",
        "centers_85",
        "tonal_90",
        "tonal_95",
        "centers_90",
        "centers_95",
    }
)


class GenerationBackend(ABC):
    @abstractmethod
    def generate(
        self, request: GenerationRequest, blueprint: QRBlueprint, seed: int
    ) -> Image.Image:
        raise NotImplementedError

    def variants(
        self, candidate: Image.Image, blueprint: QRBlueprint
    ) -> Iterable[tuple[str, Image.Image]]:
        yield "raw", candidate


class QRBackend(GenerationBackend):
    def generate(
        self, request: GenerationRequest, blueprint: QRBlueprint, seed: int
    ) -> Image.Image:
        return blueprint.image.copy()


class ControlNetBackend(GenerationBackend):
    """Lazy SD 1.5 ControlNet baseline; FreeQR-style guidance will replace this baseline."""

    _load_lock = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            started = time.perf_counter()
            try:
                try:
                    import torch
                    from diffusers import (
                        ControlNetModel,
                        DDIMScheduler,
                        DPMSolverMultistepScheduler,
                        StableDiffusionControlNetImg2ImgPipeline,
                        StableDiffusionControlNetPipeline,
                    )
                except ImportError as exc:
                    raise RuntimeError("Install the 'gpu' dependencies to use ControlNet") from exc
                if not torch.cuda.is_available():
                    raise RuntimeError("ControlNet backend requires an available CUDA GPU")
                dtype = torch.float16
                controlnet = ControlNetModel.from_pretrained(
                    self.settings.controlnet_model_id,
                    torch_dtype=dtype,
                    cache_dir=self.settings.model_cache_dir,
                )
                pipeline_class = (
                    StableDiffusionControlNetImg2ImgPipeline
                    if self.settings.controlnet_pipeline_mode == "img2img"
                    else StableDiffusionControlNetPipeline
                )
                pipe = pipeline_class.from_pretrained(
                    self.settings.base_model_id,
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    cache_dir=self.settings.model_cache_dir,
                    safety_checker=None,
                )
                scheduler_class = (
                    DDIMScheduler
                    if self.settings.controlnet_pipeline_mode == "img2img"
                    else DPMSolverMultistepScheduler
                )
                pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                pipe.set_progress_bar_config(disable=True)
                pipe.to(self.settings.device)
                self._pipeline = pipe
            except Exception:
                duration = time.perf_counter() - started
                metrics.MODEL_LOADS.labels("error").inc()
                metrics.MODEL_LOAD_DURATION.labels("error").observe(duration)
                logger.exception(
                    "controlnet_model_load_failed",
                    extra={"backend": "controlnet", "duration_ms": round(duration * 1000, 2)},
                )
                raise
            duration = time.perf_counter() - started
            metrics.MODEL_LOADS.labels("success").inc()
            metrics.MODEL_LOAD_DURATION.labels("success").observe(duration)
            metrics.MODEL_LOADED.set(1)
            logger.info(
                "controlnet_model_loaded",
                extra={"backend": "controlnet", "duration_ms": round(duration * 1000, 2)},
            )
        return self._pipeline

    def generate(
        self, request: GenerationRequest, blueprint: QRBlueprint, seed: int
    ) -> Image.Image:
        import torch

        pipe = self._load()
        generator = torch.Generator(device=self.settings.device).manual_seed(seed)
        arguments = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or None,
            "width": 512,
            "height": 512,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "controlnet_conditioning_scale": request.controlnet_scale,
            "generator": generator,
        }
        if self.settings.controlnet_pipeline_mode == "img2img":
            arguments.update(
                {
                    "image": blueprint.image,
                    "control_image": blueprint.image,
                    "strength": request.strength,
                }
            )
        else:
            arguments["image"] = blueprint.image
        result = pipe(
            **arguments,
        )
        return result.images[0].convert("RGB")

    def variants(
        self, candidate: Image.Image, blueprint: QRBlueprint
    ) -> Iterable[tuple[str, Image.Image]]:
        yield "raw", candidate
        if self.settings.latent_refinement_enabled:
            started = time.perf_counter()
            try:
                result = refine_candidate_latent(
                    self._load(),
                    candidate,
                    blueprint,
                    LatentRefinementConfig(
                        iterations=self.settings.latent_refinement_iterations,
                        learning_rate=self.settings.latent_refinement_learning_rate,
                        qr_weight=self.settings.latent_refinement_qr_weight,
                        preservation_weight=(
                            self.settings.latent_refinement_preservation_weight
                        ),
                        functional_weight=self.settings.latent_refinement_functional_weight,
                        target_module_error_rate=(
                            self.settings.latent_refinement_target_module_error_rate
                        ),
                        max_latent_delta=(
                            self.settings.latent_refinement_max_latent_delta
                        ),
                        max_mean_absolute_change=(
                            self.settings.latent_refinement_max_mean_absolute_change
                        ),
                    ),
                )
                duration = time.perf_counter() - started
                outcome = (
                    "converged"
                    if result.converged
                    else (
                        "improved"
                        if result.accepted
                        else (
                            "rejected_preservation"
                            if result.improved
                            else "no_improvement"
                        )
                    )
                )
                metrics.LATENT_REFINEMENTS.labels(outcome).inc()
                metrics.LATENT_REFINEMENT_DURATION.observe(duration)
                metrics.LATENT_REFINEMENT_ITERATIONS.observe(result.iterations)
                metrics.LATENT_REFINEMENT_MODULE_ERROR_RATE.labels("before").set(
                    result.initial_module_error_rate
                )
                metrics.LATENT_REFINEMENT_MODULE_ERROR_RATE.labels("after").set(
                    result.final_module_error_rate
                )
                metrics.LATENT_REFINEMENT_MODULE_ERROR_RATE.labels("best_observed").set(
                    result.best_observed_module_error_rate
                )
                metrics.LATENT_REFINEMENT_LOSS.labels("srl").set(result.final_srl)
                metrics.LATENT_REFINEMENT_LOSS.labels("preservation").set(
                    result.final_preservation_loss
                )
                metrics.LATENT_REFINEMENT_LOSS.labels("mean_absolute_change").set(
                    result.final_mean_absolute_change
                )
                metrics.LATENT_REFINEMENT_LOSS.labels(
                    "best_observed_mean_absolute_change"
                ).set(result.best_observed_mean_absolute_change)
                logger.info(
                    "latent_refinement_completed",
                    extra={
                        "backend": "controlnet",
                        "status": outcome,
                        "duration_ms": round(duration * 1000, 2),
                        "iterations": result.iterations,
                        "initial_module_error_rate": result.initial_module_error_rate,
                        "final_module_error_rate": result.final_module_error_rate,
                        "srl": result.final_srl,
                        "preservation_loss": result.final_preservation_loss,
                        "mean_absolute_change": result.final_mean_absolute_change,
                        "best_observed_module_error_rate": (
                            result.best_observed_module_error_rate
                        ),
                        "best_observed_mean_absolute_change": (
                            result.best_observed_mean_absolute_change
                        ),
                        "improved": result.improved,
                        "accepted": result.accepted,
                        "converged": result.converged,
                        "rejection_reason": result.rejection_reason,
                    },
                )
                if result.accepted:
                    yield "latent_srl", result.image
            except Exception:
                duration = time.perf_counter() - started
                metrics.LATENT_REFINEMENTS.labels("error").inc()
                metrics.LATENT_REFINEMENT_DURATION.observe(duration)
                logger.exception(
                    "latent_refinement_failed",
                    extra={
                        "backend": "controlnet",
                        "status": "error",
                        "duration_ms": round(duration * 1000, 2),
                    },
                )
        repair_profiles = (
            # Rounded profiles trade the visible QR grid for small, blended superellipses.
            ("rounded_16", 0.95, True, True, 16.0, 0.05, 0.18, True, True),
            ("rounded_32", 0.95, True, True, 32.0, 0.00, 0.35, True, True),
            ("rounded_48", 0.95, True, True, 48.0, 0.05, 0.25, True, True),
            # Perceptual profiles retain chroma/texture and feather center-patch edges.
            # Binary profiles remain as a safe fallback when a softer result does not scan.
            ("perceptual_16", 0.85, True, True, 16.0, 0.20, 0.10, True, False),
            ("perceptual_16_strong", 0.90, True, True, 16.0, 0.05, 0.10, True, False),
            ("perceptual_32", 0.85, True, True, 32.0, 0.15, 0.10, True, False),
            ("perceptual_32_strong", 0.85, True, True, 32.0, 0.05, 0.10, True, False),
            ("perceptual_32_wide", 0.90, True, True, 32.0, 0.10, 0.10, True, False),
            ("perceptual_48", 0.90, True, True, 48.0, 0.05, 0.10, True, False),
            ("perceptual_64", 0.90, True, True, 64.0, 0.05, 0.10, True, False),
            ("functional", 0.0, False, False, 0.0, 0.25, 0.0, False, False),
            ("incorrect_55", 0.55, True, False, 0.0, 0.25, 0.0, False, False),
            ("incorrect_72", 0.72, True, False, 0.0, 0.25, 0.0, False, False),
            ("incorrect_80", 0.80, True, False, 0.0, 0.25, 0.0, False, False),
            ("incorrect_85", 0.85, True, False, 0.0, 0.25, 0.0, False, False),
            ("uncertain_16", 0.85, True, False, 16.0, 0.25, 0.0, False, False),
            ("uncertain_32", 0.85, True, False, 32.0, 0.25, 0.0, False, False),
            ("uncertain_48", 0.85, True, False, 48.0, 0.25, 0.0, False, False),
            ("uncertain_64", 0.85, True, False, 64.0, 0.25, 0.0, False, False),
            ("centers_45", 0.45, False, False, 0.0, 0.25, 0.0, False, False),
            ("centers_60", 0.60, False, False, 0.0, 0.25, 0.0, False, False),
            ("centers_72", 0.72, False, False, 0.0, 0.25, 0.0, False, False),
            ("centers_85", 0.85, False, False, 0.0, 0.25, 0.0, False, False),
            ("tonal_90", 0.90, False, True, 0.0, 0.25, 0.0, False, False),
            ("tonal_95", 0.95, False, True, 0.0, 0.25, 0.0, False, False),
            ("centers_90", 0.90, False, False, 0.0, 0.25, 0.0, False, False),
            ("centers_95", 0.95, False, False, 0.0, 0.25, 0.0, False, False),
        )
        for (
            name,
            center_scale,
            incorrect_only,
            preserve_tone,
            confidence_margin,
            tone_factor,
            edge_feather,
            preserve_functional_tone,
            rounded_edges,
        ) in repair_profiles:
            yield (
                name,
                repair_qr_modules(
                    candidate,
                    blueprint,
                    center_scale=center_scale,
                    incorrect_only=incorrect_only,
                    preserve_tone=preserve_tone,
                    confidence_margin=confidence_margin,
                    tone_factor=tone_factor,
                    edge_feather=edge_feather,
                    preserve_functional_tone=preserve_functional_tone,
                    rounded_edges=rounded_edges,
                ),
            )


def build_backends(settings: Settings) -> dict[str, GenerationBackend]:
    return {"qr": QRBackend(), "controlnet": ControlNetBackend(settings)}
