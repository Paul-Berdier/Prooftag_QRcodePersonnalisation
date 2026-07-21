from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil

from PIL import Image

from . import metrics
from .config import Settings
from .guidance import LatentRefinementConfig, refine_candidate_latent
from .qr import QRBlueprint, module_error_rate, repair_qr_modules
from .quality import composite_guided_regions, image_change_metrics
from .schemas import GenerationRequest
from .srpg import SRPGConfig, run_srpg_controlnet_img2img

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


@dataclass(frozen=True, slots=True)
class GuidedRediffusionResult:
    image: Image.Image
    unprojected_image: Image.Image
    control_image: Image.Image
    mask_image: Image.Image
    initial_module_error_rate: float
    control_module_error_rate: float
    final_module_error_rate: float
    mask_coverage: float
    changed_pixel_ratio: float
    mean_absolute_change: float
    unprojected_changed_pixel_ratio: float
    unprojected_mean_absolute_change: float
    accepted: bool
    rejection_reason: str | None


class GenerationBackend(ABC):
    @abstractmethod
    def generate(
        self, request: GenerationRequest, blueprint: QRBlueprint, seed: int
    ) -> Image.Image:
        raise NotImplementedError

    def variants(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        *,
        request: GenerationRequest | None = None,
        seed: int | None = None,
        run_id: str | None = None,
        attempt: int | None = None,
    ) -> Iterable[tuple[str, Image.Image]]:
        yield "raw", candidate

    def debug_artifacts(self) -> dict[str, Image.Image]:
        return {}


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
        self._debug_artifacts: dict[str, Image.Image] = {}

    def debug_artifacts(self) -> dict[str, Image.Image]:
        return self._debug_artifacts.copy()

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

    def _guided_rediffuse(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        request: GenerationRequest,
        seed: int,
    ) -> GuidedRediffusionResult:
        """Run the DiffQRCoder-inspired second img2img diffusion stage.

        A sparse QR-aware control image marks incorrect or uncertain modules. The artistic
        candidate remains the img2img source, so ControlNet sees the technical constraint while
        the diffusion source retains the composition and style of stage 1.
        """
        import torch

        if self.settings.controlnet_pipeline_mode != "img2img":
            raise RuntimeError("Guided rediffusion requires the img2img ControlNet pipeline")
        control_image = repair_qr_modules(
            candidate,
            blueprint,
            center_scale=self.settings.guided_rediffusion_guide_center_scale,
            incorrect_only=True,
            preserve_tone=True,
            confidence_margin=self.settings.guided_rediffusion_guide_confidence_margin,
            tone_factor=0.08,
            edge_feather=0.25,
            preserve_functional_tone=False,
            rounded_edges=True,
        )
        rediffusion_seed = (seed + self.settings.guided_rediffusion_seed_offset) % (2**32)
        scheduler_steps = ceil(
            self.settings.guided_rediffusion_steps / self.settings.guided_rediffusion_strength
        )
        generator = torch.Generator(device=self.settings.device).manual_seed(rediffusion_seed)
        result = self._load()(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            image=candidate,
            control_image=control_image,
            width=512,
            height=512,
            num_inference_steps=scheduler_steps,
            strength=self.settings.guided_rediffusion_strength,
            guidance_scale=request.guidance_scale,
            controlnet_conditioning_scale=(self.settings.guided_rediffusion_controlnet_scale),
            generator=generator,
        )
        unprojected_image = result.images[0].convert("RGB")
        image, mask_image = composite_guided_regions(
            candidate,
            unprojected_image,
            control_image,
            dilation_px=self.settings.guided_rediffusion_mask_dilation_px,
            feather_px=self.settings.guided_rediffusion_mask_feather_px,
        )
        mask_histogram = mask_image.histogram()
        mask_coverage = sum(level * count for level, count in enumerate(mask_histogram)) / (
            255 * mask_image.width * mask_image.height
        )
        change = image_change_metrics(image, candidate)
        unprojected_change = image_change_metrics(unprojected_image, candidate)
        initial_module_error = module_error_rate(candidate, blueprint)
        final_module_error = module_error_rate(image, blueprint)
        preservation_ok = (
            change["mean_absolute_change"]
            <= self.settings.guided_rediffusion_max_mean_absolute_change
        )
        qr_improvement_ok = final_module_error < initial_module_error * (
            1.0 - self.settings.guided_rediffusion_min_relative_module_improvement
        )
        rejection_reason = None
        if not qr_improvement_ok:
            rejection_reason = "actual_module_error_not_improved"
        elif not preservation_ok:
            rejection_reason = "mean_absolute_change_limit"
        return GuidedRediffusionResult(
            image=image,
            unprojected_image=unprojected_image,
            control_image=control_image,
            mask_image=mask_image,
            initial_module_error_rate=initial_module_error,
            control_module_error_rate=module_error_rate(control_image, blueprint),
            final_module_error_rate=final_module_error,
            mask_coverage=mask_coverage,
            changed_pixel_ratio=change["changed_pixel_ratio"],
            mean_absolute_change=change["mean_absolute_change"],
            unprojected_changed_pixel_ratio=unprojected_change["changed_pixel_ratio"],
            unprojected_mean_absolute_change=unprojected_change["mean_absolute_change"],
            accepted=preservation_ok and qr_improvement_ok,
            rejection_reason=rejection_reason,
        )

    def variants(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        *,
        request: GenerationRequest | None = None,
        seed: int | None = None,
        run_id: str | None = None,
        attempt: int | None = None,
    ) -> Iterable[tuple[str, Image.Image]]:
        self._debug_artifacts.clear()
        yield "raw", candidate
        enhanced_candidate = candidate
        enhanced_prefix = ""
        if self.settings.srpg_enabled:
            started = time.perf_counter()
            try:
                if request is None or seed is None:
                    raise ValueError("request and seed are required for SRPG")
                import torch

                srpg_seed = (seed + self.settings.srpg_seed_offset) % (2**32)
                generator = torch.Generator(device=self.settings.device).manual_seed(srpg_seed)
                srpg = run_srpg_controlnet_img2img(
                    self._load(),
                    candidate,
                    blueprint,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or None,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                    config=SRPGConfig(
                        steps=self.settings.srpg_steps,
                        strength=self.settings.srpg_strength,
                        controlnet_scale=self.settings.srpg_controlnet_scale,
                        qr_weight=self.settings.srpg_qr_weight,
                        perceptual_weight=self.settings.srpg_perceptual_weight,
                        functional_weight=self.settings.srpg_functional_weight,
                        dark_threshold=self.settings.srpg_dark_threshold,
                        light_threshold=self.settings.srpg_light_threshold,
                        target_module_error_rate=(self.settings.srpg_target_module_error_rate),
                        max_noise_delta_rms=self.settings.srpg_max_noise_delta_rms,
                        max_mean_absolute_change=(self.settings.srpg_max_mean_absolute_change),
                        min_relative_module_improvement=(
                            self.settings.srpg_min_relative_module_improvement
                        ),
                        save_step_previews=self.settings.srpg_save_step_previews,
                        preview_interval=self.settings.srpg_preview_interval,
                    ),
                )
                if srpg.previews:
                    self._debug_artifacts["srpg_control"] = blueprint.image.copy()
                    for preview in srpg.previews:
                        prefix = f"srpg_step_{preview.index:02d}"
                        self._debug_artifacts[f"{prefix}_x0"] = preview.predicted_clean_image
                        self._debug_artifacts[f"{prefix}_errors"] = preview.active_module_map
                duration = time.perf_counter() - started
                outcome = "accepted" if srpg.accepted else f"rejected_{srpg.rejection_reason}"
                metrics.SRPG_RUNS.labels(outcome).inc()
                metrics.SRPG_DURATION.observe(duration)
                metrics.DURATION.labels("controlnet", "srpg").observe(duration)
                metrics.SRPG_MODULE_ERROR_RATE.labels("before").set(srpg.initial_module_error_rate)
                metrics.SRPG_MODULE_ERROR_RATE.labels("after").set(srpg.final_module_error_rate)
                metrics.SRPG_IMAGE_CHANGE.labels("changed_pixel_ratio").set(
                    srpg.changed_pixel_ratio
                )
                metrics.SRPG_IMAGE_CHANGE.labels("mean_absolute_change").set(
                    srpg.mean_absolute_change
                )
                if srpg.peak_gpu_memory_allocated_mib is not None:
                    metrics.SRPG_PEAK_GPU_MEMORY_MIB.set(srpg.peak_gpu_memory_allocated_mib)
                for step in srpg.steps:
                    step_label = str(step.index)
                    metrics.SRPG_STEP_DIAGNOSTIC.labels(step_label, "module_error_rate").set(
                        step.module_error_rate
                    )
                    metrics.SRPG_STEP_DIAGNOSTIC.labels(step_label, "srl").set(
                        step.scanning_robust_loss
                    )
                    metrics.SRPG_STEP_DIAGNOSTIC.labels(step_label, "lpips").set(
                        step.perceptual_loss
                    )
                    metrics.SRPG_STEP_DIAGNOSTIC.labels(step_label, "gradient_rms").set(
                        step.gradient_rms
                    )
                    metrics.SRPG_STEP_DIAGNOSTIC.labels(step_label, "noise_delta_rms").set(
                        step.noise_delta_rms
                    )
                    if step.gradient_clipped:
                        metrics.SRPG_GRADIENT_CLIPS.inc()
                logger.info(
                    "srpg_completed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": outcome,
                        "attempt": attempt,
                        "seed": seed,
                        "srpg_seed": srpg_seed,
                        "duration_ms": round(duration * 1000, 2),
                        "steps": len(srpg.steps),
                        "initial_module_error_rate": srpg.initial_module_error_rate,
                        "final_module_error_rate": srpg.final_module_error_rate,
                        "changed_pixel_ratio": srpg.changed_pixel_ratio,
                        "mean_absolute_change": srpg.mean_absolute_change,
                        "peak_gpu_memory_allocated_mib": (srpg.peak_gpu_memory_allocated_mib),
                        "accepted": srpg.accepted,
                        "rejection_reason": srpg.rejection_reason,
                        "preview_steps": [item.index for item in srpg.previews],
                        "step_metrics": [
                            {
                                "index": item.index,
                                "timestep": item.timestep,
                                "module_error_rate": item.module_error_rate,
                                "srl": item.scanning_robust_loss,
                                "lpips": item.perceptual_loss,
                                "gradient_rms": item.gradient_rms,
                                "noise_delta_rms": item.noise_delta_rms,
                                "gradient_clipped": item.gradient_clipped,
                                "guidance_applied": item.guidance_applied,
                            }
                            for item in srpg.steps
                        ],
                    },
                )
                # The service must independently validate every SRPG output with the
                # complete decoder/degradation matrix.  The internal gate only decides
                # whether deterministic repairs may use it as their new visual base.
                yield "srpg", srpg.image
                if srpg.accepted:
                    enhanced_candidate = srpg.image
                    enhanced_prefix = "srpg_"
            except Exception:
                duration = time.perf_counter() - started
                metrics.SRPG_RUNS.labels("error").inc()
                metrics.SRPG_DURATION.observe(duration)
                metrics.DURATION.labels("controlnet", "srpg").observe(duration)
                logger.exception(
                    "srpg_failed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": "error",
                        "attempt": attempt,
                        "seed": seed,
                        "duration_ms": round(duration * 1000, 2),
                    },
                )
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.debug("srpg_cuda_cache_cleanup_failed", exc_info=True)
        if self.settings.guided_rediffusion_enabled:
            started = time.perf_counter()
            try:
                if request is None or seed is None:
                    raise ValueError("request and seed are required for guided rediffusion")
                guided = self._guided_rediffuse(candidate, blueprint, request, seed)
                self._debug_artifacts["guided_control"] = guided.control_image
                self._debug_artifacts["guided_mask"] = guided.mask_image
                self._debug_artifacts["guided_unprojected"] = guided.unprojected_image
                if not guided.accepted:
                    self._debug_artifacts["guided_projected"] = guided.image
                duration = time.perf_counter() - started
                outcome = (
                    "quality_pass" if guided.accepted else f"rejected_{guided.rejection_reason}"
                )
                metrics.GUIDED_REDIFFUSIONS.labels(outcome).inc()
                metrics.GUIDED_REDIFFUSION_DURATION.observe(duration)
                metrics.DURATION.labels("controlnet", "guided_rediffusion").observe(duration)
                metrics.GUIDED_REDIFFUSION_MODULE_ERROR_RATE.labels("before").set(
                    guided.initial_module_error_rate
                )
                metrics.GUIDED_REDIFFUSION_MODULE_ERROR_RATE.labels("control").set(
                    guided.control_module_error_rate
                )
                metrics.GUIDED_REDIFFUSION_MODULE_ERROR_RATE.labels("after").set(
                    guided.final_module_error_rate
                )
                metrics.GUIDED_REDIFFUSION_IMAGE_CHANGE.labels("changed_pixel_ratio").set(
                    guided.changed_pixel_ratio
                )
                metrics.GUIDED_REDIFFUSION_IMAGE_CHANGE.labels("mean_absolute_change").set(
                    guided.mean_absolute_change
                )
                metrics.GUIDED_REDIFFUSION_IMAGE_CHANGE.labels("mask_coverage").set(
                    guided.mask_coverage
                )
                metrics.GUIDED_REDIFFUSION_IMAGE_CHANGE.labels(
                    "unprojected_changed_pixel_ratio"
                ).set(guided.unprojected_changed_pixel_ratio)
                metrics.GUIDED_REDIFFUSION_IMAGE_CHANGE.labels(
                    "unprojected_mean_absolute_change"
                ).set(guided.unprojected_mean_absolute_change)
                logger.info(
                    "guided_rediffusion_completed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": outcome,
                        "attempt": attempt,
                        "seed": seed,
                        "duration_ms": round(duration * 1000, 2),
                        "steps": self.settings.guided_rediffusion_steps,
                        "scheduler_steps": ceil(
                            self.settings.guided_rediffusion_steps
                            / self.settings.guided_rediffusion_strength
                        ),
                        "strength": self.settings.guided_rediffusion_strength,
                        "controlnet_scale": (self.settings.guided_rediffusion_controlnet_scale),
                        "initial_module_error_rate": guided.initial_module_error_rate,
                        "control_module_error_rate": guided.control_module_error_rate,
                        "final_module_error_rate": guided.final_module_error_rate,
                        "mask_coverage": guided.mask_coverage,
                        "changed_pixel_ratio": guided.changed_pixel_ratio,
                        "mean_absolute_change": guided.mean_absolute_change,
                        "unprojected_changed_pixel_ratio": (guided.unprojected_changed_pixel_ratio),
                        "unprojected_mean_absolute_change": (
                            guided.unprojected_mean_absolute_change
                        ),
                        "accepted": guided.accepted,
                        "rejection_reason": guided.rejection_reason,
                    },
                )
                if guided.accepted:
                    enhanced_candidate = guided.image
                    enhanced_prefix = "guided_"
                    yield "guided", guided.image
            except Exception:
                duration = time.perf_counter() - started
                metrics.GUIDED_REDIFFUSIONS.labels("error").inc()
                metrics.GUIDED_REDIFFUSION_DURATION.observe(duration)
                metrics.DURATION.labels("controlnet", "guided_rediffusion").observe(duration)
                logger.exception(
                    "guided_rediffusion_failed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": "error",
                        "attempt": attempt,
                        "seed": seed,
                        "duration_ms": round(duration * 1000, 2),
                    },
                )
        if self.settings.latent_refinement_enabled:
            started = time.perf_counter()
            try:
                result = refine_candidate_latent(
                    self._load(),
                    enhanced_candidate,
                    blueprint,
                    LatentRefinementConfig(
                        iterations=self.settings.latent_refinement_iterations,
                        learning_rate=self.settings.latent_refinement_learning_rate,
                        qr_weight=self.settings.latent_refinement_qr_weight,
                        preservation_weight=(self.settings.latent_refinement_preservation_weight),
                        functional_weight=self.settings.latent_refinement_functional_weight,
                        target_module_error_rate=(
                            self.settings.latent_refinement_target_module_error_rate
                        ),
                        max_latent_delta=(self.settings.latent_refinement_max_latent_delta),
                        max_mean_absolute_change=(
                            self.settings.latent_refinement_max_mean_absolute_change
                        ),
                        min_relative_module_improvement=(
                            self.settings.latent_refinement_min_relative_module_improvement
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
                        else ("rejected_preservation" if result.improved else "no_improvement")
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
                metrics.LATENT_REFINEMENT_LOSS.labels("best_observed_mean_absolute_change").set(
                    result.best_observed_mean_absolute_change
                )
                logger.info(
                    "latent_refinement_completed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": outcome,
                        "attempt": attempt,
                        "seed": seed,
                        "duration_ms": round(duration * 1000, 2),
                        "iterations": result.iterations,
                        "initial_module_error_rate": result.initial_module_error_rate,
                        "final_module_error_rate": result.final_module_error_rate,
                        "srl": result.final_srl,
                        "preservation_loss": result.final_preservation_loss,
                        "mean_absolute_change": result.final_mean_absolute_change,
                        "best_observed_module_error_rate": (result.best_observed_module_error_rate),
                        "best_observed_mean_absolute_change": (
                            result.best_observed_mean_absolute_change
                        ),
                        "actual_initial_module_error_rate": (
                            result.actual_initial_module_error_rate
                        ),
                        "actual_final_module_error_rate": (result.actual_final_module_error_rate),
                        "improved": result.improved,
                        "accepted": result.accepted,
                        "converged": result.converged,
                        "rejection_reason": result.rejection_reason,
                    },
                )
                if result.accepted:
                    enhanced_candidate = result.image
                    enhanced_prefix = f"{enhanced_prefix}latent_"
                    yield f"{enhanced_prefix}srl", result.image
            except Exception:
                duration = time.perf_counter() - started
                metrics.LATENT_REFINEMENTS.labels("error").inc()
                metrics.LATENT_REFINEMENT_DURATION.observe(duration)
                logger.exception(
                    "latent_refinement_failed",
                    extra={
                        "run_id": run_id,
                        "backend": "controlnet",
                        "status": "error",
                        "attempt": attempt,
                        "seed": seed,
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

        # An enhanced candidate that does not scan on its own can still need fewer visible
        # module repairs than the raw image. Try every targeted profile on that improved
        # base first, then retain the complete raw repair chain as a reliability fallback.
        if enhanced_candidate is not candidate:
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
                if name in GLOBAL_REPAIR_VARIANTS:
                    continue
                yield (
                    f"{enhanced_prefix}{name}",
                    repair_qr_modules(
                        enhanced_candidate,
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
