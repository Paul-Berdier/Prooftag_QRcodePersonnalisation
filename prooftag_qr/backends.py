from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

from PIL import Image

from . import metrics
from .config import Settings
from .qr import QRBlueprint, repair_qr_modules
from .schemas import GenerationRequest

logger = logging.getLogger(__name__)


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
        repair_profiles = (
            ("functional", 0.0, False, False),
            ("incorrect_55", 0.55, True, False),
            ("incorrect_72", 0.72, True, False),
            ("incorrect_80", 0.80, True, False),
            ("incorrect_85", 0.85, True, False),
            ("incorrect_90", 0.90, True, False),
            ("incorrect_100", 1.00, True, False),
            ("centers_45", 0.45, False, False),
            ("centers_60", 0.60, False, False),
            ("centers_72", 0.72, False, False),
            ("centers_85", 0.85, False, False),
            ("tonal_90", 0.90, False, True),
            ("tonal_95", 0.95, False, True),
            ("centers_90", 0.90, False, False),
            ("centers_95", 0.95, False, False),
        )
        for name, center_scale, incorrect_only, preserve_tone in repair_profiles:
            yield (
                name,
                repair_qr_modules(
                    candidate,
                    blueprint,
                    center_scale=center_scale,
                    incorrect_only=incorrect_only,
                    preserve_tone=preserve_tone,
                ),
            )


def build_backends(settings: Settings) -> dict[str, GenerationBackend]:
    return {"qr": QRBackend(), "controlnet": ControlNetBackend(settings)}
