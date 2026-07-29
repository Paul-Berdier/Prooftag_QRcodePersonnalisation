from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable

import numpy as np
from PIL import Image

from . import metrics
from .config import Settings
from .qr import QRBlueprint
from .schemas import GenerationRequest

logger = logging.getLogger(__name__)


def _tensor_to_pil(tensor) -> Image.Image:
    array = (
        tensor[0]
        .detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def _pil_to_tensor(image: Image.Image, *, device: str, dtype):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )


def _patch_upstream_perceptual_gradient() -> None:
    """Keep the public loss differentiable without changing its mathematics.

    Upstream wraps scalar tensors with ``torch.tensor([...])``, which detaches
    every VGG loss from the graph. ``torch.stack`` is the intended equivalent.
    """
    import torch
    from diffqrcoder.losses.perceptual_loss import PerceptualLoss

    if getattr(PerceptualLoss, "_prooftag_gradient_patch", False):
        return

    def forward(self, x, y):
        losses = [
            torch.nn.functional.mse_loss(fx, fy)
            for fx, fy in zip(self.extractor(x), self.extractor(y), strict=True)
        ]
        return torch.stack(losses).mean()

    PerceptualLoss.forward = forward
    PerceptualLoss._prooftag_gradient_patch = True


class UpstreamDiffQRCoderBackend:
    """Pinned public DiffQRCoder Stage 1, Stage 2 SRPG and SR-MPGD only."""

    _load_lock = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._debug_artifacts: dict[str, Image.Image] = {}
        self._diagnostics: dict[str, float] = {}

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            started = time.perf_counter()
            try:
                import torch
                from diffqrcoder import DiffQRCoderPipeline
                from diffusers import ControlNetModel, DDIMScheduler

                if not torch.cuda.is_available():
                    raise RuntimeError("DiffQRCoder requires an available CUDA GPU")
                _patch_upstream_perceptual_gradient()
                controlnet_arguments = {
                    "torch_dtype": torch.float16,
                    "cache_dir": self.settings.model_cache_dir,
                }
                if self.settings.controlnet_model_subfolder:
                    controlnet_arguments["subfolder"] = self.settings.controlnet_model_subfolder
                controlnet = ControlNetModel.from_pretrained(
                    self.settings.controlnet_model_id,
                    **controlnet_arguments,
                )
                pipe = DiffQRCoderPipeline.from_single_file(
                    self.settings.base_model_id,
                    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    cache_dir=self.settings.model_cache_dir,
                    safety_checker=None,
                    use_safetensors=True,
                )
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                pipe.set_progress_bar_config(disable=True)
                pipe.to(self.settings.device)
                pipe.unet.requires_grad_(False).eval()
                pipe.controlnet.requires_grad_(False).eval()
                pipe.vae.requires_grad_(False).eval()
                pipe.text_encoder.requires_grad_(False).eval()
                self._pipeline = pipe
            except Exception:
                metrics.MODEL_LOADS.labels("error").inc()
                logger.exception("diffqrcoder_model_load_failed")
                raise
            duration = time.perf_counter() - started
            metrics.MODEL_LOADS.labels("success").inc()
            metrics.MODEL_LOAD_DURATION.labels("success").observe(duration)
            metrics.MODEL_LOADED.set(1)
            logger.info(
                "diffqrcoder_model_loaded",
                extra={
                    "revision": self.settings.diffqrcoder_revision,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
        return self._pipeline

    def control_image(self, blueprint: QRBlueprint) -> Image.Image:
        return blueprint.image.convert("RGB")

    def generate(
        self, request: GenerationRequest, blueprint: QRBlueprint, seed: int
    ) -> Image.Image:
        import torch

        pipe = self._load()
        generator = torch.Generator(device=self.settings.device).manual_seed(seed)
        output = pipe._run_stage1(
            prompt=request.prompt,
            qrcode=self.control_image(blueprint),
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            eta=self.settings.srpg_eta,
            generator=generator,
            controlnet_conditioning_scale=request.controlnet_scale,
            control_guidance_start=self.settings.diffqrcoder_control_guidance_start,
            control_guidance_end=self.settings.diffqrcoder_control_guidance_end,
            output_type="pt",
        )
        image = _tensor_to_pil(output.images)
        self._diagnostics = {
            "diffqrcoder_stage1_steps": float(request.steps),
            "diffqrcoder_stage1_seed": float(seed),
        }
        return image

    def _decode_latent(self, pipe, latent) -> Image.Image:
        import torch

        with torch.no_grad():
            decoded = pipe.vae.decode(
                latent.detach() / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            decoded = pipe.image_processor.denormalize(decoded.detach())
        return _tensor_to_pil(decoded)

    def _run_stage2(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        request: GenerationRequest,
        seed: int,
    ) -> Image.Image:
        import torch

        pipe = self._load()
        self._debug_artifacts.clear()
        stage2_seed = (seed + self.settings.srpg_seed_offset) % (2**32)
        generator = torch.Generator(device=self.settings.device).manual_seed(stage2_seed)
        reference = _pil_to_tensor(
            candidate,
            device=self.settings.device,
            dtype=pipe.unet.dtype,
        )
        preview_interval = self.settings.srpg_preview_interval

        def callback(current_pipe, index, timestep, values):
            if self.settings.srpg_save_step_previews and (
                index % preview_interval == 0 or index + 1 == self.settings.srpg_steps
            ):
                self._debug_artifacts[f"stage2_step_{index + 1:03d}"] = (
                    self._decode_latent(current_pipe, values["latents"])
                )
            return values

        iterations = (
            self.settings.srmpgd_max_iterations if self.settings.srmpgd_enabled else None
        )
        output = pipe._run_stage2(
            prompt=request.prompt,
            qrcode=self.control_image(blueprint),
            qrcode_module_size=self.settings.diffqrcoder_qr_module_size,
            qrcode_padding=self.settings.diffqrcoder_qr_padding_px,
            ref_image=reference,
            negative_prompt=request.negative_prompt or None,
            num_inference_steps=self.settings.srpg_steps,
            guidance_scale=request.guidance_scale,
            eta=self.settings.srpg_eta,
            generator=generator,
            controlnet_conditioning_scale=self.settings.srpg_controlnet_scale,
            control_guidance_start=self.settings.diffqrcoder_control_guidance_start,
            control_guidance_end=self.settings.diffqrcoder_control_guidance_end,
            scanning_robust_guidance_scale=self.settings.srpg_qr_weight,
            perceptual_guidance_scale=self.settings.srpg_perceptual_weight,
            srmpgd_num_iteration=iterations,
            srmpgd_lr=self.settings.srmpgd_step_size,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="latent",
        )
        latent = output.images.detach()
        image = self._decode_latent(pipe, latent)
        self._diagnostics.update(
            {
                "diffqrcoder_revision_verified": 1.0,
                "diffqrcoder_stage2_seed": float(stage2_seed),
                "diffqrcoder_stage2_steps": float(self.settings.srpg_steps),
                "diffqrcoder_srg": float(self.settings.srpg_qr_weight),
                "diffqrcoder_pg": float(self.settings.srpg_perceptual_weight),
                "diffqrcoder_srmpgd_iterations": float(iterations or 0),
                "diffqrcoder_srmpgd_lr": float(self.settings.srmpgd_step_size),
            }
        )
        del output, latent, reference
        torch.cuda.empty_cache()
        return image

    def variants(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        *,
        request: GenerationRequest | None = None,
        seed: int | None = None,
        **_,
    ) -> Iterable[tuple[str, Image.Image]]:
        yield "raw", candidate
        if not self.settings.srpg_enabled:
            return
        if request is None or seed is None:
            raise ValueError("request and seed are required for DiffQRCoder Stage 2")
        image = self._run_stage2(candidate, blueprint, request, seed)
        yield ("srmpgd" if self.settings.srmpgd_enabled else "srpg"), image

    def debug_artifacts(self) -> dict[str, Image.Image]:
        return self._debug_artifacts.copy()

    def diagnostics(self) -> dict[str, float]:
        return self._diagnostics.copy()
