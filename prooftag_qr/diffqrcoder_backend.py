from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from types import MethodType

import numpy as np
from PIL import Image

from . import metrics
from .config import Settings
from .qr import QRBlueprint, functional_pattern_mask
from .quality import image_change_metrics, image_quality_metrics
from .schemas import GenerationRequest
from .srmpgd import SRMPGDConfig, run_srmpgd

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


def _tone_to_luminance(region: np.ndarray, target: float) -> np.ndarray:
    """Move a coloured patch toward a target luminance while preserving its hue."""
    source = region.astype(np.float32) / 255.0
    luminance = (
        0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
    )
    current = float(luminance.mean())
    if current > target:
        source *= target / max(current, 1e-6)
    elif current < target:
        blend = (target - current) / max(1.0 - current, 1e-6)
        source = source * (1.0 - blend) + blend
    return np.rint(source.clip(0, 1) * 255).astype(np.uint8)


def build_paper_qart_target(
    reference: Image.Image,
    blueprint: QRBlueprint,
    *,
    padding_px: int,
    module_size: int,
    center_fraction: float,
    dark_target: float,
    light_target: float,
) -> Image.Image:
    """Build the missing paper target ``Qart(x_hat, y)`` deterministically.

    The public repository consumes a QArt target but does not publish the QArt
    constructor used in the paper. This implementation preserves the Stage-1
    artwork, moves only data-module centres across the paper's robust thresholds,
    and copies functional modules exactly as the upstream personalized-code tool
    does. It is explicitly reported as a reconstructed target, not upstream code.
    """
    if reference.width != reference.height:
        raise ValueError("DiffQRCoder QArt target requires a square Stage-1 image")
    border = int(blueprint.border)
    core_matrix = (
        blueprint.matrix[border:-border, border:-border]
        if border
        else blueprint.matrix
    )
    core_functional = (
        functional_pattern_mask(blueprint)[border:-border, border:-border]
        if border
        else functional_pattern_mask(blueprint)
    )
    expected_core = core_matrix.shape[0] * module_size
    if padding_px * 2 + expected_core != reference.width:
        raise ValueError(
            "QArt geometry mismatch: "
            f"{padding_px}*2 + {core_matrix.shape[0]}*{module_size} "
            f"!= {reference.width}"
        )
    output = np.asarray(reference.convert("RGB"), dtype=np.uint8).copy()
    center_side = max(1, round(module_size * center_fraction))
    for row in range(core_matrix.shape[0]):
        y0 = padding_px + row * module_size
        y1 = y0 + module_size
        for col in range(core_matrix.shape[1]):
            x0 = padding_px + col * module_size
            x1 = x0 + module_size
            target_dark = bool(core_matrix[row, col])
            target_value = 0 if target_dark else 255
            if core_functional[row, col]:
                output[y0:y1, x0:x1] = target_value
                continue
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            rx0 = max(x0, cx - center_side // 2)
            ry0 = max(y0, cy - center_side // 2)
            rx1 = min(x1, rx0 + center_side)
            ry1 = min(y1, ry0 + center_side)
            output[ry0:ry1, rx0:rx1] = _tone_to_luminance(
                output[ry0:ry1, rx0:rx1],
                dark_target if target_dark else light_target,
            )
    return Image.fromarray(output, mode="RGB")


def _qart_center_error_rate(
    target: Image.Image,
    blueprint: QRBlueprint,
    *,
    padding_px: int,
    module_size: int,
) -> float:
    gray = np.asarray(target.convert("L"), dtype=np.float32) / 255.0
    border = int(blueprint.border)
    matrix = (
        blueprint.matrix[border:-border, border:-border]
        if border
        else blueprint.matrix
    )
    errors = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            cy = padding_px + row * module_size + module_size // 2
            cx = padding_px + col * module_size + module_size // 2
            errors.append((gray[cy, cx] < 0.5) != bool(matrix[row, col]))
    return float(np.mean(errors))


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


@contextmanager
def _preserve_partial_schedule_stride(pipe, *, base_steps: int, timesteps):
    """Keep the upstream manual DDIM update aligned with a truncated schedule.

    DiffQRCoder computes the previous timestep from
    ``scheduler.num_inference_steps``. Diffusers replaces that value with the
    length of a custom timestep list, although our list is a suffix of the
    original ``base_steps`` schedule. Without this compatibility wrapper, a
    partial Stage-2 restart uses the wrong alpha pair.
    """
    if timesteps is None or len(timesteps) == base_steps:
        yield
        return

    scheduler = pipe.scheduler
    original_set_timesteps = scheduler.set_timesteps

    def set_timesteps_with_original_stride(self, *args, **kwargs):
        result = original_set_timesteps(*args, **kwargs)
        if kwargs.get("timesteps") is not None:
            self.num_inference_steps = base_steps
        return result

    scheduler.set_timesteps = MethodType(
        set_timesteps_with_original_stride,
        scheduler,
    )
    try:
        yield
    finally:
        scheduler.set_timesteps = original_set_timesteps


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

    def _encode_stage1_latent(self, pipe, candidate: Image.Image):
        import torch

        image = pipe.image_processor.preprocess(candidate).to(
            device=self.settings.device,
            dtype=pipe.vae.dtype,
        )
        with torch.no_grad():
            posterior = pipe.vae.encode(image, return_dict=True).latent_dist
            latent = posterior.mode() * pipe.vae.config.scaling_factor
        return latent.detach()

    def _paper_stage2_initial_latent(
        self,
        pipe,
        candidate: Image.Image,
        *,
        generator,
    ):
        """Equation 9: noise the encoded Stage-1 image at the selected DDIM time."""
        import torch

        steps = self.settings.srpg_steps
        pipe.scheduler.set_timesteps(steps, device=self.settings.device)
        all_timesteps = pipe.scheduler.timesteps.detach().clone()
        effective_steps = max(
            1,
            min(steps, round(steps * self.settings.diffqrcoder_stage2_strength)),
        )
        start_index = steps - effective_steps
        timesteps = all_timesteps[start_index:]
        clean_latent = self._encode_stage1_latent(pipe, candidate)
        noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
        )
        first_timestep = timesteps[0].reshape(1)
        initial = pipe.scheduler.add_noise(clean_latent, noise, first_timestep)
        timestep_index = int(first_timestep.item())
        alpha = pipe.scheduler.alphas_cumprod[timestep_index].float().to(
            device=clean_latent.device
        )
        return (
            initial.detach(),
            [int(item) for item in timesteps.detach().cpu().tolist()],
            {
                "diffqrcoder_stage2_effective_steps": float(effective_steps),
                "diffqrcoder_stage2_start_timestep": float(first_timestep.item()),
                "diffqrcoder_stage2_reference_coefficient": float(alpha.sqrt().cpu()),
                "diffqrcoder_stage2_noise_coefficient": float(
                    (1.0 - alpha).sqrt().cpu()
                ),
            },
        )

    def _stage2_target(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
    ) -> Image.Image:
        if not self.settings.diffqrcoder_qart_enabled:
            return self.control_image(blueprint)
        return build_paper_qart_target(
            candidate,
            blueprint,
            padding_px=self.settings.diffqrcoder_qr_padding_px,
            module_size=self.settings.diffqrcoder_qr_module_size,
            center_fraction=self.settings.diffqrcoder_qart_center_fraction,
            dark_target=self.settings.diffqrcoder_qart_dark_target,
            light_target=self.settings.diffqrcoder_qart_light_target,
        )

    def _record_divergence_guard(
        self,
        image: Image.Image,
        candidate: Image.Image,
    ) -> None:
        change = image_change_metrics(image, candidate)
        quality = image_quality_metrics(image)
        reasons = {
            "changed_pixels": (
                change["changed_pixel_ratio"]
                > self.settings.diffqrcoder_guard_max_changed_pixel_ratio
            ),
            "mean_absolute_change": (
                change["mean_absolute_change"]
                > self.settings.diffqrcoder_guard_max_mean_absolute_change
            ),
            "clipped_pixels": (
                quality["clipped_pixel_ratio"]
                > self.settings.diffqrcoder_guard_max_clipped_pixel_ratio
            ),
        }
        self._diagnostics.update(
            {
                "diffqrcoder_guard_diverged": float(any(reasons.values())),
                "diffqrcoder_stage2_changed_pixel_ratio": float(
                    change["changed_pixel_ratio"]
                ),
                "diffqrcoder_stage2_mean_absolute_change": float(
                    change["mean_absolute_change"]
                ),
                "diffqrcoder_stage2_clipped_pixel_ratio": float(
                    quality["clipped_pixel_ratio"]
                ),
                "diffqrcoder_guard_changed_pixels": float(
                    reasons["changed_pixels"]
                ),
                "diffqrcoder_guard_mean_absolute_change": float(
                    reasons["mean_absolute_change"]
                ),
                "diffqrcoder_guard_clipped_pixels": float(
                    reasons["clipped_pixels"]
                ),
            }
        )

    def _run_stage2(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        request: GenerationRequest,
        seed: int,
        *,
        validation_callback=None,
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
        stage2_target = self._stage2_target(candidate, blueprint)
        self._debug_artifacts["stage2_reference"] = candidate.copy()
        self._debug_artifacts["stage2_control_target"] = stage2_target.copy()
        initial_latent = None
        custom_timesteps = None
        initialization_diagnostics = {
            "diffqrcoder_stage2_paper_initialization": 0.0,
            "diffqrcoder_stage2_effective_steps": float(self.settings.srpg_steps),
        }
        if (
            self.settings.diffqrcoder_stage2_initialization
            == "paper_stage1_noise"
        ):
            (
                initial_latent,
                custom_timesteps,
                initialization_diagnostics,
            ) = self._paper_stage2_initial_latent(
                pipe,
                candidate,
                generator=generator,
            )
            initialization_diagnostics[
                "diffqrcoder_stage2_paper_initialization"
            ] = 1.0
        effective_steps = int(
            initialization_diagnostics["diffqrcoder_stage2_effective_steps"]
        )
        preview_interval = self.settings.srpg_preview_interval

        def callback(current_pipe, index, timestep, values):
            if self.settings.srpg_save_step_previews and (
                index % preview_interval == 0 or index + 1 == effective_steps
            ):
                self._debug_artifacts[f"stage2_step_{index + 1:03d}"] = (
                    self._decode_latent(current_pipe, values["latents"])
                )
            return values

        with _preserve_partial_schedule_stride(
            pipe,
            base_steps=self.settings.srpg_steps,
            timesteps=custom_timesteps,
        ):
            output = pipe._run_stage2(
                prompt=request.prompt,
                qrcode=stage2_target,
                qrcode_module_size=self.settings.diffqrcoder_qr_module_size,
                qrcode_padding=self.settings.diffqrcoder_qr_padding_px,
                ref_image=reference,
                negative_prompt=request.negative_prompt or None,
                num_inference_steps=effective_steps,
                timesteps=custom_timesteps,
                guidance_scale=request.guidance_scale,
                eta=self.settings.srpg_eta,
                generator=generator,
                latents=initial_latent,
                controlnet_conditioning_scale=self.settings.srpg_controlnet_scale,
                control_guidance_start=self.settings.diffqrcoder_control_guidance_start,
                control_guidance_end=self.settings.diffqrcoder_control_guidance_end,
                scanning_robust_guidance_scale=self.settings.srpg_qr_weight,
                perceptual_guidance_scale=self.settings.srpg_perceptual_weight,
                srmpgd_num_iteration=None,
                callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents"],
                output_type="latent",
            )
        latent = output.images.detach()
        image = self._decode_latent(pipe, latent)
        self._debug_artifacts["stage2_before_srmpgd"] = image.copy()
        self._diagnostics.update(
            {
                "diffqrcoder_revision_verified": 1.0,
                "diffqrcoder_stage2_seed": float(stage2_seed),
                "diffqrcoder_stage2_steps": float(effective_steps),
                "diffqrcoder_srg": float(self.settings.srpg_qr_weight),
                "diffqrcoder_pg": float(self.settings.srpg_perceptual_weight),
                "diffqrcoder_qart_enabled": float(
                    self.settings.diffqrcoder_qart_enabled
                ),
                "diffqrcoder_qart_center_fraction": float(
                    self.settings.diffqrcoder_qart_center_fraction
                ),
                "diffqrcoder_qart_center_error_rate": _qart_center_error_rate(
                    stage2_target,
                    blueprint,
                    padding_px=self.settings.diffqrcoder_qr_padding_px,
                    module_size=self.settings.diffqrcoder_qr_module_size,
                ),
                "diffqrcoder_srmpgd_iterations": 0.0,
                "diffqrcoder_srmpgd_gamma": 0.0,
                "diffqrcoder_srmpgd_lpips_weight": 0.0,
                **initialization_diagnostics,
            }
        )
        if self.settings.srmpgd_enabled:
            def preview_srmpgd(preview_image, step):
                if self.settings.srpg_save_step_previews:
                    self._debug_artifacts[
                        f"srmpgd_iteration_{step.iteration:03d}"
                    ] = preview_image.copy()

            def paper_scanning_loss(decoded, target):
                with torch.autocast("cuda", dtype=pipe.unet.dtype):
                    return pipe.srpg.scanning_robust_loss_fn(decoded, target)

            srmpgd = run_srmpgd(
                pipe,
                latent,
                blueprint,
                SRMPGDConfig(
                    max_iterations=self.settings.srmpgd_max_iterations,
                    step_size=self.settings.srmpgd_step_size,
                    lpips_weight=self.settings.srmpgd_lpips_weight,
                    lpips_net=self.settings.srmpgd_lpips_net,
                    crop_padding_px=self.settings.srmpgd_crop_padding_px,
                    dark_threshold=self.settings.srmpgd_dark_threshold,
                    light_threshold=self.settings.srmpgd_light_threshold,
                    center_fraction=self.settings.srmpgd_center_fraction,
                    max_initial_module_error_rate=(
                        self.settings.srmpgd_max_initial_module_error_rate
                    ),
                    quiet_zone_mode="none",
                    functional_pattern_tone_factor=0.0,
                ),
                scanning_loss=paper_scanning_loss,
                validation_callback=validation_callback,
                preview_callback=preview_srmpgd,
            )
            image = srmpgd.image
            self._diagnostics.update(
                {
                    "diffqrcoder_srmpgd_iterations": float(
                        len(srmpgd.steps) - 1
                    ),
                    "diffqrcoder_srmpgd_gamma": float(
                        self.settings.srmpgd_step_size
                    ),
                    "diffqrcoder_srmpgd_lpips_weight": float(
                        self.settings.srmpgd_lpips_weight
                    ),
                    "diffqrcoder_srmpgd_selected_iteration": float(
                        srmpgd.selected_iteration
                    ),
                    "diffqrcoder_srmpgd_initial_mer": float(
                        srmpgd.initial_module_error_rate
                    ),
                    "diffqrcoder_srmpgd_final_mer": float(
                        srmpgd.final_module_error_rate
                    ),
                    "diffqrcoder_srmpgd_strict_selected": float(
                        srmpgd.steps[srmpgd.selected_iteration].strict_all
                    ),
                    "diffqrcoder_srmpgd_stopped_initial_mer": float(
                        srmpgd.stop_reason
                        == "initial_module_error_rate_above_limit"
                    ),
                    "diffqrcoder_srmpgd_stopped_non_finite": float(
                        srmpgd.stop_reason.startswith("non_finite_")
                    ),
                }
            )
            self._debug_artifacts["srmpgd_selected"] = image.copy()
            del srmpgd
        self._record_divergence_guard(image, candidate)
        del output, latent, reference, initial_latent
        torch.cuda.empty_cache()
        return image

    def variants(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        *,
        request: GenerationRequest | None = None,
        seed: int | None = None,
        validation_callback=None,
        **_,
    ) -> Iterable[tuple[str, Image.Image]]:
        yield "raw", candidate
        if not self.settings.srpg_enabled:
            return
        if request is None or seed is None:
            raise ValueError("request and seed are required for DiffQRCoder Stage 2")
        image = self._run_stage2(
            candidate,
            blueprint,
            request,
            seed,
            validation_callback=validation_callback,
        )
        yield ("srmpgd" if self.settings.srmpgd_enabled else "srpg"), image

    def debug_artifacts(self) -> dict[str, Image.Image]:
        return self._debug_artifacts.copy()

    def diagnostics(self) -> dict[str, float]:
        return self._diagnostics.copy()
