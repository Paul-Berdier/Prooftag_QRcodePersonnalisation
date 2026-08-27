from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from types import MethodType

import numpy as np
from PIL import Image

from . import metrics
from .blueprints import canonical_url_match
from .config import Settings
from .model_sources import resolve_single_file_sources
from .qart import build_qart_target
from .qr import QRBlueprint, diffqrcoder_module_error_rate, module_error_rate
from .quality import image_change_metrics, image_quality_metrics, image_sha256
from .schemas import GenerationRequest
from .srmpgd import SRMPGDConfig, run_srmpgd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Stage2Control:
    image: Image.Image
    blueprint: QRBlueprint
    match_mode: str


def _tensor_to_pil(tensor) -> Image.Image:
    array = tensor[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def _pil_to_tensor(image: Image.Image, *, device: str, dtype):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def _tensor_sha256(tensor) -> str:
    source = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{source.dtype}:{tuple(source.shape)}:".encode())
    digest.update(source.numpy().tobytes())
    return digest.hexdigest()


def _control_target_center_error_rate(
    target: Image.Image,
    blueprint: QRBlueprint,
    *,
    padding_px: int,
    module_size: int,
) -> float:
    gray = np.asarray(target.convert("L"), dtype=np.float32) / 255.0
    border = int(blueprint.border)
    matrix = blueprint.matrix[border:-border, border:-border] if border else blueprint.matrix
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
def _install_partial_schedule(
    pipe,
    *,
    base_steps: int,
    effective_steps: int,
):
    """Install a truncated DDIM schedule without the unsupported custom API.

    Diffusers 0.32 rejects ``timesteps=[...]`` for ``DDIMScheduler``. The
    upstream loop nevertheless supports a suffix of the normal schedule if it
    is installed directly on the scheduler. Keeping ``num_inference_steps`` at
    ``base_steps`` is also required because DiffQRCoder computes its previous
    alpha manually from that value.
    """
    if effective_steps == base_steps:
        yield
        return

    scheduler = pipe.scheduler
    original_set_timesteps = scheduler.set_timesteps

    def set_truncated_timesteps(self, num_inference_steps, *args, **kwargs):
        result = original_set_timesteps(base_steps, *args, **kwargs)
        self.timesteps = self.timesteps[base_steps - effective_steps :]
        self.num_inference_steps = base_steps
        return result

    scheduler.set_timesteps = MethodType(
        set_truncated_timesteps,
        scheduler,
    )
    try:
        yield
    finally:
        scheduler.set_timesteps = original_set_timesteps


@contextmanager
def _offload_unused_pipeline_modules_for_paper_srmpgd(pipe, *, lpips_net: str):
    """Free diffusion-only CUDA weights while Eq. 13-14 uses VAE + LPIPS.

    The E033 FP16 and FP32 branches reached 19.35--19.55 GiB on a 20 GiB RTX because
    UNet, ControlNet and the text encoder remained resident after Stage 2. None of these
    modules participates in the post-processing objective. Move only those modules to CPU,
    keep the VAE on CUDA, then restore the original devices for a reusable pipeline.
    """

    import torch

    moved: list[tuple[str, object, object]] = []
    primary_error: BaseException | None = None
    restoration_errors: list[str] = []
    memory_allocated = getattr(torch.cuda, "memory_allocated", None)
    memory_info = getattr(torch.cuda, "mem_get_info", None)
    reset_peak_memory_stats = getattr(torch.cuda, "reset_peak_memory_stats", None)
    max_memory_allocated = getattr(torch.cuda, "max_memory_allocated", None)

    def driver_free_bytes():
        if not callable(memory_info):
            return None
        try:
            return int(memory_info()[0])
        except Exception:
            return None

    state = {
        "offloaded_modules": (),
        "cuda_allocated_before_bytes": (
            int(memory_allocated()) if callable(memory_allocated) else None
        ),
        "cuda_driver_free_before_bytes": driver_free_bytes(),
        "cuda_allocated_after_offload_bytes": None,
        "cuda_driver_free_after_offload_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_allocated_before_restore_bytes": None,
        "cuda_allocated_after_lpips_offload_bytes": None,
        "cuda_allocated_after_restore_bytes": None,
    }
    try:
        for name in (
            "unet",
            "controlnet",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
            "srpg",
        ):
            module = getattr(pipe, name, None)
            if module is None or not hasattr(module, "parameters"):
                continue
            parameter = next(iter(module.parameters()), None)
            device = getattr(parameter, "device", None) if parameter is not None else None
            if getattr(device, "type", None) != "cuda":
                continue
            # Record before moving so a partially failed ``to`` is restored as well.
            moved.append((name, module, device))
            module.to(device="cpu")
        if moved and torch.cuda.is_available():
            torch.cuda.empty_cache()
        state["offloaded_modules"] = tuple(name for name, _, _ in moved)
        state["cuda_allocated_after_offload_bytes"] = (
            int(memory_allocated()) if callable(memory_allocated) else None
        )
        state["cuda_driver_free_after_offload_bytes"] = driver_free_bytes()
        if torch.cuda.is_available() and callable(reset_peak_memory_stats):
            reset_peak_memory_stats()
        yield state
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        state["cuda_peak_allocated_bytes"] = (
            int(max_memory_allocated()) if callable(max_memory_allocated) else None
        )
        state["cuda_allocated_before_restore_bytes"] = (
            int(memory_allocated()) if callable(memory_allocated) else None
        )
        # LPIPS is cached by run_srmpgd. Return it to CPU before reloading the diffusion
        # modules, otherwise the restoration can create a second transient memory peak.
        cached_lpips = getattr(pipe, f"_prooftag_srmpgd_lpips_{lpips_net}", None)
        if cached_lpips is not None:
            try:
                cached_lpips.to(device="cpu")
            except Exception as exc:
                restoration_errors.append(f"LPIPS: {type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state["cuda_allocated_after_lpips_offload_bytes"] = (
            int(memory_allocated()) if callable(memory_allocated) else None
        )
        for name, module, device in reversed(moved):
            try:
                module.to(device=device)
            except Exception as exc:
                restoration_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state["cuda_allocated_after_restore_bytes"] = (
            int(memory_allocated()) if callable(memory_allocated) else None
        )
        if restoration_errors:
            message = "failed to restore SR-MPGD pipeline modules: " + "; ".join(restoration_errors)
            if primary_error is not None:
                primary_error.add_note(message)
            else:
                raise RuntimeError(message)


class UpstreamDiffQRCoderBackend:
    """Pinned public DiffQRCoder Stage 1, Stage 2 SRPG and SR-MPGD only."""

    _load_lock = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None
        self._debug_artifacts: dict[str, Image.Image] = {}
        self._debug_metadata: dict[str, object] = {}
        self._diagnostics: dict[str, float] = {}
        self._stage2_control: _Stage2Control | None = None
        self._stage2_override: dict | None = None
        self._last_stage2_state: dict | None = None
        self._stage2_latent_sha256: str | None = None
        self._stage2_source_run_id: str | None = None
        self._stage2_source_method_id: str | None = None
        self._stage2_pairing_status: str | None = None
        self._srmpgd_stop_reason: str | None = None
        self._srmpgd_selected_iteration: int | None = None
        self._srmpgd_stage2_image_sha256: str | None = None
        self._srmpgd_selected_image_sha256: str | None = None

    def import_stage2_state(self, state: dict) -> None:
        actual_sha256 = _tensor_sha256(state["latent"])
        expected_sha256 = state.get("latent_sha256", actual_sha256)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "cached DiffQRCoder Stage 2 latent hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        self._stage2_override = {
            "latent": state["latent"].clone(),
            "image": state["image"].copy(),
            "reference": state["reference"].copy(),
            "control": state["control"],
            "diagnostics": dict(state["diagnostics"]),
            "latent_sha256": actual_sha256,
            "source_run_id": state.get("source_run_id"),
            "source_method_id": state.get("source_method_id"),
        }

    def export_stage2_state(self) -> dict | None:
        if self._last_stage2_state is None:
            return None
        state = self._last_stage2_state
        latent = state["latent"].clone()
        return {
            "latent": latent,
            "image": state["image"].copy(),
            "reference": state["reference"].copy(),
            "control": state["control"],
            "diagnostics": dict(state["diagnostics"]),
            "latent_sha256": _tensor_sha256(latent),
            "source_run_id": state.get("source_run_id"),
            "source_method_id": state.get("source_method_id"),
        }

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
                if self.settings.controlnet_model_revision:
                    controlnet_arguments["revision"] = self.settings.controlnet_model_revision
                if self.settings.controlnet_model_subfolder:
                    controlnet_arguments["subfolder"] = self.settings.controlnet_model_subfolder
                controlnet = ControlNetModel.from_pretrained(
                    self.settings.controlnet_model_id,
                    **controlnet_arguments,
                )
                checkpoint_path, config_path = resolve_single_file_sources(self.settings)
                pipeline_arguments = {
                    "config": config_path,
                    "controlnet": controlnet,
                    "torch_dtype": torch.float16,
                    "cache_dir": self.settings.model_cache_dir,
                    "safety_checker": None,
                    "use_safetensors": True,
                }
                pipe = DiffQRCoderPipeline.from_single_file(
                    checkpoint_path,
                    **pipeline_arguments,
                )
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                pipe._callback_tensor_inputs = list(
                    dict.fromkeys([*pipe._callback_tensor_inputs, "original_image"])
                )
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
        alpha = pipe.scheduler.alphas_cumprod[timestep_index].float().to(device=clean_latent.device)
        return (
            initial.detach(),
            {
                "diffqrcoder_stage2_effective_steps": float(effective_steps),
                "diffqrcoder_stage2_start_timestep": float(first_timestep.item()),
                "diffqrcoder_stage2_reference_coefficient": float(alpha.sqrt().cpu()),
                "diffqrcoder_stage2_noise_coefficient": float((1.0 - alpha).sqrt().cpu()),
            },
        )

    def _stage2_target(
        self,
        candidate: Image.Image,
        blueprint: QRBlueprint,
        payload: str,
    ) -> _Stage2Control:
        """Build either the exact control or the real Reed-Solomon QArt control."""
        if self.settings.diffqrcoder_stage2_target_mode == "binary_exact":
            return _Stage2Control(
                image=self.control_image(blueprint),
                blueprint=blueprint,
                match_mode="exact",
            )
        target = build_qart_target(
            candidate,
            payload,
            version=self.settings.diffqrcoder_qr_version,
            module_size=self.settings.diffqrcoder_qr_module_size,
            padding_px=self.settings.diffqrcoder_qr_padding_px,
            thresholds=self.settings.diffqrcoder_qart_thresholds,
            executable=self.settings.diffqrcoder_qart_executable,
        )
        self._diagnostics.update(
            {
                "diffqrcoder_qart_threshold": float(target.threshold),
                "diffqrcoder_qart_target_scan_pass_rate": float(target.scan_pass_rate),
                "diffqrcoder_qart_target_original_pass_rate": float(
                    target.original_passed / target.original_total if target.original_total else 0.0
                ),
                "diffqrcoder_qart_reference_cost": float(target.reference_cost),
            }
        )
        return _Stage2Control(
            image=target.image,
            blueprint=target.blueprint,
            match_mode="canonical_url_without_fragment",
        )

    def validation_kwargs(self, variant_name: str) -> dict:
        if (
            variant_name in {"srpg", "srmpgd"}
            and self._stage2_control is not None
            and self._stage2_control.match_mode == "canonical_url_without_fragment"
        ):
            return {
                "matcher": canonical_url_match,
                "match_mode": "canonical_url_without_fragment",
            }
        return {}

    def module_blueprint(
        self,
        variant_name: str,
        fallback: QRBlueprint,
    ) -> QRBlueprint:
        if variant_name in {"srpg", "srmpgd"} and self._stage2_control is not None:
            return self._stage2_control.blueprint
        return fallback

    def measure_module_error(
        self,
        variant_name: str,
        image: Image.Image,
        fallback: QRBlueprint,
    ) -> float:
        blueprint = self.module_blueprint(variant_name, fallback)
        if variant_name not in {"srpg", "srmpgd"}:
            return module_error_rate(image, blueprint)
        return diffqrcoder_module_error_rate(
            image,
            blueprint,
            padding_px=self.settings.diffqrcoder_qr_padding_px,
            module_size=self.settings.diffqrcoder_qr_module_size,
        )

    def _record_divergence_guard(
        self,
        image: Image.Image,
        candidate: Image.Image,
    ) -> None:
        change = image_change_metrics(image, candidate)
        quality = image_quality_metrics(image)
        warnings = {
            "changed_pixels": (
                change["changed_pixel_ratio"]
                > self.settings.diffqrcoder_guard_max_changed_pixel_ratio
            ),
            "mean_absolute_change": (
                change["mean_absolute_change"]
                > self.settings.diffqrcoder_guard_max_mean_absolute_change
            ),
            "clipped_pixels": (
                change["clipped_pixel_ratio_increase"]
                > self.settings.diffqrcoder_guard_max_clipped_pixel_ratio_increase
            ),
            "rgb_clipped_channels": (
                change["rgb_clipped_channel_ratio_increase"]
                > self.settings.diffqrcoder_guard_max_rgb_clipped_channel_ratio_increase
            ),
            "saturation_mean_increase": (
                change["saturation_mean_increase"]
                > self.settings.diffqrcoder_guard_max_saturation_mean_increase
            ),
            "high_saturation_increase": (
                change["high_saturation_ratio_increase"]
                > self.settings.diffqrcoder_guard_max_high_saturation_ratio_increase
            ),
        }
        hard_failures = {
            "mean_absolute_change": (
                change["mean_absolute_change"]
                > self.settings.diffqrcoder_guard_hard_max_mean_absolute_change
            ),
            "clipped_pixels": (
                change["clipped_pixel_ratio_increase"]
                > self.settings.diffqrcoder_guard_hard_max_clipped_pixel_ratio_increase
            ),
            "rgb_clipped_channels": (
                change["rgb_clipped_channel_ratio_increase"]
                > self.settings.diffqrcoder_guard_hard_max_rgb_clipped_channel_ratio_increase
            ),
            "saturation_mean_increase": (
                change["saturation_mean_increase"]
                > self.settings.diffqrcoder_guard_hard_max_saturation_mean_increase
            ),
            "high_saturation_increase": (
                change["high_saturation_ratio_increase"]
                > self.settings.diffqrcoder_guard_hard_max_high_saturation_ratio_increase
            ),
        }
        self._diagnostics.update(
            {
                "diffqrcoder_guard_warning": float(any(warnings.values())),
                "diffqrcoder_guard_diverged": float(any(hard_failures.values())),
                "diffqrcoder_stage2_changed_pixel_ratio": float(change["changed_pixel_ratio"]),
                "diffqrcoder_stage2_mean_absolute_change": float(change["mean_absolute_change"]),
                "diffqrcoder_stage2_clipped_pixel_ratio": float(quality["clipped_pixel_ratio"]),
                "diffqrcoder_stage2_rgb_clipped_channel_ratio": float(
                    quality["rgb_clipped_channel_ratio"]
                ),
                "diffqrcoder_stage2_clipped_pixel_ratio_increase": float(
                    change["clipped_pixel_ratio_increase"]
                ),
                "diffqrcoder_stage2_rgb_clipped_channel_ratio_increase": float(
                    change["rgb_clipped_channel_ratio_increase"]
                ),
                "diffqrcoder_stage2_saturation_mean": float(quality["saturation_mean"]),
                "diffqrcoder_stage2_saturation_p95": float(quality["saturation_p95"]),
                "diffqrcoder_stage2_saturation_mean_increase": float(
                    change["saturation_mean_increase"]
                ),
                "diffqrcoder_stage2_high_saturation_ratio_increase": float(
                    change["high_saturation_ratio_increase"]
                ),
                "diffqrcoder_guard_changed_pixels": float(warnings["changed_pixels"]),
                "diffqrcoder_guard_mean_absolute_change": float(warnings["mean_absolute_change"]),
                "diffqrcoder_guard_clipped_pixels": float(warnings["clipped_pixels"]),
                "diffqrcoder_guard_rgb_clipped_channels": float(warnings["rgb_clipped_channels"]),
                "diffqrcoder_guard_saturation": float(
                    warnings["saturation_mean_increase"] or warnings["high_saturation_increase"]
                ),
                "diffqrcoder_guard_hard_mean_absolute_change": float(
                    hard_failures["mean_absolute_change"]
                ),
                "diffqrcoder_guard_hard_clipped_pixels": float(hard_failures["clipped_pixels"]),
                "diffqrcoder_guard_hard_rgb_clipped_channels": float(
                    hard_failures["rgb_clipped_channels"]
                ),
                "diffqrcoder_guard_hard_saturation": float(
                    hard_failures["saturation_mean_increase"]
                    or hard_failures["high_saturation_increase"]
                ),
            }
        )

    def candidate_guard_ok(self, variant_name: str) -> bool:
        if variant_name not in {"srpg", "srmpgd"}:
            return True
        return not bool(self._diagnostics.get("diffqrcoder_guard_diverged", 0.0))

    def _apply_srmpgd(
        self,
        pipe,
        latent,
        image: Image.Image,
        original_blueprint: QRBlueprint,
        *,
        validation_callback=None,
    ) -> Image.Image:
        import torch

        if not self.settings.srmpgd_enabled:
            return image
        stage2_image_sha256 = image_sha256(image)
        paper_equations = self.settings.srmpgd_protocol == "paper_equations"
        if not paper_equations and not hasattr(pipe, "srpg"):
            from diffqrcoder.srpg import ScanningRobustPerceptualGuidance

            pipe.srpg = (
                ScanningRobustPerceptualGuidance(
                    module_size=self.settings.diffqrcoder_qr_module_size,
                    scanning_robust_guidance_scale=self.settings.srpg_qr_weight,
                    perceptual_guidance_scale=self.settings.srpg_perceptual_weight,
                )
                .to(self.settings.device)
                .to(pipe.unet.dtype)
            )

        def preview_srmpgd(preview_image, step):
            paper_milestones = {
                0,
                1,
                2,
                4,
                8,
                12,
                self.settings.srmpgd_max_iterations,
            }
            if self.settings.srpg_save_step_previews and (
                self.settings.srmpgd_protocol != "paper_equations"
                or step.iteration in paper_milestones
            ):
                self._debug_artifacts[f"srmpgd_iteration_{step.iteration:03d}"] = (
                    preview_image.copy()
                )

        def paper_scanning_loss(decoded, target):
            with torch.autocast("cuda", dtype=pipe.unet.dtype):
                return pipe.srpg.scanning_robust_loss_fn(decoded, target)

        srmpgd_config = SRMPGDConfig(
            protocol=self.settings.srmpgd_protocol,
            max_iterations=self.settings.srmpgd_max_iterations,
            step_size=self.settings.srmpgd_step_size,
            gradient_scale=self.settings.srmpgd_gradient_scale,
            min_gradient_rms=self.settings.srmpgd_min_gradient_rms,
            decode_precision=self.settings.srmpgd_decode_precision,
            lpips_weight=self.settings.srmpgd_lpips_weight,
            lpips_net=self.settings.srmpgd_lpips_net,
            lpips_device=self.settings.srmpgd_lpips_device,
            crop_padding_px=self.settings.srmpgd_crop_padding_px,
            dark_threshold=self.settings.srmpgd_dark_threshold,
            light_threshold=self.settings.srmpgd_light_threshold,
            center_fraction=self.settings.srmpgd_center_fraction,
            max_initial_module_error_rate=(self.settings.srmpgd_max_initial_module_error_rate),
            max_step_rms=self.settings.srmpgd_max_step_rms,
            max_total_delta_rms=self.settings.srmpgd_max_total_delta_rms,
            min_relative_module_improvement=(self.settings.srmpgd_min_relative_module_improvement),
            max_lpips_loss=self.settings.srmpgd_max_lpips_loss,
            max_mean_absolute_change=(self.settings.srmpgd_max_mean_absolute_change),
            max_saturation_mean_increase=(self.settings.srmpgd_max_saturation_mean_increase),
            max_high_saturation_ratio_increase=(
                self.settings.srmpgd_max_high_saturation_ratio_increase
            ),
            max_rgb_clipped_channel_ratio_increase=(
                self.settings.srmpgd_max_rgb_clipped_channel_ratio_increase
            ),
            robust_blur_weight=self.settings.srmpgd_robust_blur_weight,
            robust_blur_kernel=self.settings.srmpgd_robust_blur_kernel,
            robust_downscale_weight=(self.settings.srmpgd_robust_downscale_weight),
            robust_downscale_factor=(self.settings.srmpgd_robust_downscale_factor),
            robust_brightness_weight=(self.settings.srmpgd_robust_brightness_weight),
            robust_brightness_low=self.settings.srmpgd_robust_brightness_low,
            robust_brightness_high=(self.settings.srmpgd_robust_brightness_high),
            robust_contrast_weight=(self.settings.srmpgd_robust_contrast_weight),
            robust_contrast_factor=(self.settings.srmpgd_robust_contrast_factor),
            quiet_zone_mode="none",
            functional_pattern_tone_factor=0.0,
        )
        if paper_equations:
            memory_scope = _offload_unused_pipeline_modules_for_paper_srmpgd(
                pipe,
                lpips_net=self.settings.srmpgd_lpips_net,
            )
        else:
            # Guarded SR-MPGD uses the upstream SRPG network held by ``pipe.srpg``.
            # Keep its existing residency unchanged.
            memory_scope = nullcontext(
                {
                    "offloaded_modules": (),
                    "cuda_allocated_before_bytes": None,
                    "cuda_driver_free_before_bytes": None,
                    "cuda_allocated_after_offload_bytes": None,
                    "cuda_driver_free_after_offload_bytes": None,
                    "cuda_peak_allocated_bytes": None,
                    "cuda_allocated_before_restore_bytes": None,
                    "cuda_allocated_after_lpips_offload_bytes": None,
                    "cuda_allocated_after_restore_bytes": None,
                }
            )

        def record_srmpgd_memory(memory_state):
            offloaded_modules = memory_state["offloaded_modules"]
            allocated_before = memory_state["cuda_allocated_before_bytes"]
            allocated_after_offload = memory_state["cuda_allocated_after_offload_bytes"]
            free_before = memory_state["cuda_driver_free_before_bytes"]
            free_after_offload = memory_state["cuda_driver_free_after_offload_bytes"]
            diagnostics = {
                "diffqrcoder_srmpgd_offloaded_module_count": float(len(offloaded_modules)),
                "diffqrcoder_srmpgd_offloaded_gib": (
                    max(0.0, float(allocated_before - allocated_after_offload) / 2**30)
                    if allocated_before is not None and allocated_after_offload is not None
                    else 0.0
                ),
                "diffqrcoder_srmpgd_cuda_gib_before": (
                    float(allocated_before) / 2**30 if allocated_before is not None else 0.0
                ),
                "diffqrcoder_srmpgd_cuda_gib_after_offload": (
                    float(allocated_after_offload) / 2**30
                    if allocated_after_offload is not None
                    else 0.0
                ),
            }
            for key, diagnostic_name in (
                ("cuda_driver_free_before_bytes", "diffqrcoder_srmpgd_driver_free_gib_before"),
                (
                    "cuda_driver_free_after_offload_bytes",
                    "diffqrcoder_srmpgd_driver_free_gib_after_offload",
                ),
                (
                    "cuda_allocated_before_restore_bytes",
                    "diffqrcoder_srmpgd_cuda_gib_before_restore",
                ),
                (
                    "cuda_peak_allocated_bytes",
                    "diffqrcoder_srmpgd_cuda_peak_gib",
                ),
                (
                    "cuda_allocated_after_lpips_offload_bytes",
                    "diffqrcoder_srmpgd_cuda_gib_after_lpips_offload",
                ),
                (
                    "cuda_allocated_after_restore_bytes",
                    "diffqrcoder_srmpgd_cuda_gib_after_restore",
                ),
            ):
                value = memory_state[key]
                diagnostics[diagnostic_name] = (
                    float(value) / 2**30 if value is not None else 0.0
                )
            diagnostics["diffqrcoder_srmpgd_driver_free_gib_gained"] = (
                max(0.0, float(free_after_offload - free_before) / 2**30)
                if free_before is not None and free_after_offload is not None
                else 0.0
            )
            self._diagnostics.update(diagnostics)

        srmpgd_memory = None
        try:
            with memory_scope as srmpgd_memory:
                srmpgd = run_srmpgd(
                    pipe,
                    latent,
                    original_blueprint,
                    srmpgd_config,
                    initial_image=image,
                    scanning_loss=None if paper_equations else paper_scanning_loss,
                    validation_callback=validation_callback,
                    preview_callback=preview_srmpgd,
                )
        except BaseException as exc:
            if srmpgd_memory is not None:
                record_srmpgd_memory(srmpgd_memory)
                if paper_equations:
                    module_count = len(srmpgd_memory["offloaded_modules"])
                    free_after = srmpgd_memory["cuda_driver_free_after_offload_bytes"]
                    phase = getattr(pipe, "_prooftag_srmpgd_phase", "unknown")
                    note_parts = [
                        "paper SR-MPGD failed after temporarily offloading "
                        f"{module_count} diffusion modules",
                        f"phase={phase}",
                        f"stage2_source_run_id={self._stage2_source_run_id or 'unknown'}",
                        f"stage2_source_method_id={self._stage2_source_method_id or 'unknown'}",
                        f"stage2_latent_sha256={self._stage2_latent_sha256 or 'unknown'}",
                        f"stage2_image_sha256={stage2_image_sha256}",
                    ]
                    if free_after is not None:
                        note_parts.append(
                            f"driver_free_after_offload_gib={float(free_after) / 2**30:.3f}"
                        )
                    note = "; ".join(note_parts)
                    # ``BaseException.add_note`` is absent from ``str(exc)`` and therefore
                    # disappeared from campaign exports. Wrap the original error so the next
                    # technical archive contains the residency proof in its normal error field.
                    if isinstance(exc, Exception):
                        raise RuntimeError(f"{type(exc).__name__}: {exc}; {note}") from exc
                    if hasattr(exc, "add_note"):
                        exc.add_note(note)
            raise
        record_srmpgd_memory(srmpgd_memory)
        if self.settings.srpg_save_step_previews:
            # This is the same Stage-2 latent decoded in the SR-MPGD VAE precision before
            # any update. It separates reconstruction/precision effects from Eq. 14.
            self._debug_artifacts["srmpgd_redecoded_iteration_000"] = (
                srmpgd.initial_redecoded_image.copy()
            )
        initial_redecode_change = image_change_metrics(
            srmpgd.initial_redecoded_image,
            image,
        )
        image = srmpgd.image
        selected_image_sha256 = image_sha256(image)
        iteration_zero_exact = (
            srmpgd.selected_iteration != 0 or selected_image_sha256 == stage2_image_sha256
        )
        if not iteration_zero_exact:
            raise RuntimeError("SR-MPGD iteration zero changed the Stage-2 raster")
        self._srmpgd_stop_reason = srmpgd.stop_reason
        self._srmpgd_selected_iteration = srmpgd.selected_iteration
        attempted_steps = list(srmpgd.steps[1:]) or [srmpgd.steps[0]]
        initial_step = srmpgd.steps[0]
        best_attempted = max(
            attempted_steps,
            key=lambda step: (
                step.strict_all,
                step.pass_rate,
                step.worst_decoder_pass_rate,
                step.worst_scenario_pass_rate,
                -step.actual_module_error_rate,
                -step.scanning_robust_loss,
            ),
        )
        self._debug_metadata["srmpgd_trace"] = {
            "protocol": self.settings.srmpgd_protocol,
            "lpips_device": self.settings.srmpgd_lpips_device,
            "target": "original_qr",
            "selected_iteration": srmpgd.selected_iteration,
            "stop_reason": srmpgd.stop_reason,
            "initial_stage2_image_sha256": stage2_image_sha256,
            "initial_redecoded_image_sha256": image_sha256(srmpgd.initial_redecoded_image),
            "initial_redecode_change": initial_redecode_change,
            "robust_loss_enabled": any(
                value > 0
                for value in (
                    self.settings.srmpgd_robust_blur_weight,
                    self.settings.srmpgd_robust_downscale_weight,
                    self.settings.srmpgd_robust_brightness_weight,
                    self.settings.srmpgd_robust_contrast_weight,
                )
            ),
            "steps": [asdict(step) for step in srmpgd.steps],
        }
        self._diagnostics.update(
            {
                "diffqrcoder_srmpgd_iterations": float(len(srmpgd.steps) - 1),
                "diffqrcoder_srmpgd_paper_equations": float(
                    self.settings.srmpgd_protocol == "paper_equations"
                ),
                "diffqrcoder_srmpgd_gamma": float(self.settings.srmpgd_step_size),
                "diffqrcoder_srmpgd_gradient_scale": float(self.settings.srmpgd_gradient_scale),
                "diffqrcoder_srmpgd_effective_gradient_scale": float(
                    initial_step.gradient_scale or 0.0
                ),
                "diffqrcoder_srmpgd_decode_float32": float(
                    self.settings.srmpgd_decode_precision == "float32"
                ),
                "diffqrcoder_srmpgd_lpips_weight": float(self.settings.srmpgd_lpips_weight),
                "diffqrcoder_srmpgd_lpips_on_cpu": float(
                    self.settings.srmpgd_lpips_device == "cpu"
                ),
                "diffqrcoder_srmpgd_initial_gradient_rms": float(initial_step.gradient_rms or 0.0),
                "diffqrcoder_srmpgd_initial_image_gradient_rms": float(
                    initial_step.image_gradient_rms or 0.0
                ),
                "diffqrcoder_srmpgd_zero_gradient_stop": float(
                    srmpgd.stop_reason.startswith("zero_")
                ),
                "diffqrcoder_srmpgd_selected_iteration": float(srmpgd.selected_iteration),
                "diffqrcoder_srmpgd_iteration_zero_exact": float(iteration_zero_exact),
                "diffqrcoder_srmpgd_initial_mer": float(srmpgd.initial_module_error_rate),
                "diffqrcoder_srmpgd_final_mer": float(srmpgd.final_module_error_rate),
                "diffqrcoder_srmpgd_strict_selected": float(
                    srmpgd.steps[srmpgd.selected_iteration].strict_all
                ),
                "diffqrcoder_srmpgd_stopped_initial_mer": float(
                    srmpgd.stop_reason == "initial_module_error_rate_above_limit"
                ),
                "diffqrcoder_srmpgd_stopped_non_finite": float(
                    srmpgd.stop_reason.startswith("non_finite_")
                ),
                "diffqrcoder_srmpgd_stopped_aesthetic_guard": float(
                    srmpgd.stop_reason.startswith("aesthetic_guard_failed_")
                ),
                "diffqrcoder_srmpgd_selected_aesthetic_guard": float(
                    srmpgd.steps[srmpgd.selected_iteration].aesthetic_guard_passed
                ),
                "diffqrcoder_srmpgd_selected_qr_gain_sufficient": float(
                    srmpgd.steps[srmpgd.selected_iteration].qr_gain_sufficient
                ),
                "diffqrcoder_srmpgd_selected_lpips": float(
                    srmpgd.steps[srmpgd.selected_iteration].lpips_loss
                ),
                "diffqrcoder_srmpgd_selected_mean_absolute_change": float(
                    srmpgd.steps[srmpgd.selected_iteration].mean_absolute_change
                ),
                "diffqrcoder_srmpgd_selected_latent_delta_rms": float(
                    srmpgd.steps[srmpgd.selected_iteration].latent_delta_rms
                ),
                "diffqrcoder_srmpgd_max_applied_step_rms": float(
                    max(
                        (step.applied_step_rms or 0.0 for step in srmpgd.steps),
                        default=0.0,
                    )
                ),
                "diffqrcoder_srmpgd_robust_loss_enabled": float(
                    self._debug_metadata["srmpgd_trace"]["robust_loss_enabled"]
                ),
                "diffqrcoder_srmpgd_attempted_best_iteration": float(best_attempted.iteration),
                "diffqrcoder_srmpgd_attempted_best_pass_rate": float(best_attempted.pass_rate),
                "diffqrcoder_srmpgd_attempted_best_mer": float(
                    best_attempted.actual_module_error_rate
                ),
                "diffqrcoder_srmpgd_attempted_best_srl": float(best_attempted.scanning_robust_loss),
                "diffqrcoder_srmpgd_attempted_best_lpips": float(best_attempted.lpips_loss),
                "diffqrcoder_srmpgd_attempted_best_change": float(
                    best_attempted.mean_absolute_change
                ),
                "diffqrcoder_srmpgd_attempted_best_guard": float(
                    best_attempted.aesthetic_guard_passed
                ),
                "diffqrcoder_srmpgd_attempted_best_qr_gain": float(
                    best_attempted.qr_gain_sufficient
                ),
                "diffqrcoder_srmpgd_attempted_best_eligible": float(
                    best_attempted.eligible_for_selection
                ),
                "diffqrcoder_srmpgd_min_attempted_srl": float(
                    min(step.scanning_robust_loss for step in attempted_steps)
                ),
                "diffqrcoder_srmpgd_min_attempted_mer": float(
                    min(step.actual_module_error_rate for step in attempted_steps)
                ),
                "diffqrcoder_srmpgd_eligible_attempts": float(
                    sum(step.eligible_for_selection for step in attempted_steps)
                ),
            }
        )
        self._debug_artifacts["srmpgd_selected"] = image.copy()
        self._debug_metadata["srmpgd_stage2_image_sha256"] = stage2_image_sha256
        self._debug_metadata["srmpgd_selected_image_sha256"] = selected_image_sha256
        # Debug metadata is not part of the laboratory CSV export.  Keep the
        # same hashes in backend provenance so an offline audit can prove an
        # iteration-zero no-op without depending on locating an aliased parent
        # row in a possibly retried campaign export.
        self._srmpgd_stage2_image_sha256 = stage2_image_sha256
        self._srmpgd_selected_image_sha256 = selected_image_sha256
        return image

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
        self._debug_metadata.clear()
        self._stage2_source_run_id = None
        self._stage2_source_method_id = None
        self._stage2_pairing_status = "generated_source"
        self._srmpgd_stop_reason = None
        self._srmpgd_selected_iteration = None
        self._srmpgd_stage2_image_sha256 = None
        self._srmpgd_selected_image_sha256 = None
        if self._stage2_override is not None:
            cached = self._stage2_override
            self._stage2_override = None
            stage2_control = cached["control"]
            self._stage2_control = stage2_control
            latent = cached["latent"].to(
                device=self.settings.device,
                dtype=pipe.unet.dtype,
            )
            self._stage2_latent_sha256 = _tensor_sha256(latent)
            if self._stage2_latent_sha256 != cached["latent_sha256"]:
                raise RuntimeError(
                    "imported DiffQRCoder Stage 2 latent changed while moving to GPU"
                )
            self._stage2_source_run_id = cached["source_run_id"]
            self._stage2_source_method_id = cached["source_method_id"]
            self._stage2_pairing_status = "exact_reuse"
            image = cached["image"].copy()
            self._diagnostics = dict(cached["diagnostics"])
            self._diagnostics["diffqrcoder_stage2_reused"] = 1.0
            self._diagnostics["diffqrcoder_stage2_pairing_exact"] = 1.0
            self._debug_artifacts["stage2_reference"] = cached["reference"].copy()
            self._debug_artifacts["stage2_control_target"] = stage2_control.image.copy()
            self._debug_artifacts["stage2_before_srmpgd"] = image.copy()
            image = self._apply_srmpgd(
                pipe,
                latent,
                image,
                blueprint,
                validation_callback=validation_callback,
            )
            self._record_divergence_guard(image, candidate)
            del latent
            torch.cuda.empty_cache()
            return image
        stage2_seed = (seed + self.settings.srpg_seed_offset) % (2**32)
        generator = torch.Generator(device=self.settings.device).manual_seed(stage2_seed)
        reference = _pil_to_tensor(
            candidate,
            device=self.settings.device,
            dtype=pipe.unet.dtype,
        )
        stage2_control = self._stage2_target(candidate, blueprint, request.payload)
        self._stage2_control = stage2_control
        stage2_target = stage2_control.image
        stage2_blueprint = stage2_control.blueprint
        self._debug_artifacts["stage2_reference"] = candidate.copy()
        self._debug_artifacts["stage2_control_target"] = stage2_target.copy()
        initial_latent = None
        initialization_diagnostics = {
            "diffqrcoder_stage2_paper_initialization": 0.0,
            "diffqrcoder_stage2_effective_steps": float(self.settings.srpg_steps),
        }
        if self.settings.diffqrcoder_stage2_initialization == "paper_stage1_noise":
            (
                initial_latent,
                initialization_diagnostics,
            ) = self._paper_stage2_initial_latent(
                pipe,
                candidate,
                generator=generator,
            )
            initialization_diagnostics["diffqrcoder_stage2_paper_initialization"] = 1.0
        effective_steps = int(initialization_diagnostics["diffqrcoder_stage2_effective_steps"])
        preview_interval = self.settings.srpg_preview_interval

        def callback(current_pipe, index, timestep, values):
            if self.settings.srpg_save_step_previews and (
                index % preview_interval == 0 or index + 1 == effective_steps
            ):
                self._debug_artifacts[f"stage2_x0_estimate_step_{index + 1:03d}"] = _tensor_to_pil(
                    current_pipe.image_processor.denormalize(values["original_image"].detach())
                )
            return values

        with _install_partial_schedule(
            pipe,
            base_steps=self.settings.srpg_steps,
            effective_steps=effective_steps,
        ):
            output = pipe._run_stage2(
                prompt=request.prompt,
                qrcode=stage2_target,
                qrcode_module_size=self.settings.diffqrcoder_qr_module_size,
                qrcode_padding=self.settings.diffqrcoder_qr_padding_px,
                ref_image=reference,
                negative_prompt=request.negative_prompt or None,
                num_inference_steps=effective_steps,
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
                callback_on_step_end_tensor_inputs=[
                    "latents",
                    "original_image",
                ],
                output_type="latent",
            )
        latent = output.images.detach()
        self._stage2_latent_sha256 = _tensor_sha256(latent)
        image = self._decode_latent(pipe, latent)
        self._debug_artifacts["stage2_before_srmpgd"] = image.copy()
        self._diagnostics.update(
            {
                "diffqrcoder_revision_verified": 1.0,
                "diffqrcoder_stage2_seed": float(stage2_seed),
                "diffqrcoder_stage2_steps": float(effective_steps),
                "diffqrcoder_srg": float(self.settings.srpg_qr_weight),
                "diffqrcoder_pg": float(self.settings.srpg_perceptual_weight),
                "diffqrcoder_stage2_control_target_exact": float(
                    stage2_control.match_mode == "exact"
                ),
                "diffqrcoder_stage2_control_target_qart": float(
                    stage2_control.match_mode == "canonical_url_without_fragment"
                ),
                "diffqrcoder_stage2_control_target_center_error_rate": (
                    _control_target_center_error_rate(
                        stage2_target,
                        stage2_blueprint,
                        padding_px=self.settings.diffqrcoder_qr_padding_px,
                        module_size=self.settings.diffqrcoder_qr_module_size,
                    )
                ),
                "diffqrcoder_srmpgd_iterations": 0.0,
                "diffqrcoder_srmpgd_gamma": 0.0,
                "diffqrcoder_srmpgd_lpips_weight": 0.0,
                **initialization_diagnostics,
            }
        )
        self._diagnostics["diffqrcoder_stage2_reused"] = 0.0
        self._diagnostics["diffqrcoder_stage2_pairing_exact"] = 0.0
        self._last_stage2_state = {
            "latent": latent.detach().cpu().clone(),
            "image": image.copy(),
            "reference": candidate.copy(),
            "control": stage2_control,
            "diagnostics": dict(self._diagnostics),
        }
        image = self._apply_srmpgd(
            pipe,
            latent,
            image,
            blueprint,
            validation_callback=validation_callback,
        )
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
        effective_variant = "srpg"
        if self.settings.srmpgd_enabled and (self._srmpgd_selected_iteration or 0) > 0:
            effective_variant = "srmpgd"
        yield effective_variant, image

    def debug_artifacts(self) -> dict[str, Image.Image]:
        return self._debug_artifacts.copy()

    def debug_metadata(self) -> dict[str, object]:
        return self._debug_metadata.copy()

    def diagnostics(self) -> dict[str, float]:
        return self._diagnostics.copy()

    def provenance(self) -> dict[str, str]:
        values = {
            "base_model_id": self.settings.base_model_id,
            "base_model_revision": self.settings.base_model_revision,
            "base_model_config_id": self.settings.base_model_config_id,
            "base_model_config_revision": self.settings.base_model_config_revision,
            "controlnet_model_id": self.settings.controlnet_model_id,
            "controlnet_model_subfolder": self.settings.controlnet_model_subfolder,
            "controlnet_model_revision": self.settings.controlnet_model_revision,
            "diffqrcoder_revision": self.settings.diffqrcoder_revision,
        }
        if self._stage2_latent_sha256:
            values["stage2_latent_sha256"] = self._stage2_latent_sha256
        if self._stage2_source_run_id:
            values["stage2_source_run_id"] = self._stage2_source_run_id
        if self._stage2_source_method_id:
            values["stage2_source_method_id"] = self._stage2_source_method_id
        if self._stage2_pairing_status:
            values["stage2_pairing_status"] = self._stage2_pairing_status
        if self._srmpgd_stop_reason:
            values["srmpgd_stop_reason"] = self._srmpgd_stop_reason
        if self.settings.srmpgd_enabled:
            values["srmpgd_protocol"] = self.settings.srmpgd_protocol
            values["srmpgd_target"] = "original_qr"
        if self._srmpgd_stage2_image_sha256:
            values["srmpgd_stage2_image_sha256"] = self._srmpgd_stage2_image_sha256
        if self._srmpgd_selected_image_sha256:
            values["srmpgd_selected_image_sha256"] = self._srmpgd_selected_image_sha256
        return values
