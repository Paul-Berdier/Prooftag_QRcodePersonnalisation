from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .qr import QRBlueprint, functional_pattern_mask


@dataclass(frozen=True, slots=True)
class ModuleLayout:
    """Pixel-to-module maps shared by the NumPy and PyTorch implementations."""

    module_ids: np.ndarray
    center_mask: np.ndarray
    gaussian_weights: np.ndarray
    target_dark: np.ndarray
    functional: np.ndarray
    module_count: int


@dataclass(frozen=True, slots=True)
class TorchModuleLayout:
    module_ids: Any
    center_mask: Any
    center_counts: Any
    gaussian_weights: Any
    target_dark: Any
    functional: Any
    module_count: int


@dataclass(frozen=True, slots=True)
class LatentRefinementConfig:
    iterations: int = 8
    learning_rate: float = 0.20
    qr_weight: float = 1.0
    preservation_weight: float = 0.15
    functional_weight: float = 4.0
    center_fraction: float = 1 / 3
    target_module_error_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class LatentRefinementResult:
    image: Image.Image
    iterations: int
    initial_module_error_rate: float
    final_module_error_rate: float
    final_srl: float
    final_preservation_loss: float
    improved: bool
    converged: bool


def build_module_layout(
    blueprint: QRBlueprint,
    height: int,
    width: int,
    *,
    center_fraction: float = 1 / 3,
) -> ModuleLayout:
    """Build the Gaussian and central-submodule maps described by DiffQRCoder.

    QR modules do not necessarily divide a 512 px image evenly. The boundaries use the
    same rounding convention as the production module repair, avoiding a one-pixel drift
    between the differentiable objective and the final validator.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 < center_fraction <= 1.0:
        raise ValueError("center_fraction must be between 0 (exclusive) and 1")

    rows, cols = blueprint.matrix.shape
    module_ids = np.empty((height, width), dtype=np.int64)
    center_mask = np.zeros((height, width), dtype=np.float32)
    gaussian_weights = np.zeros((height, width), dtype=np.float32)

    for row in range(rows):
        y0 = round(row * height / rows)
        y1 = max(y0 + 1, round((row + 1) * height / rows))
        for col in range(cols):
            x0 = round(col * width / cols)
            x1 = max(x0 + 1, round((col + 1) * width / cols))
            module_id = row * cols + col
            module_ids[y0:y1, x0:x1] = module_id

            module_height = y1 - y0
            module_width = x1 - x0
            center_height = max(1, int(np.ceil(module_height * center_fraction)))
            center_width = max(1, int(np.ceil(module_width * center_fraction)))
            center_y0 = y0 + (module_height - center_height) // 2
            center_x0 = x0 + (module_width - center_width) // 2
            center_mask[
                center_y0 : center_y0 + center_height,
                center_x0 : center_x0 + center_width,
            ] = 1.0

            yy, xx = np.mgrid[:module_height, :module_width]
            sigma = max(1.0, float((min(module_height, module_width) - 1) // 5))
            distance = (
                (yy + 0.5 - module_height / 2) ** 2
                + (xx + 0.5 - module_width / 2) ** 2
            )
            weights = np.exp(-distance / (2 * sigma**2)).astype(np.float32)
            weights /= float(weights.sum())
            gaussian_weights[y0:y1, x0:x1] = weights

    return ModuleLayout(
        module_ids=module_ids,
        center_mask=center_mask,
        gaussian_weights=gaussian_weights,
        target_dark=blueprint.matrix.astype(bool).reshape(-1),
        functional=functional_pattern_mask(blueprint).reshape(-1),
        module_count=rows * cols,
    )


def prepare_torch_layout(
    blueprint: QRBlueprint,
    height: int,
    width: int,
    *,
    device: Any,
    dtype: Any,
    center_fraction: float = 1 / 3,
) -> TorchModuleLayout:
    """Move a module layout to a torch device without importing torch at module import."""
    import torch

    layout = build_module_layout(
        blueprint,
        height,
        width,
        center_fraction=center_fraction,
    )
    module_ids = torch.as_tensor(layout.module_ids.reshape(-1), device=device, dtype=torch.long)
    center_mask = torch.as_tensor(layout.center_mask.reshape(-1), device=device, dtype=dtype)
    center_counts = torch.zeros(layout.module_count, device=device, dtype=dtype)
    center_counts.scatter_add_(0, module_ids, center_mask)
    return TorchModuleLayout(
        module_ids=module_ids,
        center_mask=center_mask,
        center_counts=center_counts.clamp_min(1),
        gaussian_weights=torch.as_tensor(
            layout.gaussian_weights.reshape(-1), device=device, dtype=dtype
        ),
        target_dark=torch.as_tensor(layout.target_dark, device=device, dtype=torch.bool),
        functional=torch.as_tensor(layout.functional, device=device, dtype=torch.bool),
        module_count=layout.module_count,
    )


def scanning_robust_loss(
    images: Any,
    blueprint: QRBlueprint,
    *,
    functional_weight: float = 4.0,
    center_fraction: float = 1 / 3,
    layout: TorchModuleLayout | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Differentiable module-level Scanning Robust Loss (SRL).

    ``images`` must be a BCHW torch tensor in [0, 1]. Correct central submodules are
    detached from the objective, matching DiffQRCoder's module-level early stopping.
    Functional patterns receive a larger weight but remain governed by the same stop rule.
    """
    import torch

    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError("images must be a BCHW tensor with 1 or 3 channels")
    if functional_weight < 1.0:
        raise ValueError("functional_weight must be at least 1")
    if images.shape[1] == 3:
        coefficients = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        grayscale = (images * coefficients).sum(dim=1)
    else:
        grayscale = images[:, 0]

    height, width = grayscale.shape[-2:]
    prepared = layout or prepare_torch_layout(
        blueprint,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
        center_fraction=center_fraction,
    )
    if prepared.module_ids.numel() != height * width:
        raise ValueError("layout dimensions do not match images")

    batch = grayscale.shape[0]
    flat = grayscale.reshape(batch, -1)
    ids = prepared.module_ids.unsqueeze(0).expand(batch, -1)
    target_dark_pixels = prepared.target_dark[prepared.module_ids].unsqueeze(0)
    pixel_error = torch.where(
        target_dark_pixels,
        torch.relu(2 * flat - 1),
        torch.relu(1 - 2 * flat),
    )
    module_errors = torch.zeros(
        (batch, prepared.module_count), device=images.device, dtype=images.dtype
    )
    module_errors.scatter_add_(
        1,
        ids,
        pixel_error * prepared.gaussian_weights.unsqueeze(0),
    )

    center_sums = torch.zeros_like(module_errors)
    center_sums.scatter_add_(1, ids, flat * prepared.center_mask.unsqueeze(0))
    center_means = center_sums / prepared.center_counts.unsqueeze(0)
    predicted_dark = center_means < 0.5
    targets = prepared.target_dark.unsqueeze(0).expand(batch, -1)
    active = predicted_dark.ne(targets).detach()
    module_weights = torch.where(
        prepared.functional,
        images.new_tensor(functional_weight),
        images.new_tensor(1.0),
    ).unsqueeze(0)
    active_weights = active.to(images.dtype) * module_weights
    loss = (module_errors * active_weights).sum() / active_weights.sum().clamp_min(1)

    functional_active = active[:, prepared.functional]
    data_active = active[:, ~prepared.functional]
    diagnostics = {
        "module_error_rate": active.to(images.dtype).mean(),
        "functional_error_rate": (
            functional_active.to(images.dtype).mean()
            if functional_active.numel()
            else images.new_tensor(0.0)
        ),
        "data_error_rate": (
            data_active.to(images.dtype).mean()
            if data_active.numel()
            else images.new_tensor(0.0)
        ),
        "active_modules": active.sum(),
        "mean_module_error": module_errors.mean(),
    }
    return loss, diagnostics


def multiscale_preservation_loss(images: Any, references: Any) -> Any:
    """Cheap differentiable perceptual surrogate used until LPIPS is benchmarked."""
    import torch.nn.functional as functional

    if images.shape != references.shape:
        raise ValueError("images and references must have the same shape")
    loss = functional.l1_loss(images, references)
    for kernel in (2, 4, 8):
        if min(images.shape[-2:]) >= kernel:
            loss = loss + functional.l1_loss(
                functional.avg_pool2d(images, kernel),
                functional.avg_pool2d(references, kernel),
            )
    return loss / 4


def _pil_to_tensor(image: Image.Image, *, device: Any, dtype: Any) -> Any:
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.as_tensor(array, device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(image: Any) -> Image.Image:
    array = (
        image.detach()
        .float()
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="RGB")


def refine_candidate_latent(
    pipeline: Any,
    candidate: Image.Image,
    blueprint: QRBlueprint,
    config: LatentRefinementConfig,
) -> LatentRefinementResult:
    """Refine only the VAE latent while keeping every model parameter frozen.

    The best intermediate latent is retained. If no iteration improves the central-module
    error rate, the original image is returned unchanged; this prevents an experimental
    optimizer from silently degrading a production candidate.
    """
    import torch

    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.qr_weight <= 0:
        raise ValueError("qr_weight must be positive")
    if config.preservation_weight < 0:
        raise ValueError("preservation_weight cannot be negative")

    vae = pipeline.vae
    vae.requires_grad_(False)
    vae.eval()
    device = pipeline._execution_device
    dtype = next(vae.parameters()).dtype
    candidate = candidate.convert("RGB").resize(blueprint.image.size, Image.Resampling.LANCZOS)
    reference = _pil_to_tensor(candidate, device=device, dtype=dtype)
    layout = prepare_torch_layout(
        blueprint,
        reference.shape[-2],
        reference.shape[-1],
        device=device,
        dtype=dtype,
        center_fraction=config.center_fraction,
    )
    with torch.no_grad():
        _, initial_diagnostics = scanning_robust_loss(
            reference,
            blueprint,
            functional_weight=config.functional_weight,
            layout=layout,
        )
        encoded = vae.encode(reference * 2 - 1).latent_dist.mode()
        scaling_factor = vae.config.scaling_factor
        latent = encoded * scaling_factor

    initial_error = float(initial_diagnostics["module_error_rate"].float().item())
    best_error = initial_error
    best_image = reference.detach()
    best_srl = 0.0
    best_preservation = 0.0
    completed_iterations = 0

    with torch.enable_grad():
        # Evaluate the VAE reconstruction, then every updated latent including the final one.
        for iteration in range(config.iterations + 1):
            latent = latent.detach().requires_grad_(True)
            decoded = vae.decode(latent / scaling_factor, return_dict=False)[0]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            srl, diagnostics = scanning_robust_loss(
                decoded,
                blueprint,
                functional_weight=config.functional_weight,
                layout=layout,
            )
            preservation = multiscale_preservation_loss(decoded, reference)
            objective = config.qr_weight * srl + config.preservation_weight * preservation
            error_rate = float(diagnostics["module_error_rate"].float().item())
            if error_rate < best_error:
                best_error = error_rate
                best_image = decoded.detach()
                best_srl = float(srl.detach().float().item())
                best_preservation = float(preservation.detach().float().item())
            if error_rate <= config.target_module_error_rate:
                break
            if iteration == config.iterations:
                break

            gradient = torch.autograd.grad(objective, latent, only_inputs=True)[0]
            normalized_gradient = gradient / gradient.abs().mean().clamp_min(1e-6)
            latent = latent - config.learning_rate * normalized_gradient
            completed_iterations += 1

    improved = best_error < initial_error
    return LatentRefinementResult(
        image=_tensor_to_pil(best_image) if improved else candidate,
        iterations=completed_iterations,
        initial_module_error_rate=initial_error,
        final_module_error_rate=best_error,
        final_srl=best_srl,
        final_preservation_loss=best_preservation,
        improved=improved,
        converged=best_error <= config.target_module_error_rate,
    )
