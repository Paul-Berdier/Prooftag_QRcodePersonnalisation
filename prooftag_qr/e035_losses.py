"""E035 loss profiles and diagnostics for the SR-MPGD fidelity gate.

This module is intentionally additive. It leaves the E033/E034 implementation and
published artefacts untouched while exposing two losses evaluated on exactly the same
border-free QR core:

* ``paper_v3``: the equation-level proxy already used by E034;
* ``upstream_code_e24ea73``: the public DiffQRCoder implementation at the pinned
  revision e24ea73ee2e13c7e6e87cb422e8b11784e70ae00.

The upstream profile reproduces the public code's 8x8 centre window for a 20 px module,
its asymmetric 0.45/0.65 stopping margins, and its OpenCV Gaussian mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

LossProfile = Literal["paper_v3", "upstream_code_e24ea73"]
UPSTREAM_REVISION = "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"


@dataclass(frozen=True, slots=True)
class UpstreamNumpyLayout:
    """Pixel/module maps matching the pinned public DiffQRCoder loss."""

    module_ids: np.ndarray
    center_mask: np.ndarray
    center_counts: np.ndarray
    gaussian_weights: np.ndarray
    target_dark: np.ndarray
    functional: np.ndarray
    module_count: int
    rows: int
    cols: int
    module_height: int
    module_width: int
    center_y0: int
    center_y1: int
    center_x0: int
    center_x1: int


@dataclass(frozen=True, slots=True)
class UpstreamTorchLayout:
    module_ids: Any
    center_mask: Any
    center_counts: Any
    gaussian_weights: Any
    target_dark: Any
    functional: Any
    module_count: int
    rows: int
    cols: int
    module_height: int
    module_width: int
    center_y0: int
    center_y1: int
    center_x0: int
    center_x1: int


def upstream_center_slice(module_size: int) -> slice:
    """Return RegionMeanFilter's exact centre slice.

    Public DiffQRCoder uses ``center = int(size / 2)`` and
    ``radius = ceil(size / 6)`` followed by ``center-radius:center+radius``.
    For the E035 module size 20 this is ``6:14`` (8 pixels).
    """

    if module_size < 1:
        raise ValueError("module_size must be positive")
    center = int(module_size / 2)
    radius = int(math.ceil(module_size / 6))
    start = center - radius
    stop = center + radius
    if start < 0 or stop > module_size or start >= stop:
        raise ValueError("invalid upstream centre geometry")
    return slice(start, stop)


def upstream_gaussian_kernel(
    module_size: int,
    *,
    sigma: float = 1.5,
    cutoff: float = 0.1,
) -> np.ndarray:
    """Build the exact public Gaussian mask used by DiffQRCoder.

    The public implementation forms an OpenCV 1-D Gaussian kernel, takes its outer
    product, min-max normalises the 2-D mask to [0, 1], then zeroes values below 0.1.
    It does *not* renormalise the remaining values to sum to one.
    """

    if module_size < 1:
        raise ValueError("module_size must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= cutoff < 1:
        raise ValueError("cutoff must be in [0, 1)")
    one_dimensional = cv2.getGaussianKernel(module_size, sigma, cv2.CV_32F)
    kernel = one_dimensional @ one_dimensional.T
    minimum = float(kernel.min())
    maximum = float(kernel.max())
    if maximum <= minimum:
        normalised = np.ones_like(kernel, dtype=np.float32)
    else:
        normalised = ((kernel - minimum) / (maximum - minimum)).astype(np.float32)
    normalised[normalised < cutoff] = 0.0
    return normalised


def build_upstream_layout(
    blueprint: Any,
    height: int,
    width: int,
    *,
    sigma: float = 1.5,
    cutoff: float = 0.1,
) -> UpstreamNumpyLayout:
    """Build an exact constant-module layout for the border-free E035 QR core."""

    from .qr import functional_pattern_mask

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    matrix = np.asarray(blueprint.matrix, dtype=bool)
    rows, cols = matrix.shape
    if rows <= 0 or cols <= 0:
        raise ValueError("blueprint matrix cannot be empty")
    if height % rows or width % cols:
        raise ValueError(
            "upstream loss requires an integer constant module geometry; "
            f"got canvas {(width, height)} for matrix {(cols, rows)}"
        )
    module_height = height // rows
    module_width = width // cols
    if module_height != module_width:
        raise ValueError("upstream public loss requires square modules")

    module_ids_grid = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    module_ids = np.repeat(
        np.repeat(module_ids_grid, module_height, axis=0), module_width, axis=1
    )

    center_y = upstream_center_slice(module_height)
    center_x = upstream_center_slice(module_width)
    center_tile = np.zeros((module_height, module_width), dtype=np.float32)
    center_tile[center_y, center_x] = 1.0
    center_mask = np.tile(center_tile, (rows, cols))
    center_counts = np.full(
        rows * cols,
        float((center_y.stop - center_y.start) * (center_x.stop - center_x.start)),
        dtype=np.float32,
    )

    gaussian_tile = upstream_gaussian_kernel(
        module_height,
        sigma=sigma,
        cutoff=cutoff,
    )
    gaussian_weights = np.tile(gaussian_tile, (rows, cols)).astype(np.float32)
    functional = np.asarray(functional_pattern_mask(blueprint), dtype=bool).reshape(-1)

    return UpstreamNumpyLayout(
        module_ids=module_ids,
        center_mask=center_mask,
        center_counts=center_counts,
        gaussian_weights=gaussian_weights,
        target_dark=matrix.reshape(-1),
        functional=functional,
        module_count=rows * cols,
        rows=rows,
        cols=cols,
        module_height=module_height,
        module_width=module_width,
        center_y0=center_y.start,
        center_y1=center_y.stop,
        center_x0=center_x.start,
        center_x1=center_x.stop,
    )


def prepare_upstream_torch_layout(
    blueprint: Any,
    height: int,
    width: int,
    *,
    device: Any,
    dtype: Any,
) -> UpstreamTorchLayout:
    """Move the exact upstream layout to a torch device."""

    import torch

    layout = build_upstream_layout(blueprint, height, width)
    return UpstreamTorchLayout(
        module_ids=torch.as_tensor(
            layout.module_ids.reshape(-1), device=device, dtype=torch.long
        ),
        center_mask=torch.as_tensor(
            layout.center_mask.reshape(-1), device=device, dtype=dtype
        ),
        center_counts=torch.as_tensor(layout.center_counts, device=device, dtype=dtype),
        gaussian_weights=torch.as_tensor(
            layout.gaussian_weights.reshape(-1), device=device, dtype=dtype
        ),
        target_dark=torch.as_tensor(layout.target_dark, device=device, dtype=torch.bool),
        functional=torch.as_tensor(layout.functional, device=device, dtype=torch.bool),
        module_count=layout.module_count,
        rows=layout.rows,
        cols=layout.cols,
        module_height=layout.module_height,
        module_width=layout.module_width,
        center_y0=layout.center_y0,
        center_y1=layout.center_y1,
        center_x0=layout.center_x0,
        center_x1=layout.center_x1,
    )


def _validate_bchw(images: Any) -> None:
    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError("images must be a BCHW tensor with one or three channels")


def _paper_luminance(images: Any) -> Any:
    """Equation-level luminance used by the local E034 paper proxy."""

    _validate_bchw(images)
    if images.shape[1] == 1:
        return images[:, 0]
    coefficients = images.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (images * coefficients).sum(dim=1)


def _upstream_luminance(images: Any) -> Any:
    """Pinned upstream coefficients from ``diffqrcoder.image_processor``."""

    _validate_bchw(images)
    if images.shape[1] == 1:
        # Convenience for CPU unit tests. The public class itself asserts RGB input;
        # the GPU E035 runner cross-checks the RGB local path against that class.
        return images[:, 0]
    coefficients = images.new_tensor((0.2999, 0.587, 0.1114)).view(1, 3, 1, 1)
    return (images * coefficients).sum(dim=1)


def upstream_qrcode_tensor(
    blueprint: Any,
    height: int,
    width: int,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    """Build the exact 0=black/1=white tensor consumed by upstream SRL.

    ``qrcode`` stores dark modules as true/one, while DiffQRCoder's loss treats
    zero as a black target and one as a white target. The conversion is therefore
    an inversion followed by nearest-neighbour module expansion.
    """

    import torch

    matrix = np.asarray(blueprint.matrix, dtype=bool)
    rows, cols = matrix.shape
    if height <= 0 or width <= 0 or height % rows or width % cols:
        raise ValueError(
            "upstream qrcode tensor requires a positive integer module geometry"
        )
    module_height = height // rows
    module_width = width // cols
    if module_height != module_width:
        raise ValueError("upstream qrcode tensor requires square modules")
    target_light = np.repeat(
        np.repeat((~matrix).astype(np.float32), module_height, axis=0),
        module_width,
        axis=1,
    )
    return torch.as_tensor(
        target_light,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).unsqueeze(0)


def upstream_code_scanning_robust_loss(
    images: Any,
    blueprint: Any,
    *,
    layout: UpstreamTorchLayout | None = None,
    dark_threshold: float = 0.45,
    light_threshold: float = 0.65,
) -> tuple[Any, dict[str, Any]]:
    """Differentiate the pinned public DiffQRCoder SRL implementation.

    Correct modules are stopped by the centre margin:

    * black target: stop when centre mean <= 0.45;
    * white target: stop when centre mean >= 0.65.

    Remaining pixel errors are convolved module-by-module with the public Gaussian
    mask, then the resulting per-module sums are multiplied by the detached stopping
    mask and averaged over batch × modules. This exactly matches the pinned public
    ``Conv2d(kernel_size=module_size, stride=module_size)`` implementation. No
    functional-pattern weighting is applied.
    """

    import torch

    if not 0 < dark_threshold < light_threshold < 1:
        raise ValueError("thresholds must satisfy 0 < dark < light < 1")
    grayscale = _upstream_luminance(images)
    height, width = grayscale.shape[-2:]
    prepared = layout or prepare_upstream_torch_layout(
        blueprint,
        height,
        width,
        device=images.device,
        dtype=images.dtype,
    )
    if prepared.module_ids.numel() != height * width:
        raise ValueError("layout dimensions do not match images")

    batch = grayscale.shape[0]
    flat = grayscale.reshape(batch, -1)
    ids = prepared.module_ids.unsqueeze(0).expand(batch, -1)
    target_dark_pixels = prepared.target_dark[prepared.module_ids].unsqueeze(0)
    sample_error = torch.where(
        target_dark_pixels,
        2 * torch.relu(flat - dark_threshold),
        2 * torch.relu(light_threshold - flat),
    )

    center_sums = torch.zeros(
        (batch, prepared.module_count), device=images.device, dtype=images.dtype
    )
    center_sums.scatter_add_(
        1,
        ids,
        flat * prepared.center_mask.unsqueeze(0),
    )
    center_means = center_sums / prepared.center_counts.unsqueeze(0).clamp_min(1)
    targets = prepared.target_dark.unsqueeze(0).expand(batch, -1)
    active = torch.where(
        targets,
        center_means > dark_threshold,
        center_means < light_threshold,
    ).detach()
    # Public GaussianFilter is a Conv2d with kernel_size=stride=module_size.
    # Its output is therefore one *weighted sum* per module, not a per-pixel mean.
    module_weighted_errors = torch.zeros_like(center_means)
    module_weighted_errors.scatter_add_(
        1,
        ids,
        sample_error * prepared.gaussian_weights.unsqueeze(0),
    )
    masked_module_errors = module_weighted_errors * active.to(images.dtype)
    loss = masked_module_errors.mean()

    functional_active = active[:, prepared.functional]
    data_active = active[:, ~prepared.functional]
    diagnostics = {
        "profile": "upstream_code_e24ea73",
        "upstream_revision": UPSTREAM_REVISION,
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
        "active_mask": active,
        "center_means": center_means,
        "module_weighted_errors": module_weighted_errors,
        "masked_module_errors": masked_module_errors,
        "loss_reduction": "mean_over_batch_and_modules",
        "dark_threshold": images.new_tensor(dark_threshold),
        "light_threshold": images.new_tensor(light_threshold),
        "center_height": images.new_tensor(prepared.center_y1 - prepared.center_y0),
        "center_width": images.new_tensor(prepared.center_x1 - prepared.center_x0),
    }
    return loss, diagnostics


def paper_v3_scanning_robust_loss(
    images: Any,
    blueprint: Any,
    *,
    layout: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Call the unchanged equation-level E034 SRL control."""

    from .guidance import scanning_robust_loss

    loss, diagnostics = scanning_robust_loss(
        images,
        blueprint,
        functional_weight=1.0,
        center_fraction=1 / 3,
        dark_threshold=0.5,
        light_threshold=0.5,
        layout=layout,
    )
    return loss, {"profile": "paper_v3", **diagnostics}


def evaluate_loss_profiles(
    images: Any,
    blueprint: Any,
    *,
    paper_layout: Any | None = None,
    upstream_layout: UpstreamTorchLayout | None = None,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Evaluate both E035 losses without changing either branch's update rule."""

    return {
        "paper_v3": paper_v3_scanning_robust_loss(
            images,
            blueprint,
            layout=paper_layout,
        ),
        "upstream_code_e24ea73": upstream_code_scanning_robust_loss(
            images,
            blueprint,
            layout=upstream_layout,
        ),
    }


def _module_means(flat: Any, ids: Any, module_count: int, weights: Any) -> Any:
    import torch

    batch = flat.shape[0]
    expanded_ids = ids.unsqueeze(0).expand(batch, -1)
    sums = torch.zeros(
        (batch, module_count), device=flat.device, dtype=flat.dtype
    )
    counts = torch.zeros_like(sums)
    sums.scatter_add_(1, expanded_ids, flat * weights.unsqueeze(0))
    counts.scatter_add_(
        1,
        expanded_ids,
        weights.unsqueeze(0).expand(batch, -1),
    )
    return sums / counts.clamp_min(1)


def _quantile(values: Any, probability: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.float().quantile(probability).detach().cpu())


def module_diagnostics(
    images: Any,
    blueprint: Any,
    *,
    paper_layout: Any,
    upstream_layout: UpstreamTorchLayout,
) -> dict[str, Any]:
    """Return cross-profile counts and luminance margins for a BCHW unit tensor."""

    import torch

    paper_grayscale = _paper_luminance(images)
    upstream_grayscale = _upstream_luminance(images)
    batch = paper_grayscale.shape[0]
    if batch != 1:
        raise ValueError("E035 module diagnostics require batch size one")
    paper_flat = paper_grayscale.reshape(batch, -1)
    upstream_flat = upstream_grayscale.reshape(batch, -1)
    ids = upstream_layout.module_ids
    ones = torch.ones_like(ids, dtype=images.dtype)
    full_means = _module_means(
        paper_flat,
        ids,
        upstream_layout.module_count,
        ones,
    )
    paper_center_means = _module_means(
        paper_flat,
        paper_layout.module_ids,
        paper_layout.module_count,
        paper_layout.center_mask,
    )
    upstream_center_means = _module_means(
        upstream_flat,
        ids,
        upstream_layout.module_count,
        upstream_layout.center_mask,
    )
    targets = upstream_layout.target_dark.unsqueeze(0)
    full_errors = (full_means < 0.5) != targets
    paper_errors = (paper_center_means < 0.5) != targets
    upstream_active = torch.where(
        targets,
        upstream_center_means > 0.45,
        upstream_center_means < 0.65,
    )
    functional = upstream_layout.functional.unsqueeze(0)
    data = ~functional
    dark = targets
    light = ~targets

    def count(mask: Any) -> int:
        return int(mask.sum().detach().cpu())

    result = {
        "module_count": upstream_layout.module_count,
        "paper_center_error_count": count(paper_errors),
        "paper_center_error_rate": float(paper_errors.float().mean().cpu()),
        "upstream_margin_active_count": count(upstream_active),
        "upstream_margin_active_rate": float(upstream_active.float().mean().cpu()),
        "full_module_error_count": count(full_errors),
        "full_module_error_rate": float(full_errors.float().mean().cpu()),
        "functional_paper_center_error_count": count(paper_errors & functional),
        "data_paper_center_error_count": count(paper_errors & data),
        "functional_upstream_active_count": count(upstream_active & functional),
        "data_upstream_active_count": count(upstream_active & data),
        "functional_full_module_error_count": count(full_errors & functional),
        "data_full_module_error_count": count(full_errors & data),
        "black_upstream_center_q05": _quantile(upstream_center_means[dark], 0.05),
        "black_upstream_center_q50": _quantile(upstream_center_means[dark], 0.50),
        "black_upstream_center_q95": _quantile(upstream_center_means[dark], 0.95),
        "white_upstream_center_q05": _quantile(upstream_center_means[light], 0.05),
        "white_upstream_center_q50": _quantile(upstream_center_means[light], 0.50),
        "white_upstream_center_q95": _quantile(upstream_center_means[light], 0.95),
        "full_module_luminance_q05": _quantile(full_means.reshape(-1), 0.05),
        "full_module_luminance_q50": _quantile(full_means.reshape(-1), 0.50),
        "full_module_luminance_q95": _quantile(full_means.reshape(-1), 0.95),
    }
    return result


def module_error_maps(
    images: Any,
    blueprint: Any,
    *,
    paper_layout: Any,
    upstream_layout: UpstreamTorchLayout,
) -> dict[str, np.ndarray]:
    """Return 2-D boolean maps for the three E035 error definitions."""

    import torch

    paper_grayscale = _paper_luminance(images)
    upstream_grayscale = _upstream_luminance(images)
    if paper_grayscale.shape[0] != 1:
        raise ValueError("E035 maps require batch size one")
    paper_flat = paper_grayscale.reshape(1, -1)
    upstream_flat = upstream_grayscale.reshape(1, -1)
    ids = upstream_layout.module_ids
    ones = torch.ones_like(ids, dtype=images.dtype)
    full_means = _module_means(
        paper_flat,
        ids,
        upstream_layout.module_count,
        ones,
    )
    paper_center_means = _module_means(
        paper_flat,
        paper_layout.module_ids,
        paper_layout.module_count,
        paper_layout.center_mask,
    )
    upstream_center_means = _module_means(
        upstream_flat,
        ids,
        upstream_layout.module_count,
        upstream_layout.center_mask,
    )
    targets = upstream_layout.target_dark.unsqueeze(0)
    maps = {
        "paper_center_error": (paper_center_means < 0.5) != targets,
        "upstream_margin_active": torch.where(
            targets,
            upstream_center_means > 0.45,
            upstream_center_means < 0.65,
        ),
        "full_module_error": (full_means < 0.5) != targets,
    }
    return {
        name: value.reshape(upstream_layout.rows, upstream_layout.cols)
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
        for name, value in maps.items()
    }


def combined_gradient_gate(
    *,
    selected_srl: float,
    selected_srl_image_gradient_rms: float | None,
    objective_image_gradient_rms: float | None,
    latent_gradient_rms: float | None,
    applied_step_rms: float | None,
    gradient_tolerance: float = 1e-12,
    loss_zero_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the corrected E035 update gate.

    A zero SRL legitimately has a zero SRL-only gradient. In that case LPIPS may still
    make the combined objective, latent gradient and applied step non-zero. The SRL-only
    gradient is therefore required only while the selected SRL is above the loss-zero
    tolerance.
    """

    if gradient_tolerance < 0 or loss_zero_tolerance < 0:
        raise ValueError("gate tolerances cannot be negative")

    def finite_above(value: float | None, threshold: float) -> bool:
        return value is not None and math.isfinite(value) and value > threshold

    srl_gradient_required = selected_srl > loss_zero_tolerance
    srl_gradient_passed = (
        not srl_gradient_required
        or finite_above(selected_srl_image_gradient_rms, gradient_tolerance)
    )
    objective_gradient_passed = finite_above(
        objective_image_gradient_rms,
        gradient_tolerance,
    )
    latent_gradient_passed = finite_above(latent_gradient_rms, gradient_tolerance)
    applied_step_passed = finite_above(applied_step_rms, gradient_tolerance)
    passed = all(
        (
            srl_gradient_passed,
            objective_gradient_passed,
            latent_gradient_passed,
            applied_step_passed,
        )
    )
    return {
        "passed": passed,
        "srl_gradient_required": srl_gradient_required,
        "srl_gradient_passed": srl_gradient_passed,
        "objective_gradient_passed": objective_gradient_passed,
        "latent_gradient_passed": latent_gradient_passed,
        "applied_step_passed": applied_step_passed,
        "gradient_tolerance": gradient_tolerance,
        "loss_zero_tolerance": loss_zero_tolerance,
    }
