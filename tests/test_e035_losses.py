from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from PIL import Image

from prooftag_qr.e035_losses import (
    build_upstream_layout,
    combined_gradient_gate,
    prepare_upstream_torch_layout,
    upstream_center_slice,
    upstream_code_scanning_robust_loss,
    upstream_gaussian_kernel,
    upstream_qrcode_tensor,
)
from prooftag_qr.guidance import prepare_torch_layout, scanning_robust_loss
from prooftag_qr.qr import QRBlueprint

torch = pytest.importorskip("torch")


def blueprint(matrix: np.ndarray, module_size: int) -> QRBlueprint:
    return QRBlueprint(
        image=Image.new(
            "RGB",
            (matrix.shape[1] * module_size, matrix.shape[0] * module_size),
            "white",
        ),
        matrix=matrix.astype(bool),
        version=1,
        border=0,
    )


def _official_reference_loss(
    images: torch.Tensor,
    qrcode: torch.Tensor,
    *,
    module_size: int,
) -> torch.Tensor:
    """Independent literal port of DiffQRCoder e24ea73 for CPU parity tests."""

    coefficients = images.new_tensor((0.2999, 0.587, 0.1114)).view(1, 3, 1, 1)
    gray = (images * coefficients).sum(dim=1, keepdim=True)
    error = (
        2 * torch.relu(gray - 0.45) * (1 - qrcode)
        + 2 * torch.relu(0.65 - gray) * qrcode
    )

    one_dimensional = cv2.getGaussianKernel(module_size, 1.5, cv2.CV_32F)
    kernel = one_dimensional @ one_dimensional.T
    kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
    kernel[kernel < 0.1] = 0
    kernel_tensor = images.new_tensor(kernel).view(1, 1, module_size, module_size)
    sampled = torch.nn.functional.conv2d(
        error,
        kernel_tensor,
        stride=module_size,
    )

    center = int(module_size / 2)
    radius = math.ceil(module_size / 6)
    center_kernel = images.new_zeros((1, 1, module_size, module_size))
    center_kernel[
        :,
        :,
        center - radius : center + radius,
        center - radius : center + radius,
    ] = 1
    center_kernel /= center_kernel.sum()
    center_mean = torch.nn.functional.conv2d(
        gray.detach(),
        center_kernel,
        stride=module_size,
    )
    target_center = torch.nn.functional.conv2d(
        (qrcode.detach() > 0.5).to(images.dtype),
        center_kernel,
        stride=module_size,
    )
    active = (
        ((target_center == 0) & (center_mean > 0.45))
        | ((target_center == 1) & (center_mean < 0.65))
    ).to(images.dtype)
    return (sampled * active).mean()


def test_upstream_module_20_geometry_is_exact() -> None:
    center = upstream_center_slice(20)
    assert (center.start, center.stop) == (6, 14)
    layout = build_upstream_layout(
        blueprint(np.zeros((21, 21), dtype=bool), 20),
        420,
        420,
    )
    assert layout.module_height == layout.module_width == 20
    assert (layout.center_y0, layout.center_y1) == (6, 14)
    assert (layout.center_x0, layout.center_x1) == (6, 14)
    assert np.all(layout.center_counts == 64)


def test_upstream_gaussian_matches_public_opencv_recipe() -> None:
    actual = upstream_gaussian_kernel(20, sigma=1.5, cutoff=0.1)
    one_dimensional = cv2.getGaussianKernel(20, 1.5, cv2.CV_32F)
    reference = one_dimensional @ one_dimensional.T
    reference = (reference - reference.min()) / (reference.max() - reference.min())
    reference[reference < 0.1] = 0
    np.testing.assert_allclose(actual, reference.astype(np.float32), rtol=0, atol=1e-7)
    assert actual.max() == pytest.approx(1.0)
    assert actual.min() == 0.0
    assert actual.sum() != pytest.approx(1.0)


def test_upstream_qrcode_tensor_uses_zero_for_black_and_one_for_white() -> None:
    matrix = np.array([[True, False], [False, True]], dtype=bool)
    qr = blueprint(matrix, 3)
    target = upstream_qrcode_tensor(
        qr,
        6,
        6,
        device="cpu",
        dtype=torch.float32,
    )
    assert target.shape == (1, 1, 6, 6)
    assert torch.all(target[0, 0, :3, :3] == 0)
    assert torch.all(target[0, 0, :3, 3:] == 1)
    assert torch.all(target[0, 0, 3:, :3] == 1)
    assert torch.all(target[0, 0, 3:, 3:] == 0)


def test_white_center_point_five_stops_paper_but_remains_active_upstream() -> None:
    matrix = np.zeros((21, 21), dtype=bool)
    qr = blueprint(matrix, 20)
    image = torch.full((1, 1, 420, 420), 0.5, dtype=torch.float32)
    paper_layout = prepare_torch_layout(
        qr,
        420,
        420,
        device=image.device,
        dtype=image.dtype,
        center_fraction=1 / 3,
    )
    upstream_layout = prepare_upstream_torch_layout(
        qr,
        420,
        420,
        device=image.device,
        dtype=image.dtype,
    )
    paper_loss, paper_diagnostics = scanning_robust_loss(
        image,
        qr,
        functional_weight=1.0,
        center_fraction=1 / 3,
        dark_threshold=0.5,
        light_threshold=0.5,
        layout=paper_layout,
    )
    upstream_loss, upstream_diagnostics = upstream_code_scanning_robust_loss(
        image,
        qr,
        layout=upstream_layout,
    )
    assert float(paper_loss) == pytest.approx(0.0, abs=1e-12)
    assert int(paper_diagnostics["active_modules"]) == 0
    assert int(upstream_diagnostics["active_modules"]) == 21 * 21
    assert float(upstream_loss) > 0


def test_upstream_threshold_equalities_are_stopped() -> None:
    black_matrix = np.ones((21, 21), dtype=bool)
    white_matrix = np.zeros((21, 21), dtype=bool)
    for matrix, value, expected_active in (
        (black_matrix, 0.45, 0),
        (black_matrix, 0.4501, 21 * 21),
        (white_matrix, 0.65, 0),
        (white_matrix, 0.6499, 21 * 21),
    ):
        qr = blueprint(matrix, 6)
        image = torch.full((1, 1, 126, 126), value, dtype=torch.float32)
        loss, diagnostics = upstream_code_scanning_robust_loss(image, qr)
        assert int(diagnostics["active_modules"]) == expected_active
        assert math.isfinite(float(loss))


def test_vectorized_upstream_loss_matches_slow_reference() -> None:
    module_size = 6
    matrix = np.indices((21, 21)).sum(axis=0) % 2 == 0
    qr = blueprint(matrix, module_size)
    generator = torch.Generator().manual_seed(35035)
    image = torch.rand((1, 1, 126, 126), generator=generator)
    actual, _ = upstream_code_scanning_robust_loss(image, qr)

    gray = image[0, 0].numpy()
    kernel = upstream_gaussian_kernel(module_size)
    module_errors: list[float] = []
    center = upstream_center_slice(module_size)
    for row in range(21):
        for col in range(21):
            y0 = row * module_size
            x0 = col * module_size
            region = gray[y0 : y0 + module_size, x0 : x0 + module_size]
            center_mean = float(region[center, center].mean())
            target_dark = bool(matrix[row, col])
            active = center_mean > 0.45 if target_dark else center_mean < 0.65
            if target_dark:
                error = 2 * np.maximum(region - 0.45, 0)
            else:
                error = 2 * np.maximum(0.65 - region, 0)
            module_errors.append(float((error * kernel).sum()) if active else 0.0)
    expected = float(np.mean(module_errors))
    assert float(actual) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_upstream_value_and_gradient_match_independent_official_port() -> None:
    module_size = 6
    matrix = np.indices((21, 21)).sum(axis=0) % 3 == 0
    qr = blueprint(matrix, module_size)
    target = upstream_qrcode_tensor(
        qr,
        126,
        126,
        device="cpu",
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(935035)
    local_image = torch.rand(
        (1, 3, 126, 126),
        generator=generator,
        dtype=torch.float32,
    ).requires_grad_(True)
    reference_image = local_image.detach().clone().requires_grad_(True)

    local_loss, _ = upstream_code_scanning_robust_loss(local_image, qr)
    reference_loss = _official_reference_loss(
        reference_image,
        target,
        module_size=module_size,
    )
    local_gradient = torch.autograd.grad(local_loss, local_image)[0]
    reference_gradient = torch.autograd.grad(reference_loss, reference_image)[0]

    assert float(local_loss.detach()) == pytest.approx(
        float(reference_loss.detach()), abs=2e-6, rel=2e-6
    )
    torch.testing.assert_close(
        local_gradient,
        reference_gradient,
        atol=2e-6,
        rtol=2e-6,
    )


def test_upstream_rgb_grayscale_coefficients_match_public_code() -> None:
    matrix = np.zeros((21, 21), dtype=bool)
    qr = blueprint(matrix, 6)
    # The public coefficients are deliberately non-standard and sum to 0.9983.
    rgb = torch.zeros((1, 3, 126, 126), dtype=torch.float32)
    rgb[:, 0] = 1.0
    _, diagnostics = upstream_code_scanning_robust_loss(rgb, qr)
    center_mean = float(diagnostics["center_means"][0, 0])
    assert center_mean == pytest.approx(0.2999, abs=1e-7)


def test_corrected_gradient_gate_accepts_zero_srl_with_lpips_update() -> None:
    gate = combined_gradient_gate(
        selected_srl=0.0,
        selected_srl_image_gradient_rms=0.0,
        objective_image_gradient_rms=2e-7,
        latent_gradient_rms=3e-8,
        applied_step_rms=3e-5,
        gradient_tolerance=1e-12,
        loss_zero_tolerance=1e-12,
    )
    assert gate["passed"] is True
    assert gate["srl_gradient_required"] is False


def test_corrected_gradient_gate_requires_srl_gradient_while_loss_is_positive() -> None:
    gate = combined_gradient_gate(
        selected_srl=1e-3,
        selected_srl_image_gradient_rms=0.0,
        objective_image_gradient_rms=2e-7,
        latent_gradient_rms=3e-8,
        applied_step_rms=3e-5,
    )
    assert gate["passed"] is False
    assert gate["srl_gradient_required"] is True
    assert gate["srl_gradient_passed"] is False
