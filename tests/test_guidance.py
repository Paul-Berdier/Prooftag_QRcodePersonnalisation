import numpy as np
import pytest

from prooftag_qr.guidance import build_module_layout, qr_core_geometry
from prooftag_qr.qr import generate_qr


def test_module_layout_covers_every_pixel_and_normalizes_each_module():
    blueprint = generate_qr("https://example.prooftag.test/t/srl-layout", "H")
    layout = build_module_layout(blueprint, 127, 131)

    assert layout.module_ids.shape == (127, 131)
    assert layout.module_ids.min() == 0
    assert layout.module_ids.max() == layout.module_count - 1
    totals = np.bincount(
        layout.module_ids.reshape(-1),
        weights=layout.gaussian_weights.reshape(-1),
        minlength=layout.module_count,
    )
    assert np.allclose(totals, 1.0, atol=1e-5)
    center_counts = np.bincount(
        layout.module_ids.reshape(-1),
        weights=layout.center_mask.reshape(-1),
        minlength=layout.module_count,
    )
    assert np.all(center_counts >= 1)


def test_module_layout_rejects_invalid_dimensions_and_center_fraction():
    blueprint = generate_qr("https://example.prooftag.test/t/srl-invalid", "Q")

    with pytest.raises(ValueError, match="height and width"):
        build_module_layout(blueprint, 0, 512)
    with pytest.raises(ValueError, match="center_fraction"):
        build_module_layout(blueprint, 512, 512, center_fraction=0)


def test_srl_is_zero_for_reference_and_differentiable_for_inverted_qr():
    torch = pytest.importorskip("torch")
    from prooftag_qr.guidance import scanning_robust_loss

    blueprint = generate_qr("https://example.prooftag.test/t/srl-reference", "H", size=128)
    array = np.asarray(blueprint.image, dtype=np.float32) / 255.0
    reference = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    reference_loss, reference_diagnostics = scanning_robust_loss(reference, blueprint)
    assert reference_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert reference_diagnostics["active_modules"].item() == 0

    inverted = (1 - reference).clone().requires_grad_(True)
    inverted_loss, inverted_diagnostics = scanning_robust_loss(inverted, blueprint)
    inverted_loss.backward()

    assert inverted_loss.item() > 0
    assert inverted_diagnostics["module_error_rate"].item() == pytest.approx(1.0)
    assert inverted_diagnostics["active_mask"].shape == (1, blueprint.matrix.size)
    assert inverted_diagnostics["active_mask"].all()
    assert inverted.grad is not None
    assert torch.isfinite(inverted.grad).all()
    assert inverted.grad.abs().sum().item() > 0


def test_srl_supports_diffqrcoder_asymmetric_decode_thresholds():
    torch = pytest.importorskip("torch")
    from prooftag_qr.guidance import scanning_robust_loss

    blueprint = generate_qr("https://example.prooftag.test/t/srl-thresholds", "H", size=128)
    ambiguous = torch.full((1, 3, 128, 128), 0.55, requires_grad=True)

    _, symmetric = scanning_robust_loss(ambiguous, blueprint)
    official_loss, official = scanning_robust_loss(
        ambiguous,
        blueprint,
        dark_threshold=0.45,
        light_threshold=0.65,
    )

    assert official["active_modules"].item() == blueprint.matrix.size
    assert official["active_modules"].item() > symmetric["active_modules"].item()
    official_loss.backward()
    assert ambiguous.grad is not None


def test_srl_rejects_inverted_thresholds():
    torch = pytest.importorskip("torch")
    from prooftag_qr.guidance import scanning_robust_loss

    blueprint = generate_qr("https://example.prooftag.test/t/srl-invalid-thresholds", "H")
    image = torch.zeros((1, 3, 512, 512))

    with pytest.raises(ValueError, match="thresholds"):
        scanning_robust_loss(
            image,
            blueprint,
            dark_threshold=0.70,
            light_threshold=0.60,
        )


def test_srl_equation_6_normalizes_by_all_modules_not_only_active_modules():
    torch = pytest.importorskip("torch")
    from prooftag_qr.guidance import scanning_robust_loss

    blueprint = generate_qr("https://example.test/srl-normalization", "M", size=128)
    count = blueprint.matrix.shape[0]
    image = np.asarray(blueprint.image, dtype=np.float32) / 255.0
    module_row = blueprint.border + 10
    module_col = blueprint.border + 10
    y0 = round(module_row * image.shape[0] / count)
    y1 = round((module_row + 1) * image.shape[0] / count)
    x0 = round(module_col * image.shape[1] / count)
    x1 = round((module_col + 1) * image.shape[1] / count)
    image[y0:y1, x0:x1] = 1.0 - image[y0:y1, x0:x1]
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    loss, diagnostics = scanning_robust_loss(
        tensor,
        blueprint,
        functional_weight=1.0,
    )

    assert diagnostics["active_modules"].item() == 1
    assert 0 < loss.item() <= 1 / blueprint.matrix.size


def test_qr_core_geometry_removes_the_quiet_zone_without_resampling_modules():
    blueprint = generate_qr("https://example.test/core-crop", "M", size=512)

    geometry = qr_core_geometry(blueprint, 512, 512)

    assert geometry.blueprint.border == 0
    assert geometry.blueprint.matrix.shape[0] == (
        blueprint.matrix.shape[0] - 2 * blueprint.border
    )
    assert geometry.left == geometry.top
    assert geometry.right == geometry.bottom
    assert geometry.blueprint.image.size == (
        geometry.right - geometry.left,
        geometry.bottom - geometry.top,
    )
