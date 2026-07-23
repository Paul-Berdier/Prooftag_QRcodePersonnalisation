import numpy as np
import pytest
from PIL import Image

from prooftag_qr.geometry import (
    aligned_module_diagnostics,
    aligned_module_error_rate,
    generate_aligned_qr,
)


@pytest.mark.parametrize(
    ("canvas", "module_size", "padding"),
    [(744, 20, 82), (768, 20, 94), (744, 16, 140), (768, 16, 152)],
)
def test_version_three_core_is_exactly_aligned(canvas, module_size, padding):
    aligned = generate_aligned_qr(
        "https://ptag.io/t/e013",
        version=3,
        error_correction="H",
        mask_pattern=4,
        module_size=module_size,
        canvas_size=canvas,
    )

    assert aligned.core_matrix.shape == (29, 29)
    assert aligned.core_size == 29 * module_size
    assert aligned.padding_px == padding
    assert aligned.image.size == (canvas, canvas)
    assert aligned_module_error_rate(aligned.image, aligned) == 0
    assert aligned_module_diagnostics(aligned.image, aligned)["threshold_safe_rate"] == 1


def test_alignment_rejects_a_sub_four_module_quiet_zone():
    with pytest.raises(ValueError, match="quiet zone"):
        generate_aligned_qr(
            "https://ptag.io/t/e013",
            version=3,
            error_correction="M",
            mask_pattern=4,
            module_size=20,
            canvas_size=736 - 16,
        )


def test_aligned_error_rate_reads_only_the_core():
    aligned = generate_aligned_qr(
        "https://ptag.io/t/e013",
        version=3,
        error_correction="M",
        mask_pattern=4,
        module_size=20,
        canvas_size=744,
    )
    changed = np.asarray(aligned.image).copy()
    changed[: aligned.padding_px, :] = 0
    changed[:, : aligned.padding_px] = 0

    assert aligned_module_error_rate(Image.fromarray(changed), aligned) == 0
