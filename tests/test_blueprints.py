import numpy as np
from PIL import Image

from prooftag_qr.blueprints import (
    align_qart_output,
    build_adaptive_blueprint,
    canonical_url_match,
    exact_mask_candidates,
    grid_visibility_score,
)


def test_canonical_url_match_only_ignores_a_fragment():
    expected = "https://ptag.io/t/abc"
    assert canonical_url_match(expected, expected)
    assert canonical_url_match(f"{expected}#12345", expected)
    assert not canonical_url_match(f"{expected}?changed=1#12345", expected)
    assert not canonical_url_match("", expected)


def test_exact_mask_search_returns_eight_aligned_payload_candidates():
    reference = Image.new("RGB", (744, 744), (128, 128, 128))
    candidates = exact_mask_candidates(
        "https://ptag.io/t/a",
        reference,
        version=3,
        error_correction="M",
        module_size=20,
        canvas_size=744,
    )
    assert len(candidates) == 8
    assert {item.aligned.mask_pattern for item in candidates} == set(range(8))
    assert all(item.aligned.image.size == (744, 744) for item in candidates)
    assert candidates == sorted(
        candidates, key=lambda item: (item.reference_cost, item.grid_visibility)
    )


def test_adaptive_blueprint_keeps_quiet_zone_and_functional_modules_binary():
    reference = Image.new("RGB", (744, 744), (160, 120, 80))
    aligned = exact_mask_candidates(
        "https://ptag.io/t/b",
        reference,
        version=3,
        error_correction="M",
        module_size=20,
        canvas_size=744,
    )[0].aligned
    result = build_adaptive_blueprint(reference, aligned)
    array = np.asarray(result.image)
    assert np.all(array[: aligned.padding_px] == 255)
    assert np.all(result.center_fractions[result.functional_modules] == 1.0)
    assert np.all(result.center_fractions[~result.functional_modules] <= 0.92)
    assert grid_visibility_score(result.image, aligned) >= 0


def test_qart_alignment_crops_known_ten_module_border_without_resampling():
    version = 3
    module_size = 4
    core_modules = 17 + 4 * version
    raw_modules = core_modules + 20
    raw = np.full((raw_modules * module_size, raw_modules * module_size, 3), 255, np.uint8)
    start = 10 * module_size
    raw[start : start + module_size, start : start + module_size] = 0
    aligned = align_qart_output(
        Image.fromarray(raw),
        payload="https://ptag.io/t/c",
        version=version,
        module_size=module_size,
        canvas_size=224,
    )
    assert aligned.core_matrix.shape == (29, 29)
    assert aligned.core_matrix[0, 0] == 1
    assert aligned.image.size == (224, 224)
