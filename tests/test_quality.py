import numpy as np
from PIL import Image

from prooftag_qr.quality import composite_guided_regions, image_change_metrics


def test_image_change_metrics_compare_a_variant_with_its_raw_image():
    raw = Image.new("RGB", (10, 10), (100, 100, 100))
    variant_array = np.full((10, 10, 3), 100, dtype=np.uint8)
    variant_array[:5] = 120
    variant = Image.fromarray(variant_array)

    metrics = image_change_metrics(variant, raw)

    assert metrics["changed_pixel_ratio"] == 0.5
    assert metrics["mean_absolute_change"] == 10 / 255
    assert metrics["clipped_pixel_ratio_increase"] == 0
    assert metrics["rgb_clipped_channel_ratio_increase"] == 0
    assert metrics["saturation_mean_increase"] == 0
    assert metrics["high_saturation_ratio_increase"] == 0


def test_image_change_metrics_measure_new_clipping_not_baseline_contrast():
    raw_array = np.full((10, 10, 3), 100, dtype=np.uint8)
    raw_array[:2] = 0
    variant_array = raw_array.copy()
    variant_array[2:5] = 255

    metrics = image_change_metrics(
        Image.fromarray(variant_array),
        Image.fromarray(raw_array),
    )

    assert metrics["clipped_pixel_ratio_increase"] == 0.3
    assert metrics["rgb_clipped_channel_ratio_increase"] == 0.3


def test_guided_composite_changes_only_the_control_neighborhood():
    reference = Image.new("RGB", (64, 64), "black")
    generated = Image.new("RGB", (64, 64), "white")
    control_array = np.asarray(reference).copy()
    control_array[28:36, 28:36] = 255
    control = Image.fromarray(control_array)

    composite, mask = composite_guided_regions(
        reference,
        generated,
        control,
        dilation_px=2,
        feather_px=2,
    )

    composite_array = np.asarray(composite)
    mask_array = np.asarray(mask)
    assert composite_array[0, 0].max() == 0
    assert composite_array[32, 32].min() == 255
    assert 0 < np.count_nonzero(mask_array) < mask_array.size


def test_guided_composite_is_unchanged_when_control_matches_reference():
    reference = Image.new("RGB", (32, 32), "navy")
    generated = Image.new("RGB", (32, 32), "white")

    composite, mask = composite_guided_regions(reference, generated, reference)

    assert np.array_equal(np.asarray(composite), np.asarray(reference))
    assert np.count_nonzero(np.asarray(mask)) == 0
