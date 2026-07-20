import numpy as np
from PIL import Image

from prooftag_qr.quality import image_change_metrics


def test_image_change_metrics_compare_a_variant_with_its_raw_image():
    raw = Image.new("RGB", (10, 10), (100, 100, 100))
    variant_array = np.full((10, 10, 3), 100, dtype=np.uint8)
    variant_array[:5] = 120
    variant = Image.fromarray(variant_array)

    metrics = image_change_metrics(variant, raw)

    assert metrics["changed_pixel_ratio"] == 0.5
    assert metrics["mean_absolute_change"] == 10 / 255
