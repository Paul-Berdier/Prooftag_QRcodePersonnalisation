import numpy as np
from PIL import Image

from prooftag_qr.e046_quiet_zone import (
    compare_core_bytes,
    compose_scene_preserving_quiet_zone,
    core_bounds,
    quiet_zone_metrics,
)


def _gradient_image() -> Image.Image:
    yy, xx = np.mgrid[0:736, 0:736]
    array = np.stack(
        [
            (xx / 735.0) * 180 + 20,
            (yy / 735.0) * 150 + 30,
            ((xx + yy) / 1470.0) * 160 + 40,
        ],
        axis=2,
    )
    return Image.fromarray(np.rint(array).astype(np.uint8), mode="RGB")


def test_scene_preserving_quiet_zone_keeps_canvas_and_core_bytes():
    source = _gradient_image()
    output, evidence = compose_scene_preserving_quiet_zone(source)
    assert output.size == source.size == (736, 736)
    assert compare_core_bytes(source, output)
    assert evidence["core_byte_identical"] is True
    assert evidence["no_crop"] is True
    assert evidence["uniform_flat_replacement"] is False


def test_scene_preserving_border_is_light_but_not_uniform_white():
    source = _gradient_image()
    output, evidence = compose_scene_preserving_quiet_zone(
        source,
        minimum_luminance=0.78,
    )
    metrics = quiet_zone_metrics(output)
    assert metrics.luminance_p05 >= 0.77
    assert metrics.unique_color_count_capped > 2
    assert metrics.flat_uniform is False
    assert np.asarray(output).max() < 255
    assert evidence["delivery_guard_pass"] is True


def test_core_geometry_is_exact_580_pixels():
    source = _gradient_image()
    assert core_bounds(source, 78) == (78, 78, 658, 658)
