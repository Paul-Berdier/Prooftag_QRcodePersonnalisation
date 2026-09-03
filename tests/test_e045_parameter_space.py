from prooftag_qr.e045_parameter_space import (
    PARAMETERS,
    effective_signature,
    validate_geometry,
)


def test_parameter_space_covers_the_full_pipeline():
    assert len(PARAMETERS) >= 90
    stages = {item["stage"] for item in PARAMETERS}
    assert {
        "qr", "prompt", "stage1", "stage2",
        "srmpgd", "evaluation", "runtime",
    } <= stages


def test_effective_signature_ignores_aliases_but_not_gamma():
    a = {"id": "recipe-a", "name": "A", "srmpgd": {"gamma": 500.0}}
    b = {"id": "recipe-b", "name": "B", "srmpgd": {"gamma": 500.0}}
    c = {"id": "recipe-c", "name": "C", "srmpgd": {"gamma": 1000.0}}
    assert effective_signature(a) == effective_signature(b)
    assert effective_signature(a) != effective_signature(c)


def test_geometry_forbids_an_insufficient_canvas_and_short_quiet_zone():
    errors = validate_geometry(
        {
            "qr_version": 3,
            "module_size_px": 20,
            "quiet_zone_modules": 3,
            "canvas_px": 640,
        }
    )
    assert any("quiet_zone_modules" in error for error in errors)
    assert any("canvas insuffisant" in error for error in errors)
