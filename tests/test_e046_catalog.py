from prooftag_qr.e046_catalog import (
    PARENT_RECIPES,
    PROMPTS,
    QR_SOFTWARE_ENGINE,
    SRMPGD_RECIPES,
    build_candidates,
    scientific_plan,
)


def test_catalog_covers_eight_masks_and_varied_profiles():
    assert len(PROMPTS) == 8
    assert len(PARENT_RECIPES) == 8
    assert {recipe.qr_mask_pattern for recipe in PARENT_RECIPES} == set(range(8))
    assert len(SRMPGD_RECIPES) == 4
    assert len(build_candidates("smoke")) == 2
    assert len(build_candidates("pilot")) == 8
    assert len(build_candidates("full")) == 24


def test_primary_engine_is_wechat_only():
    assert "qr-scanner-wechat" in QR_SOFTWARE_ENGINE
    assert "opencv" not in QR_SOFTWARE_ENGINE.lower()
    assert "zxing" not in QR_SOFTWARE_ENGINE.lower()


def test_plan_is_stable_and_forbids_uniform_final_border():
    kwargs = {
        "profile": "pilot",
        "source_commit": "a" * 40,
        "e045_plan_id": "288024247c39d585",
        "e045_manifest_sha256": "b" * 64,
    }
    first = scientific_plan(**kwargs)
    second = scientific_plan(**kwargs)
    assert first["scientific_plan_hash"] == second["scientific_plan_hash"]
    assert first["plan_id"] == second["plan_id"]
    assert first["quiet_zone_policy"]["flat_white_or_uniform_replacement_eligible"] is False
    assert first["phone_truth_available"] is False
    assert first["production_ready"] is False
