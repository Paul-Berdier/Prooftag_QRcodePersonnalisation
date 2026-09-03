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
    assert len(build_candidates("smoke")) == 6
    assert len(build_candidates("pilot")) == 48
    assert len(build_candidates("full")) == 128


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


def test_each_prompt_receives_multiple_independent_candidates():
    pilot = build_candidates("pilot")
    counts = {}
    seeds = set()
    for candidate in pilot:
        counts[candidate.prompt_id] = counts.get(candidate.prompt_id, 0) + 1
        assert candidate.seed not in seeds
        seeds.add(candidate.seed)
    assert set(counts.values()) == {6}


def test_plan_declares_hard_validity_and_multiobjective_policy():
    plan = scientific_plan(
        profile="pilot",
        source_commit="a" * 40,
        e045_plan_id="288024247c39d585",
        e045_manifest_sha256="b" * 64,
    )
    assert plan["selected_parents_per_prompt"] == 2
    assert plan["selected_parent_count"] == 16
    assert plan["validity_policy"]["final_original_exact_required"] is True
    assert plan["validity_policy"]["final_minimum_exact_presets"] == 34
    assert plan["multiobjective_policy"]["weights"] == {
        "wechat_robustness": 0.40,
        "clip_prompt_alignment": 0.25,
        "hps_human_preference": 0.20,
        "clip_aesthetic": 0.15,
    }
