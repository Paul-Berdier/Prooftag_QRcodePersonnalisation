from prooftag_qr.e046_campaign import (
    _annotate_prompt_objectives,
    _pareto_front,
    _prompt_tournament_rank,
)


def _plan():
    return {
        "validity_policy": {
            "final_original_exact_required": True,
            "final_minimum_exact_presets": 34,
            "ideal_minimum_exact_presets": 37,
            "refinement_minimum_exact_presets": 16,
        },
        "multiobjective_policy": {
            "weights": {
                "wechat_robustness": 0.40,
                "clip_prompt_alignment": 0.25,
                "hps_human_preference": 0.20,
                "clip_aesthetic": 0.15,
            }
        },
    }


def _row(
    name,
    *,
    exact,
    original,
    clip,
    hps,
    aesthetic,
    mer=0.05,
    eligible=True,
    prompt="p1",
):
    return {
        "candidate_id": name,
        "prompt_id": prompt,
        "prompt_variant_index": 0,
        "stage": "stage2",
        "source_kind": "parent",
        "variant": "stage2_raw",
        "quiet_zone_variant": "raw",
        "eligible_final": eligible,
        "visual_guard_pass": eligible,
        "uniform_quiet_zone_replacement": False,
        "wechat_exact_presets": exact,
        "wechat_original_exact": original,
        "clip_score": clip,
        "hpsv2_1": hps,
        "clip_aesthetic": aesthetic,
        "module_error_rate": mer,
        "lpips": 0.01,
    }


def test_beauty_cannot_compensate_for_invalid_qr():
    rows = _annotate_prompt_objectives(
        [
            _row(
                "valid",
                exact=35,
                original=True,
                clip=0.70,
                hps=0.20,
                aesthetic=5.0,
            ),
            _row(
                "beautiful-invalid",
                exact=5,
                original=False,
                clip=0.95,
                hps=0.45,
                aesthetic=7.5,
            ),
        ],
        _plan(),
    )
    ordered = sorted(rows, key=_prompt_tournament_rank)
    assert ordered[0]["candidate_id"] == "valid"
    assert ordered[0]["software_valid_final"] is True
    assert ordered[1]["software_valid_final"] is False


def test_inside_same_validity_tier_prompt_and_beauty_scores_choose_winner():
    rows = _annotate_prompt_objectives(
        [
            _row(
                "beautiful",
                exact=36,
                original=True,
                clip=0.85,
                hps=0.35,
                aesthetic=6.5,
            ),
            _row(
                "plain",
                exact=36,
                original=True,
                clip=0.55,
                hps=0.15,
                aesthetic=4.0,
            ),
        ],
        _plan(),
    )
    ordered = sorted(rows, key=_prompt_tournament_rank)
    assert ordered[0]["candidate_id"] == "beautiful"
    assert ordered[0]["software_validity_tier"] == 2
    assert ordered[0]["multiobjective_prompt_score"] > ordered[1][
        "multiobjective_prompt_score"
    ]


def test_pareto_excludes_non_valid_final_rows():
    rows = _annotate_prompt_objectives(
        [
            _row(
                "valid-a",
                exact=36,
                original=True,
                clip=0.70,
                hps=0.20,
                aesthetic=5.0,
            ),
            _row(
                "valid-b",
                exact=34,
                original=True,
                clip=0.90,
                hps=0.35,
                aesthetic=6.0,
            ),
            _row(
                "invalid",
                exact=10,
                original=False,
                clip=0.99,
                hps=0.50,
                aesthetic=8.0,
            ),
        ],
        _plan(),
    )
    front = _pareto_front(rows)
    names = {row["candidate_id"] for row in front}
    assert "invalid" not in names
    assert names == {"valid-a", "valid-b"}


def test_parent_selection_is_per_prompt_not_global(tmp_path):
    import json
    from prooftag_qr.e046_campaign import select_parents
    from prooftag_qr.e046_catalog import scientific_plan

    output = tmp_path / "e046"
    plan = scientific_plan(
        profile="smoke",
        source_commit="a" * 40,
        e045_plan_id="288024247c39d585",
        e045_manifest_sha256="b" * 64,
    )
    plan_dir = output / plan["plan_id"]
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (plan_dir / "PARENT_SCORING_COMPLETE.json").write_text(
        json.dumps({"scored_count": len(plan["candidates"])}),
        encoding="utf-8",
    )

    for index, candidate in enumerate(plan["candidates"]):
        scoring = plan_dir / "parents" / candidate["id"] / "scoring"
        scoring.mkdir(parents=True)
        # Prompt 2 is deliberately much weaker than prompt 1. A global selector
        # would omit it; the tournament must still select its best candidate.
        prompt_number = 0 if candidate["prompt_id"].startswith("p01") else 1
        row = _row(
            candidate["id"],
            exact=(36 - index) if prompt_number == 0 else (12 - index % 3),
            original=prompt_number == 0,
            clip=0.60 + (index % 3) * 0.05,
            hps=0.18 + (index % 3) * 0.02,
            aesthetic=4.8 + (index % 3) * 0.2,
            prompt=candidate["prompt_id"],
        )
        row.update(
            {
                "prompt": candidate["prompt"],
                "prompt_family": candidate["prompt_family"],
                "parent_recipe_id": candidate["parent_recipe_id"],
                "seed": candidate["seed"],
            }
        )
        (scoring / "comparison.json").write_text(
            json.dumps([row]), encoding="utf-8"
        )

    result = select_parents(output_root=output, plan_id=plan["plan_id"])
    assert result["prompt_count"] == 2
    assert result["selected_parent_count"] == 2
    assert {row["prompt_id"] for row in result["selected"]} == {
        "p01_brutalist_courtyard",
        "p02_bioluminescent_mycelium",
    }
