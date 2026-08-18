from __future__ import annotations

import pytest

from prooftag_qr.e027_policy import (
    build_e027_holdout_plan,
    build_e027_holdout_prompts,
    e027_policy_winner_entries,
    evaluate_e027_policies,
    select_e027_candidate,
)


def _row(
    prompt: str,
    seed: int,
    state: str,
    *,
    exact: float,
    tolerance: float,
    hps: float,
    aesthetic: float = 5.0,
    clip: float = 0.6,
    saturation: float = 0.01,
) -> dict:
    return {
        "prompt_id": prompt,
        "seed": seed,
        "role": "e027_pipeline_state",
        "pipeline_state": state,
        "method_id": f"e027_{state}",
        "status": "accepted" if exact else "rejected",
        "generation_run_id": f"{prompt}-{seed}-{state}",
        "qr_success": exact,
        "qr_tolerance": tolerance,
        "hpsv2_1": hps,
        "clip_aesthetic": aesthetic,
        "clip_score": clip,
        "saturation_risk": saturation,
        "duration_ms": 10_000.0,
        "output_variant": state,
    }


def test_e027_prompt_bank_is_deterministic_unique_and_skips_seen_texts():
    first = build_e027_holdout_prompts(12)
    second = build_e027_holdout_prompts(12)
    assert first == second
    assert len({item["text"] for item in first}) == 12
    assert {item["id"].split("_")[1] for item in first} == {"simple", "atypical"}

    without_first = build_e027_holdout_prompts(12, seen_prompt_texts=[first[0]["text"]])
    assert first[0]["text"] not in {item["text"] for item in without_first}
    assert len(without_first) == 12


def test_e027_plan_pairs_stage1_stage2_and_srmpgd_with_exact_reuse_order():
    prompts = build_e027_holdout_prompts(4)
    plan = build_e027_holdout_plan(
        payload="https://ptag.io/t/e027",
        prompts=prompts,
        seeds=(41, 53),
        prompts_per_campaign=2,
        qr_tolerance_threshold=0.80,
    )

    assert plan.public["protocol"] == "e027-v1-paired-cascade-full-forced-srmpgd"
    assert plan.public["context_count"] == 8
    assert plan.public["trial_count"] == 24
    assert len(plan.campaigns) == 2
    assert [item["id"] for item in plan.campaigns[0]["methods"]] == [
        "e027_stage1",
        "e027_stage2",
        "e027_srmpgd",
    ]
    stage2, srmpgd = plan.campaigns[0]["methods"][1:]
    assert stage2["tools"]["srpg_enabled"] is True
    assert stage2["tools"]["srmpgd_enabled"] is False
    assert srmpgd["tools"]["srpg_enabled"] is True
    assert srmpgd["tools"]["srmpgd_enabled"] is True
    assert srmpgd["tools"]["settings"]["srmpgd_min_qr_tolerance"] == 0.80
    for name in (
        "srpg_steps",
        "srpg_controlnet_scale",
        "srpg_qr_weight",
        "srpg_perceptual_weight",
        "diffqrcoder_stage2_strength",
        "diffqrcoder_stage2_target_mode",
    ):
        assert stage2["tools"]["settings"][name] == srmpgd["tools"]["settings"][name]
    assert {row["pipeline_state"] for row in plan.predictions} == {
        "stage1",
        "stage2",
        "srmpgd",
    }
    assert "https://ptag.io/t/e027" not in str(plan.public)


def test_e027_selection_is_qr_first_then_tolerance_then_visual_quality():
    rows = [
        _row("p1", 1, "stage1", exact=0, tolerance=1.0, hps=0.9),
        _row("p1", 1, "stage2", exact=1, tolerance=0.8, hps=0.4),
        _row("p1", 1, "srmpgd", exact=1, tolerance=0.9, hps=0.1),
    ]
    assert select_e027_candidate(rows)["pipeline_state"] == "srmpgd"

    tied = [
        _row("p2", 1, "stage2", exact=1, tolerance=0.9, hps=0.5),
        _row("p2", 1, "srmpgd", exact=1, tolerance=0.9, hps=0.2),
    ]
    assert select_e027_candidate(tied)["pipeline_state"] == "stage2"


def test_e027_compares_cascade_full_selection_and_forced_srmpgd():
    rows = [
        # The cascade stops immediately. The full policy buys tolerance; forced agrees.
        _row("p1", 1, "stage1", exact=1, tolerance=0.90, hps=0.50),
        _row("p1", 1, "stage2", exact=1, tolerance=0.95, hps=0.30),
        _row("p1", 1, "srmpgd", exact=1, tolerance=0.96, hps=0.20),
        # Stage 1 fails. Stage 2 and SR-MPGD tie on QR, so aesthetics keeps Stage 2.
        _row("p2", 2, "stage1", exact=0, tolerance=0.10, hps=0.70),
        _row("p2", 2, "stage2", exact=1, tolerance=0.85, hps=0.60),
        _row("p2", 2, "srmpgd", exact=1, tolerance=0.85, hps=0.20),
        # Stage 1 decodes but misses the robust gate. SR-MPGD is genuinely best.
        _row("p3", 3, "stage1", exact=1, tolerance=0.70, hps=0.80),
        _row("p3", 3, "stage2", exact=1, tolerance=0.85, hps=0.60),
        _row("p3", 3, "srmpgd", exact=1, tolerance=0.90, hps=0.40),
    ]

    report = evaluate_e027_policies(rows, qr_tolerance_threshold=0.80)
    decisions = {
        (row["prompt_id"], row["policy"]): row for row in report["decisions"]
    }
    assert decisions[("p1", "cascade")]["selected_state"] == "stage1"
    assert decisions[("p1", "cascade")]["estimated_generation_units"] == 1
    assert decisions[("p1", "full_lexicographic")]["selected_state"] == "srmpgd"
    assert decisions[("p2", "full_lexicographic")]["selected_state"] == "stage2"
    assert decisions[("p2", "forced_srmpgd")]["selected_state"] == "srmpgd"
    assert decisions[("p3", "cascade")]["selected_state"] == "srmpgd"
    assert report["policies"]["cascade"]["estimated_generation_units"] == 7
    assert report["policies"]["full_lexicographic"]["selected_state_counts"] == {
        "srmpgd": 2,
        "stage2": 1,
    }
    assert report["policies"]["forced_srmpgd"]["delivery_gate_success_rate"] == 1.0
    assert report["policies"]["full_lexicographic"]["prompts"] == 3
    assert (
        report["policies"]["full_lexicographic"]["prompt_all_seed_success_rate"]
        == 1.0
    )
    assert report["paired_comparisons"][
        "full_lexicographic_vs_forced_srmpgd"
    ]["same_selected_run"] == 2

    winners = e027_policy_winner_entries(rows, report["decisions"])
    assert len(winners) == 9
    assert {item["policy"] for item in winners} == {
        "cascade",
        "full_lexicographic",
        "forced_srmpgd",
    }


def test_e027_counts_a_missing_srmpgd_state_as_technical_incompleteness():
    rows = [
        _row("p1", 1, "stage1", exact=0, tolerance=0.0, hps=0.4),
        _row("p1", 1, "stage2", exact=1, tolerance=0.9, hps=0.3),
    ]
    report = evaluate_e027_policies(rows)
    assert report["contexts"] == 1
    assert report["complete_contexts"] == 0
    forced = next(
        row for row in report["decisions"] if row["policy"] == "forced_srmpgd"
    )
    assert forced["selected"] is False
    assert forced["deliverable"] is False
    assert report["policies"]["forced_srmpgd"]["exact_qr_success_rate"] == 0.0


def test_e027_rejects_training_prompt_overlap():
    prompt = build_e027_holdout_prompts(1)[0]
    with pytest.raises(ValueError, match="already occur in training"):
        build_e027_holdout_plan(
            payload="https://ptag.io/t/e027",
            prompts=[prompt],
            seen_prompt_texts=[prompt["text"]],
            seeds=(41,),
        )
