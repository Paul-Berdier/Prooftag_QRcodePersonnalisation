from __future__ import annotations

import csv
import json

import pytest

from prooftag_qr.advisor_inference import (
    AdvisorInferenceRunner,
    build_advisor_inference_plan,
    deduplicate_advisor_inference_results,
    effective_candidate_signature,
    load_advisor_inference_results,
    select_advisor_inference_winners,
    summarize_advisor_inference_results,
)
from prooftag_qr.parameter_advisor import ParameterRecommendation, RecipeCandidate


def _candidate(identifier: str, index: int) -> RecipeCandidate:
    return RecipeCandidate(
        id=f"recipe-{index}",
        method_id=identifier,
        signature=f"signature-{index}",
        observations=100 - index,
        configuration={
            "id": identifier,
            "name": identifier,
            "backend": "controlnet",
            "enabled": True,
            "output_variant": "raw",
            "reuse_stage1": True,
            "generation": {
                "steps": 20 + index,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "model": {},
            "tools": {
                "srpg_enabled": False,
                "srmpgd_enabled": False,
                "settings": {},
            },
        },
    )


def _srmpgd_candidate(
    identifier: str,
    index: int,
    *,
    gamma: float,
    strength: float = 0.65,
) -> RecipeCandidate:
    candidate = _candidate(identifier, index)
    candidate.configuration.update(
        {
            "output_variant": "srmpgd",
            "generation": {
                "steps": 40,
                "guidance_scale": 7.5,
                "controlnet_scale": 1.35,
                "strength": 1.0,
            },
            "tools": {
                "srpg_enabled": True,
                "srmpgd_enabled": True,
                "settings": {
                    "srpg_steps": 40,
                    "srpg_qr_weight": 500.0,
                    "srpg_perceptual_weight": 2.0,
                    "diffqrcoder_stage2_strength": strength,
                    "srmpgd_max_iterations": 4,
                    "srmpgd_step_size": gamma,
                    "srmpgd_lpips_weight": 0.1,
                },
            },
        }
    )
    return candidate


class _Advisor:
    def __init__(self, *, safe_count: int = 3):
        self.safe_count = safe_count

    def recommend(self, *, candidates, limit, **_):
        output = []
        for rank, candidate in enumerate(candidates, start=1):
            probability = 0.99 - rank * 0.01
            output.append(
                ParameterRecommendation(
                    rank=rank,
                    candidate=candidate,
                    scan_safe=rank <= self.safe_count,
                    predicted_qr_success=probability,
                    qr_success_uncertainty=0.02,
                    qr_success_lower_bound=probability - 0.02,
                    predicted_qr_tolerance=0.8 - rank * 0.01,
                    predicted_clip_aesthetic=5.0 + rank * 0.1,
                    predicted_clip_score=0.6,
                    predicted_hpsv2_1=0.2,
                    predicted_human_aesthetic=None,
                    predicted_human_prompt_fidelity=None,
                    predicted_human_qr_discretion=None,
                    predicted_human_overall=None,
                    predicted_duration_ms=10_000.0,
                    predicted_saturation_risk=0.01,
                )
            )
        return output[:limit]


def _plan():
    candidates = [
        _candidate("recommended_a", 1),
        _candidate("recommended_b", 2),
        _candidate("diffqrcoder_stage1", 3),
    ]
    return build_advisor_inference_plan(
        advisor=_Advisor(),
        candidates=candidates,
        prompts=[
            {"id": "unseen_simple", "text": "A single red chair in a white studio."},
            {
                "id": "unseen_atypical",
                "text": "A glass nautilus folding moonlight into origami corridors.",
            },
        ],
        payload="https://ptag.io/t/e026i",
        advisor_sha256="a" * 64,
        prompt_embedding_provider=lambda _: [0.1, 0.2],
        seen_prompt_texts=["A training-only blue vase."],
        seeds=(41, 53),
        top_k=2,
    )


def test_inference_plan_is_deterministic_redacted_and_uses_advisor_top_k():
    first = _plan()
    second = _plan()

    assert first.plan_id == second.plan_id
    assert first.public == second.public
    assert first.public["trial_count"] == 12
    assert "https://ptag.io/t/e026i" not in json.dumps(first.public)
    assert first.public["payload_length"] == len(first.payload)
    assert len(first.campaigns) == 2
    assert all(len(item["methods"]) == 3 for item in first.campaigns)
    assert all(len(item["seeds"]) == 2 for item in first.campaigns)
    assert {item["role"] for item in first.predictions} == {
        "advisor_recommendation",
        "fixed_baseline",
    }
    assert {
        item["source_method_id"]
        for item in first.predictions
        if item["role"] == "advisor_recommendation"
    } == {"recommended_a", "recommended_b"}


def test_inference_plan_rejects_a_prompt_seen_during_training():
    with pytest.raises(ValueError, match="already occur in training"):
        build_advisor_inference_plan(
            advisor=_Advisor(),
            candidates=[
                _candidate("recommended_a", 1),
                _candidate("diffqrcoder_stage1", 2),
            ],
            prompts=[{"id": "duplicate", "text": " The SAME training prompt. "}],
            payload="https://ptag.io/t/e026i",
            advisor_sha256="b" * 64,
            seen_prompt_texts=["the same training prompt."],
            top_k=1,
        )


def test_e026j_selects_robust_balanced_and_aesthetic_effective_recipes():
    candidates = [
        _candidate("recipe_robust", 1),
        _candidate("recipe_balanced", 2),
        _candidate("recipe_aesthetic", 3),
        _candidate("diffqrcoder_stage1", 4),
    ]
    plan = build_advisor_inference_plan(
        advisor=_Advisor(),
        candidates=candidates,
        prompts=[{"id": "unseen", "text": "A blue cup on a wooden shelf."}],
        payload="https://ptag.io/t/e026j",
        advisor_sha256="d" * 64,
        seeds=(41,),
        top_k=3,
    )

    selected = [
        row for row in plan.predictions if row["role"] == "advisor_recommendation"
    ]
    assert [row["selection_profile"] for row in selected] == [
        "robust",
        "balanced",
        "aesthetic_scannable",
    ]
    assert len({row["effective_candidate_signature"] for row in selected}) == 3
    assert plan.public["comparison_trial_count"] == 4
    assert plan.public["scan_safe_recommendation_count"] == 3
    assert plan.public["exploratory_recommendation_count"] == 0


def test_e026j_fills_a_missing_scan_safe_slot_with_an_explicit_exploration():
    candidates = [
        _candidate("recipe_robust", 1),
        _candidate("recipe_balanced", 2),
        _candidate("recipe_exploratory", 3),
        _candidate("diffqrcoder_stage1", 4),
    ]
    plan = build_advisor_inference_plan(
        advisor=_Advisor(safe_count=2),
        candidates=candidates,
        prompts=[{"id": "unseen", "text": "A paper lighthouse in violet fog."}],
        payload="https://ptag.io/t/e026j",
        advisor_sha256="e" * 64,
        seeds=(41,),
        top_k=3,
    )

    selected = [
        row for row in plan.predictions if row["role"] == "advisor_recommendation"
    ]
    assert [row["selection_profile"] for row in selected] == [
        "robust",
        "balanced",
        "aesthetic_exploratory",
    ]
    assert [row["scan_safe"] for row in selected] == [True, True, False]
    assert len({row["effective_candidate_signature"] for row in selected}) == 3
    assert plan.public["protocol"] == "e026j-v2-scan-safe-exploratory-fallback"
    assert plan.public["scan_safe_recommendation_count"] == 2
    assert plan.public["exploratory_recommendation_count"] == 1


def test_inference_plan_materializes_and_deduplicates_paired_srpg_prerequisites():
    candidates = [
        _srmpgd_candidate("srmpgd_gamma_30", 1, gamma=30.0),
        _srmpgd_candidate("srmpgd_gamma_300", 2, gamma=300.0),
        _srmpgd_candidate(
            "srmpgd_strength_80", 3, gamma=100.0, strength=0.80
        ),
        _candidate("diffqrcoder_stage1", 4),
    ]
    plan = build_advisor_inference_plan(
        advisor=_Advisor(),
        candidates=candidates,
        prompts=[{"id": "unseen", "text": "A silver bicycle in morning fog."}],
        payload="https://ptag.io/t/e026i",
        advisor_sha256="c" * 64,
        seeds=(41, 53),
        top_k=2,
    )

    methods = plan.campaigns[0]["methods"]
    prerequisites = [
        item
        for item in methods
        if item["id"].startswith("e026j_dep_")
    ]
    assert len(prerequisites) == 2
    assert prerequisites[0]["output_variant"] == "srpg"
    assert prerequisites[0]["tools"]["srpg_enabled"] is True
    assert prerequisites[0]["tools"]["srmpgd_enabled"] is False
    assert not any(
        key.startswith("srmpgd_")
        for key in prerequisites[0]["tools"]["settings"]
    )
    assert [item["output_variant"] for item in methods] == [
        "srpg",
        "auto",
        "srpg",
        "auto",
        "raw",
    ]
    adaptive = [item for item in methods if item["output_variant"] == "auto"]
    assert all(
        item["tools"]["settings"]["srmpgd_min_qr_tolerance"] == 0.80
        for item in adaptive
    )
    assert plan.public["protocol"] == "e026j-v2-scan-safe-exploratory-fallback"
    assert plan.public["trial_count"] == 10
    assert plan.public["comparison_trial_count"] == 6
    assert plan.public["prerequisite_trial_count"] == 4
    assert sum(
        row["role"] == "srmpgd_prerequisite" for row in plan.predictions
    ) == 2
    recommendations = [
        row for row in plan.predictions if row["role"] == "advisor_recommendation"
    ]
    assert {row["selection_profile"] for row in recommendations} == {
        "robust",
        "aesthetic_scannable",
    }
    assert len({row["effective_candidate_signature"] for row in recommendations}) == 2
    assert effective_candidate_signature(candidates[0]) == effective_candidate_signature(
        candidates[1]
    )


def test_inference_runner_resumes_without_resubmitting_completed_campaign(tmp_path):
    plan = _plan()
    runner = AdvisorInferenceRunner(
        plan=plan,
        api_url="http://example.invalid",
        output_root=tmp_path,
        poll_seconds=0,
    )
    calls = []

    def request(method, path, payload=None, *, raw=False):
        calls.append((method, path, payload, raw))
        if path.endswith("?limit=100") or path.endswith("?limit=500"):
            return []
        if method == "POST":
            return {"id": f"campaign-{len([item for item in calls if item[0] == 'POST'])}"}
        if path.endswith("/results.csv"):
            return b"trial_id,prompt_id,method_id,status\n1,p,m,accepted\n"
        return {
            "id": path.rsplit("/", 1)[-1],
            "status": "completed",
            "completed_trials": 6,
            "total_trials": 6,
            "accepted_trials": 6,
            "trials": [],
        }

    runner._request = request
    first = runner.run()
    second = runner.run()

    assert first["status"] == "completed"
    assert second == first
    assert first["completed_campaigns"] == 2
    assert first["exports"] == 2
    assert len([item for item in calls if item[0] == "POST"]) == 2


def test_inference_results_join_predictions_and_select_scannable_winner(tmp_path):
    output_dir = tmp_path / "plan"
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True)
    predictions = [
        {
            "prompt_id": "p1",
            "prompt_text": "Prompt one",
            "plan_method_id": "e026i_r01_a",
            "source_method_id": "a",
            "role": "advisor_recommendation",
            "advisor_rank": 1,
            "predicted_qr_success": 0.95,
        },
        {
            "prompt_id": "p1",
            "prompt_text": "Prompt one",
            "plan_method_id": "e026i_r02_b",
            "source_method_id": "b",
            "role": "advisor_recommendation",
            "advisor_rank": 2,
            "predicted_qr_success": 0.90,
            "requested_source_output_variant": "srmpgd",
        },
    ]
    (output_dir / "advisor-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in predictions), encoding="utf-8"
    )
    fields = [
        "trial_id",
        "campaign_id",
        "prompt_id",
        "method_id",
        "status",
        "seed",
        "generation_run_id",
        "selected_variant",
        "quality_qr_verify_any_exact",
        "quality_qr_verify_tolerance_score",
        "quality_clip_aesthetic",
        "quality_clip_score",
        "quality_hpsv2_1",
        "quality_high_saturation_pixel_ratio",
        "quality_rgb_clipped_channel_ratio",
        "total_ms",
        "module_error_rate",
        "quality_diffqrcoder_srmpgd_selected_iteration",
        "error",
    ]
    with (exports_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "t1",
                "campaign_id": "c1",
                "prompt_id": "p1",
                "method_id": "e026i_r01_a",
                "status": "rejected",
                "seed": 1,
                "generation_run_id": "run-1",
                "selected_variant": "srpg",
                "quality_qr_verify_any_exact": 0,
                "quality_qr_verify_tolerance_score": 0.9,
                "quality_clip_aesthetic": 8.0,
                "quality_clip_score": 0.8,
                "quality_hpsv2_1": 0.3,
                "quality_high_saturation_pixel_ratio": 0.01,
                "quality_rgb_clipped_channel_ratio": 0.02,
                "total_ms": 1000,
                "module_error_rate": 0.1,
            }
        )
        writer.writerow(
            {
                "trial_id": "t2",
                "campaign_id": "c1",
                "prompt_id": "p1",
                "method_id": "e026i_r02_b",
                "status": "accepted",
                "seed": 1,
                "generation_run_id": "run-2",
                "selected_variant": "srmpgd",
                "quality_qr_verify_any_exact": 1,
                "quality_qr_verify_tolerance_score": 0.7,
                "quality_clip_aesthetic": 6.0,
                "quality_clip_score": 0.6,
                "quality_hpsv2_1": 0.2,
                "quality_high_saturation_pixel_ratio": 0.01,
                "quality_rgb_clipped_channel_ratio": 0.01,
                "total_ms": 2000,
                "module_error_rate": 0.05,
                "quality_diffqrcoder_srmpgd_selected_iteration": 0,
            }
        )

    rows = load_advisor_inference_results(output_dir)
    winners = select_advisor_inference_winners(rows)

    assert len(rows) == 2
    assert rows[0]["predicted_qr_probability"] == pytest.approx(0.95)
    assert rows[0]["saturation_risk"] == pytest.approx(0.02)
    assert rows[1]["output_variant"] == "srpg"
    assert rows[1]["srmpgd_noop"] is True
    assert rows[1]["srmpgd_effective"] is False
    assert winners[0]["trial_id"] == "t2"


def test_inference_summary_counts_missing_measurements_as_technical_failures():
    rows = [
        {
            "prompt_id": "p1",
            "seed": 1,
            "role": "advisor_recommendation",
            "advisor_rank": 1,
            "status": "accepted",
            "qr_success": 1.0,
            "clip_aesthetic": 5.0,
            "clip_score": 0.6,
            "hpsv2_1": 0.2,
        },
        {
            "prompt_id": "p1",
            "seed": 2,
            "role": "advisor_recommendation",
            "advisor_rank": 1,
            "status": "error",
            "qr_success": None,
        },
        {
            "prompt_id": "p1",
            "seed": 1,
            "role": "advisor_recommendation",
            "advisor_rank": 2,
            "status": "rejected",
            "qr_success": 0.0,
        },
        {
            "prompt_id": "p1",
            "seed": 2,
            "role": "advisor_recommendation",
            "advisor_rank": 2,
            "status": "error",
            "qr_success": None,
        },
        {
            "prompt_id": "p1",
            "seed": 1,
            "role": "fixed_baseline",
            "advisor_rank": 9,
            "status": "accepted",
            "qr_success": 1.0,
        },
        {
            "prompt_id": "p1",
            "seed": 1,
            "role": "srmpgd_prerequisite",
            "advisor_rank": 1,
            "status": "accepted",
            "qr_success": 1.0,
        },
    ]

    summary = summarize_advisor_inference_results(rows)

    assert summary["images_planned"] == 5
    assert summary["images_measured"] == 3
    assert summary["technical_error_images"] == 2
    assert summary["rank1_qr_verify_success_rate"] == pytest.approx(0.5)
    assert summary["rank1_qr_verify_success_rate_generated"] == pytest.approx(1.0)
    assert summary["top_k_image_qr_verify_success_rate"] == pytest.approx(0.25)
    assert summary["top_k_prompt_seed_coverage"] == pytest.approx(0.5)
    assert summary["baseline_qr_verify_success_rate"] == pytest.approx(1.0)


def test_inference_deduplicates_measured_images_and_reports_noop_srmpgd():
    rows = [
        {
            "trial_id": "t1",
            "prompt_id": "p1",
            "seed": 41,
            "role": "advisor_recommendation",
            "advisor_rank": 1,
            "source_method_id": "srmpgd-g30",
            "requested_source_output_variant": "srmpgd",
            "final_image_sha256": "same-image",
            "srmpgd_selected_iteration": 0.0,
            "srmpgd_noop": True,
            "srmpgd_effective": False,
            "status": "accepted",
            "qr_success": 1.0,
            "qr_tolerance": 0.9,
        },
        {
            "trial_id": "t2",
            "prompt_id": "p1",
            "seed": 41,
            "role": "advisor_recommendation",
            "advisor_rank": 2,
            "source_method_id": "srmpgd-g300",
            "requested_source_output_variant": "srmpgd",
            "final_image_sha256": "same-image",
            "srmpgd_selected_iteration": 0.0,
            "srmpgd_noop": True,
            "srmpgd_effective": False,
            "status": "accepted",
            "qr_success": 1.0,
            "qr_tolerance": 0.8,
        },
        {
            "trial_id": "t3",
            "prompt_id": "p1",
            "seed": 41,
            "role": "advisor_recommendation",
            "advisor_rank": 3,
            "source_method_id": "different-stage2",
            "requested_source_output_variant": "srpg",
            "final_image_sha256": "different-image",
            "srmpgd_selected_iteration": None,
            "srmpgd_noop": False,
            "srmpgd_effective": False,
            "status": "accepted",
            "qr_success": 1.0,
            "qr_tolerance": 0.7,
        },
    ]

    unique = deduplicate_advisor_inference_results(rows)
    summary = summarize_advisor_inference_results(rows)

    assert len(unique) == 2
    assert unique[0]["duplicate_count"] == 1
    assert unique[0]["duplicate_method_ids"] == ["srmpgd-g30", "srmpgd-g300"]
    assert summary["top_k_images_measured"] == 3
    assert summary["top_k_unique_images_measured"] == 2
    assert summary["top_k_duplicate_images"] == 1
    assert summary["srmpgd_requested_images"] == 2
    assert summary["srmpgd_effective_images"] == 0
    assert summary["srmpgd_noop_images"] == 2
