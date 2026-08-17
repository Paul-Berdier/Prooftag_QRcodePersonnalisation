from __future__ import annotations

import csv
import json

import pytest

from prooftag_qr.advisor_inference import (
    AdvisorInferenceRunner,
    build_advisor_inference_plan,
    load_advisor_inference_results,
    select_advisor_inference_winners,
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


class _Advisor:
    def recommend(self, *, candidates, limit, **_):
        output = []
        for rank, candidate in enumerate(candidates, start=1):
            probability = 0.99 - rank * 0.01
            output.append(
                ParameterRecommendation(
                    rank=rank,
                    candidate=candidate,
                    scan_safe=rank < 4,
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
            }
        )

    rows = load_advisor_inference_results(output_dir)
    winners = select_advisor_inference_winners(rows)

    assert len(rows) == 2
    assert rows[0]["predicted_qr_probability"] == pytest.approx(0.95)
    assert rows[0]["saturation_risk"] == pytest.approx(0.02)
    assert winners[0]["trial_id"] == "t2"
