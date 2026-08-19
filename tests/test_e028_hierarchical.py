from __future__ import annotations

from dataclasses import replace

from prooftag_qr.e028_hierarchical import (
    E028_PIPELINE_STATES,
    audit_e028_pairing,
    build_e028_conditional_datasets,
    build_e028_hierarchical_plan,
    evaluate_e028_policies,
)
from prooftag_qr.parameter_advisor import ParameterRecommendation


class DeterministicAdvisor:
    def recommend(self, *, candidates, limit, **_):
        ranked = []
        for index, candidate in enumerate(candidates):
            generation = candidate.configuration.get("generation") or {}
            settings = (candidate.configuration.get("tools") or {}).get("settings") or {}
            score = (
                float(generation.get("controlnet_scale", 0.0))
                + float(settings.get("diffqrcoder_stage2_strength", 0.0))
                + float(settings.get("srmpgd_step_size", 0.0)) / 1000.0
            )
            ranked.append(
                ParameterRecommendation(
                    rank=0,
                    candidate=candidate,
                    scan_safe=score >= 1.0,
                    predicted_qr_success=min(0.99, 0.55 + score / 10.0),
                    qr_success_uncertainty=0.02,
                    qr_success_lower_bound=min(0.97, 0.50 + score / 10.0),
                    predicted_qr_tolerance=min(0.99, 0.60 + score / 10.0),
                    predicted_clip_aesthetic=5.0 + index / 100.0,
                    predicted_clip_score=0.5 + index / 1000.0,
                    predicted_hpsv2_1=0.2 + index / 1000.0,
                    predicted_human_aesthetic=None,
                    predicted_human_prompt_fidelity=None,
                    predicted_human_qr_discretion=None,
                    predicted_human_overall=None,
                    predicted_duration_ms=10_000.0,
                    predicted_saturation_risk=0.01,
                )
            )
        ranked.sort(
            key=lambda item: (
                item.scan_safe,
                item.qr_success_lower_bound,
                item.predicted_clip_aesthetic,
            ),
            reverse=True,
        )
        return [replace(item, rank=index) for index, item in enumerate(ranked[:limit], 1)]


def test_e028_plan_uses_prompt_advisor_at_every_stage_and_orders_exact_reuse():
    plan = build_e028_hierarchical_plan(
        advisor=DeterministicAdvisor(),
        candidates=[],
        prompts=[
            {
                "id": "e028_test",
                "text": "A quiet blue glass sculpture under soft gallery light.",
            }
        ],
        payload="https://ptag.io/t/e028-test",
        advisor_sha256="a" * 64,
        seeds=(101, 202, 303),
        stage1_top_k=2,
        stage2_top_k=2,
    )

    assert plan.public["context_count"] == 3
    assert plan.public["trial_count"] == 39
    assert plan.public["candidate_pool_counts"]["stage1"] >= 2
    assert plan.public["candidate_pool_counts"]["stage2"] >= 2
    assert plan.public["candidate_pool_counts"]["srmpgd"] >= 2
    states = [item["pipeline_state"] for item in plan.predictions]
    assert states == sorted(states, key=E028_PIPELINE_STATES.index)
    assert states.count("stage1") == 3
    assert states.count("stage2") == 5
    assert states.count("srmpgd") == 5
    advisor_rows = [
        item for item in plan.predictions if not item["fixed_control"]
    ]
    assert {item["pipeline_state"] for item in advisor_rows} == {
        "stage1",
        "stage2",
        "srmpgd",
    }
    assert all(
        item["parent_stage1_method_id"]
        for item in advisor_rows
        if item["pipeline_state"] in {"stage2", "srmpgd"}
    )
    assert all(
        item["parent_stage2_method_id"]
        for item in advisor_rows
        if item["pipeline_state"] == "srmpgd"
    )
    assert "payload" not in plan.public
    assert plan.public["payload_sha256"]


def _row(
    method_id: str,
    state: str,
    *,
    fixed: bool = False,
    chain: str = "advisor-s1-01-s2-01",
    qr_success: float = 0.0,
    tolerance: float = 0.0,
):
    return {
        "trial_id": f"trial-{method_id}",
        "prompt_id": "prompt-1",
        "prompt_text": "A quiet blue glass sculpture.",
        "seed": 101.0,
        "method_id": method_id,
        "source_method_id": method_id,
        "role": f"e028_{'fixed' if fixed else 'advisor'}_{state}",
        "pipeline_state": state,
        "fixed_control": fixed,
        "chain_id": "fixed" if fixed else chain,
        "generation_run_id": f"run-{method_id}",
        "status": "accepted" if qr_success else "rejected",
        "qr_success": qr_success,
        "qr_tolerance": tolerance,
        "clip_aesthetic": 5.5,
        "clip_score": 0.65,
        "hpsv2_1": 0.25,
        "saturation_risk": 0.01,
        "duration_ms": 10_000.0,
        "candidate_configuration": {
            "id": method_id,
            "backend": "controlnet",
            "output_variant": "raw" if state == "stage1" else state,
            "generation": {"steps": 40},
            "model": {},
            "tools": {"settings": {}},
        },
        "payload_length": 28.0,
        "error_correction": "M",
    }


def test_e028_policy_never_delivers_stage1_and_uses_srmpgd_as_fallback():
    rows = [
        _row("fixed-s1", "stage1", fixed=True, qr_success=1.0, tolerance=1.0),
        _row("fixed-s2", "stage2", fixed=True, qr_success=1.0, tolerance=0.9),
        _row("fixed-m", "srmpgd", fixed=True, qr_success=1.0, tolerance=0.95),
        _row("advisor-s1", "stage1", qr_success=1.0, tolerance=1.0),
        _row("advisor-s2", "stage2", qr_success=1.0, tolerance=0.9),
        _row("advisor-m", "srmpgd", qr_success=1.0, tolerance=0.92),
    ]
    next(item for item in rows if item["method_id"] == "advisor-s2")[
        "saturation_risk"
    ] = 0.12
    report = evaluate_e028_policies(rows, qr_tolerance_threshold=0.8)

    assert all(value["stage1_deliveries"] == 0 for value in report["policies"].values())
    top1 = next(
        item for item in report["decisions"] if item["policy"] == "advisor_top1"
    )
    assert top1["selected_state"] == "srmpgd"
    assert top1["deliverable"] is True
    fixed = next(
        item for item in report["decisions"] if item["policy"] == "fixed_cascade"
    )
    assert fixed["selected_state"] == "stage2"


def test_e028_pairing_audit_proves_stage1_image_and_stage2_latent_reuse():
    stage1 = _row("advisor-s1", "stage1")
    stage1.update(
        {
            "generation_run_id": "run-stage1",
            "final_image_sha256": "image-hash",
        }
    )
    stage2 = _row("advisor-s2", "stage2")
    stage2.update(
        {
            "generation_run_id": "run-stage2",
            "parent_stage1_method_id": "advisor-s1",
            "stage1_reused": 1.0,
            "stage1_source_run_id": "run-stage1",
            "stage2_latent_sha256": "latent-hash",
        }
    )
    srmpgd = _row("advisor-m", "srmpgd")
    srmpgd.update(
        {
            "generation_run_id": "run-mpgd",
            "parent_stage1_method_id": "advisor-s1",
            "parent_stage2_method_id": "advisor-s2",
            "stage1_reused": 1.0,
            "stage1_source_run_id": "run-stage1",
            "stage2_source_run_id": "run-stage2",
            "stage2_source_latent_sha256": "latent-hash",
            "stage2_pairing_status": "exact_reuse",
            "stage2_pairing_exact": 1.0,
        }
    )

    audit = audit_e028_pairing([stage1, stage2, srmpgd])
    assert len(audit) == 2
    assert all(item["complete"] for item in audit)
    assert next(item for item in audit if item["pipeline_state"] == "srmpgd")[
        "stage2_exact_reuse"
    ]


def test_e028_conditional_datasets_include_measured_parent_features():
    stage1 = _row("advisor-s1", "stage1", qr_success=0.0, tolerance=0.2)
    stage2 = _row("advisor-s2", "stage2", qr_success=1.0, tolerance=0.9)
    stage2["parent_stage1_method_id"] = "advisor-s1"
    srmpgd = _row("advisor-m", "srmpgd", qr_success=1.0, tolerance=0.95)
    srmpgd["parent_stage2_method_id"] = "advisor-s2"

    datasets = build_e028_conditional_datasets([stage1, stage2, srmpgd])
    assert len(datasets["stage2"].records) == 1
    assert len(datasets["srmpgd"].records) == 1
    assert datasets["stage2"].records[0].context_features[
        "parent_qr_tolerance"
    ] == 0.2
    assert datasets["srmpgd"].records[0].context_features[
        "parent_qr_tolerance"
    ] == 0.9
