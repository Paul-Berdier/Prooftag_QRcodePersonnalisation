import hashlib

import pytest

from prooftag_qr.policy import (
    ConservativeDeliveryGate,
    assess_stage2_candidate,
    attempts_for_target,
    best_candidate,
    conservative_qr_score,
    deliverable_candidate,
    delivery_probability,
    select_stage2_cascade,
)


def test_delivery_probability_and_required_budget():
    assert delivery_probability(0.8, 3) == pytest.approx(0.992)
    assert attempts_for_target(0.8, 0.99) == 3
    assert attempts_for_target(0.0, 0.99) is None


def test_delivery_gate_precedes_aesthetics():
    beautiful_failure = {
        "strict_all": False,
        "original_pass_rate": 1.0,
        "pass_rate": 0.96,
        "clip_aesthetic": 9.0,
    }
    plain_success = {
        "strict_all": True,
        "original_pass_rate": 1.0,
        "pass_rate": 1.0,
        "clip_aesthetic": 1.0,
    }

    assert best_candidate([beautiful_failure, plain_success]) is plain_success
    assert deliverable_candidate([beautiful_failure]) is None
    assert deliverable_candidate([beautiful_failure, plain_success]) is plain_success


def _stage2(
    method_id: str,
    tolerances: tuple[float, ...],
    *,
    successes: tuple[float, ...] | None = None,
    saturation: float = 0.01,
) -> dict:
    successes = successes or tuple(1.0 for _ in tolerances)
    raster_hash = hashlib.sha256(method_id.encode("utf-8")).hexdigest()
    return {
        "method_id": method_id,
        "generation_run_id": f"run-{method_id}",
        "status": "accepted",
        "pipeline_state": "stage2",
        "output_variant": "srpg",
        "final_image_sha256": raster_hash,
        "saturation_risk": saturation,
        "qr_verify_observations": [
            {
                "qr_success": success,
                "qr_tolerance": tolerance,
                "image_sha256": raster_hash,
            }
            for success, tolerance in zip(successes, tolerances, strict=True)
        ],
    }


def test_conservative_qr_score_uses_every_payload_result_and_worst_tolerance():
    candidate = _stage2(
        "primary",
        (0.96, 0.82, 0.91),
        successes=(1.0, 0.0, 1.0),
    )

    score = conservative_qr_score(candidate)

    assert score.observation_count == 3
    assert score.minimum_tolerance == pytest.approx(0.82)
    assert score.mean_tolerance == pytest.approx((0.96 + 0.82 + 0.91) / 3)
    assert score.payload_exact_all is False
    assert score.raster_hash_consistent is True


def test_conservative_gate_rejects_an_optimistic_run_and_saturation():
    candidate = _stage2("primary", (0.95, 0.79, 0.92), saturation=0.06)

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert "conservative_qr_tolerance_below_threshold" in assessment.rejection_reasons
    assert "saturation_above_threshold" in assessment.rejection_reasons


def test_conservative_gate_prefers_preset_intersection_over_each_run_minimum():
    candidate = _stage2("primary", (0.92, 0.91, 0.94))
    candidate["conservative_qr_tolerance"] = 0.78

    assessment = assess_stage2_candidate(candidate)

    assert assessment.qr.minimum_tolerance == pytest.approx(0.78)
    assert assessment.deliverable is False
    assert "conservative_qr_tolerance_below_threshold" in assessment.rejection_reasons


def test_conservative_gate_rejects_measurements_from_another_raster():
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["qr_verify_observations"][1]["image_sha256"] = "another-raster"

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.qr.raster_hash_consistent is False
    assert "qr_observations_do_not_share_the_selected_raster" in assessment.rejection_reasons


def test_conservative_gate_requires_a_sha256_for_every_observation():
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["qr_verify_observations"][1].pop("image_sha256")

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.qr.raster_hash_consistent is False
    assert "qr_observations_do_not_share_the_selected_raster" in assessment.rejection_reasons


def test_conservative_gate_rejects_non_binary_payload_marker():
    candidate = _stage2(
        "primary",
        (0.95, 0.92, 0.91),
        successes=(1.0, 0.9, 1.0),
    )

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.qr.payload_exact_all is False
    assert "payload_not_exact_on_every_observation" in assessment.rejection_reasons


@pytest.mark.parametrize("status", [None, "", "mystery", "failed", "timeout"])
def test_conservative_gate_rejects_unknown_or_nonterminal_status(status):
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["status"] = status

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.generated is False
    assert "generation_failed" in assessment.rejection_reasons


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_conservative_gate_accepts_e029_terminal_generation_statuses(status):
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["status"] = status

    assert assess_stage2_candidate(candidate).deliverable is True


@pytest.mark.parametrize("invalid", [True, False, -1, 1.01, 100, "true", "nan"])
def test_conservative_gate_rejects_invalid_qr_tolerance(invalid):
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["qr_verify_observations"][1]["qr_tolerance"] = invalid

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.qr.minimum_tolerance is None
    assert "missing_qr_tolerance" in assessment.rejection_reasons


@pytest.mark.parametrize("invalid", [True, False, -1, 1.01, 100, "true", "nan"])
def test_conservative_gate_rejects_invalid_saturation_risk(invalid):
    candidate = _stage2("primary", (0.95, 0.92, 0.91))
    candidate["saturation_risk"] = invalid

    assessment = assess_stage2_candidate(candidate)

    assert assessment.deliverable is False
    assert assessment.maximum_saturation_risk is None
    assert "missing_saturation_risk" in assessment.rejection_reasons


def test_stage2_cascade_stops_after_passing_fixed_recipe():
    primary = _stage2("fixed", (0.88, 0.87, 0.89))
    visually_better_alternate = _stage2("advisor", (1.0, 1.0, 1.0))
    visually_better_alternate["clip_aesthetic"] = 9.0

    decision = select_stage2_cascade(primary, visually_better_alternate)

    assert decision.next_action == "deliver_primary_stage2"
    assert decision.selected_candidate is primary
    assert decision.alternate is None
    assert decision.stage1_was_delivered is False
    assert decision.srmpgd_was_requested is False


def test_stage2_cascade_uses_alternate_only_after_primary_fails():
    primary = _stage2("fixed", (0.79, 0.95, 0.90))
    alternate = _stage2("advisor", (0.84, 0.82, 0.88))

    request = select_stage2_cascade(primary)
    decision = select_stage2_cascade(primary, alternate)

    assert request.next_action == "generate_alternate_stage2"
    assert request.selected_candidate is None
    assert request.as_dict()["primary_rejection_reasons"] == [
        "conservative_qr_tolerance_below_threshold"
    ]
    assert decision.next_action == "deliver_alternate_stage2"
    assert decision.selected_candidate is alternate
    assert decision.as_dict()["conservative_qr_tolerance"] == pytest.approx(0.82)


@pytest.mark.parametrize(
    ("pipeline_state", "output_variant"),
    [("stage1", "raw"), ("srmpgd", "srmpgd")],
)
def test_stage2_cascade_never_delivers_stage1_or_srmpgd(
    pipeline_state: str,
    output_variant: str,
):
    forbidden = _stage2("forbidden", (1.0, 1.0, 1.0))
    forbidden.update({"pipeline_state": pipeline_state, "output_variant": output_variant})
    alternate = _stage2("alternate", (0.91, 0.89, 0.90))

    decision = select_stage2_cascade(forbidden, alternate)

    assert decision.next_action == "deliver_alternate_stage2"
    assert decision.selected_candidate is alternate
    assert "not_a_stage2_srpg_raster" in decision.primary.rejection_reasons
    assert decision.stage1_was_delivered is False
    assert decision.srmpgd_was_requested is False


def test_stage2_cascade_requires_repeated_observations_by_default():
    raster_hash = hashlib.sha256(b"historical-e029").hexdigest()
    historical_e029_row = {
        "status": "accepted",
        "pipeline_state": "stage2",
        "output_variant": "srpg",
        "qr_success": 1.0,
        "qr_tolerance": 1.0,
        "saturation_risk": 0.0,
        "final_image_sha256": raster_hash,
    }

    conservative = select_stage2_cascade(historical_e029_row)
    replay = select_stage2_cascade(
        historical_e029_row,
        gate=ConservativeDeliveryGate(minimum_qr_observations=1),
    )

    assert conservative.next_action == "generate_alternate_stage2"
    assert "insufficient_qr_observations" in conservative.primary.rejection_reasons
    assert replay.next_action == "deliver_primary_stage2"


def test_stage2_cascade_rejects_both_failures_without_srmpgd_fallback():
    primary = _stage2("fixed", (0.2, 0.3, 0.4))
    alternate = _stage2("advisor", (0.7, 0.6, 0.79))

    decision = select_stage2_cascade(primary, alternate)

    assert decision.next_action == "reject"
    assert decision.deliverable is False
    assert decision.selected_candidate is None
    assert decision.srmpgd_was_requested is False
