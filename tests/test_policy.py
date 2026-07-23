import pytest

from prooftag_qr.policy import (
    attempts_for_target,
    best_candidate,
    deliverable_candidate,
    delivery_probability,
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
