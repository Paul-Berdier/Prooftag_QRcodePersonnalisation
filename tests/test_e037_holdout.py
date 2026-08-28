from __future__ import annotations

from prooftag_qr.e036_trust_region import BRANCH_GLOBAL
from prooftag_qr.e037_holdout import (
    CASE_COUNT,
    FROZEN_CONFIG,
    GLOBAL_POLICY,
    HOLDOUT_CASES,
    PAYLOAD,
    _assert_frozen_protocol,
)


def test_e037_protocol_is_frozen() -> None:
    _assert_frozen_protocol()
    assert CASE_COUNT == 10
    assert len(HOLDOUT_CASES) == 10
    assert len({case.case_id for case in HOLDOUT_CASES}) == 10
    assert len({case.seed for case in HOLDOUT_CASES}) == 10
    assert PAYLOAD == "https://ptag.io/t/e037"
    assert GLOBAL_POLICY.name == BRANCH_GLOBAL
    assert GLOBAL_POLICY.latent_radius_rms == 0.05
    assert GLOBAL_POLICY.lpips_budget == 0.05
    assert GLOBAL_POLICY.core_mae_budget == 0.05
    assert FROZEN_CONFIG.gamma == 1000.0
    assert FROZEN_CONFIG.max_iterations == 4
    assert FROZEN_CONFIG.lpips_weight == 0.01


def test_e037_cases_cover_distinct_visual_domains() -> None:
    text = " ".join(case.prompt.lower() for case in HOLDOUT_CASES)
    for token in (
        "courtyard",
        "railway",
        "wine cellar",
        "alpine",
        "ramen",
        "botanical",
        "lighthouse",
        "workshop",
        "paris cafe",
        "vineyard",
    ):
        assert token in text
