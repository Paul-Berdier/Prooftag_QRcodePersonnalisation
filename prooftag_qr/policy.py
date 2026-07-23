from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def delivery_probability(single_attempt_probability: float, attempts: int) -> float:
    """Probability that at least one attempt succeeds under an independence approximation."""
    if not 0 <= single_attempt_probability <= 1:
        raise ValueError("single_attempt_probability must be between zero and one")
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    return 1 - (1 - single_attempt_probability) ** attempts


def attempts_for_target(
    single_attempt_probability: float,
    target_probability: float = 0.999,
) -> int | None:
    """Return the minimum candidate budget, or ``None`` when success is impossible."""
    if not 0 <= single_attempt_probability <= 1:
        raise ValueError("single_attempt_probability must be between zero and one")
    if not 0 < target_probability < 1:
        raise ValueError("target_probability must be strictly between zero and one")
    if single_attempt_probability == 0:
        return None
    if single_attempt_probability == 1:
        return 1
    return math.ceil(math.log1p(-target_probability) / math.log1p(-single_attempt_probability))


def candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Lexicographic Prooftag objective: scan first, aesthetics only after the gate."""
    return (
        bool(candidate.get("strict_all", False)),
        float(candidate.get("original_pass_rate", 0.0)),
        float(candidate.get("pass_rate", 0.0)),
        float(candidate.get("worst_decoder_pass_rate", 0.0)),
        float(candidate.get("worst_scenario_pass_rate", 0.0)),
        _optional_score(candidate.get("clip_aesthetic")),
        _optional_score(candidate.get("clip_score")),
        -float(candidate.get("duration_s", float("inf"))),
    )


def best_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return max(candidates, key=candidate_rank) if candidates else None


def deliverable_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Never label an unvalidated image as deliverable."""
    passing = [candidate for candidate in candidates if candidate.get("strict_all", False)]
    return best_candidate(passing)


def _optional_score(value: Any) -> float:
    return float(value) if value is not None else float("-inf")
