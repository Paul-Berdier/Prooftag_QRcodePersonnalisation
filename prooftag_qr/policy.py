from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_GENERATED_STATUSES = frozenset({"accepted", "rejected"})
_STAGE2_OUTPUTS = frozenset({"srpg"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ConservativeDeliveryGate:
    """Production gate for repeated QR-Verify measurements of one raster.

    A delivery passes only when every QR observation decodes the expected payload,
    the worst measured QR tolerance reaches the threshold and the worst saturation
    measurement remains below its guard.  Three observations are required by
    default so a single optimistic QR-Verify run cannot release an image.
    """

    qr_tolerance_threshold: float = 0.80
    saturation_threshold: float = 0.05
    minimum_qr_observations: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.qr_tolerance_threshold <= 1.0:
            raise ValueError("qr_tolerance_threshold must be between zero and one")
        if not 0.0 <= self.saturation_threshold <= 1.0:
            raise ValueError("saturation_threshold must be between zero and one")
        if self.minimum_qr_observations < 1:
            raise ValueError("minimum_qr_observations must be at least one")


@dataclass(frozen=True, slots=True)
class ConservativeQRScore:
    observation_count: int
    payload_exact_all: bool
    minimum_tolerance: float | None
    mean_tolerance: float | None
    maximum_tolerance: float | None
    raster_hash_consistent: bool | None


@dataclass(frozen=True, slots=True)
class Stage2CandidateAssessment:
    generated: bool
    stage2_contract_valid: bool
    qr: ConservativeQRScore
    maximum_saturation_risk: float | None
    deliverable: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Stage2CascadeDecision:
    """Decision of the fixed Stage 2 -> alternate Stage 2 cascade.

    ``next_action`` is intentionally limited to Stage 2 operations.  SR-MPGD is
    never scheduled by this policy and Stage 1 is never a delivery candidate.
    """

    next_action: str
    selected_role: str | None
    selected_candidate: Mapping[str, Any] | None
    primary: Stage2CandidateAssessment | None
    alternate: Stage2CandidateAssessment | None
    deliverable: bool
    stage1_was_delivered: bool = False
    srmpgd_was_requested: bool = False

    def as_dict(self) -> dict[str, Any]:
        selected = self.selected_candidate or {}
        if self.selected_role == "primary_stage2":
            assessment = self.primary
        elif self.selected_role == "alternate_stage2":
            assessment = self.alternate
        else:
            assessment = self.alternate or self.primary
        qr = assessment.qr if assessment is not None else None
        return {
            "next_action": self.next_action,
            "selected_role": self.selected_role,
            "selected_method_id": selected.get("method_id"),
            "selected_generation_run_id": selected.get("generation_run_id"),
            "deliverable": self.deliverable,
            "conservative_qr_success": bool(qr and qr.payload_exact_all),
            "conservative_qr_tolerance": qr.minimum_tolerance if qr else None,
            "qr_observation_count": qr.observation_count if qr else 0,
            "maximum_saturation_risk": (assessment.maximum_saturation_risk if assessment else None),
            "rejection_reasons": list(assessment.rejection_reasons) if assessment else [],
            "primary_rejection_reasons": (
                list(self.primary.rejection_reasons) if self.primary else []
            ),
            "alternate_rejection_reasons": (
                list(self.alternate.rejection_reasons) if self.alternate else []
            ),
            "stage1_was_delivered": self.stage1_was_delivered,
            "srmpgd_was_requested": self.srmpgd_was_requested,
        }


def _finite(value: Any) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or isinstance(value, (Mapping, Sequence))
        and not isinstance(value, (str, bytes))
    ):
        return None
    normalized = str(value).strip().casefold()
    if not normalized:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unit_interval(value: Any) -> float | None:
    result = _finite(value)
    if result is None or not 0.0 <= result <= 1.0:
        return None
    return result


def _binary_indicator(value: Any) -> float | None:
    result = _unit_interval(value)
    return result if result in {0.0, 1.0} else None


def _sha256(value: Any) -> str | None:
    normalized = str(value or "").strip().casefold()
    return normalized if _SHA256_PATTERN.fullmatch(normalized) else None


def _observation_value(observation: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in observation:
            return observation[name]
    return None


def _qr_observations(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = candidate.get("qr_verify_observations")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [dict(item) for item in raw if isinstance(item, Mapping)]

    success = _binary_indicator(candidate.get("qr_success"))
    tolerance = _unit_interval(candidate.get("qr_tolerance"))
    if success is None and tolerance is None:
        return []
    return [
        {
            "qr_success": success,
            "qr_tolerance": tolerance,
            "image_sha256": candidate.get("final_image_sha256"),
            "saturation_risk": candidate.get("saturation_risk"),
        }
    ]


def conservative_qr_score(candidate: Mapping[str, Any]) -> ConservativeQRScore:
    """Reduce repeated QR-Verify observations to their worst trustworthy result.

    ``qr_verify_observations`` is the preferred input.  Each item may expose
    ``qr_success``/``payload_exact`` and ``qr_tolerance``/``tolerance``.  The
    scalar E029 fields remain supported as one historical observation.
    """

    observations = _qr_observations(candidate)
    successes: list[bool] = []
    tolerances: list[float] = []
    hashes: list[str] = []
    for observation in observations:
        success = _binary_indicator(
            _observation_value(
                observation,
                ("qr_success", "payload_exact", "exact_payload_match"),
            )
        )
        tolerance = _unit_interval(_observation_value(observation, ("qr_tolerance", "tolerance")))
        successes.append(success == 1.0)
        if tolerance is not None:
            tolerances.append(tolerance)
        image_hash = _sha256(observation.get("image_sha256"))
        if image_hash:
            hashes.append(image_hash)

    expected_hash = _sha256(candidate.get("final_image_sha256"))
    raster_hash_consistent = (
        bool(observations)
        and bool(expected_hash)
        and (
            len(hashes) == len(observations)
            and len(set(hashes)) == 1
            and hashes[0] == expected_hash
        )
    )

    explicit_present = (
        "conservative_qr_tolerance" in candidate or "qr_verify_conservative_tolerance" in candidate
    )
    explicit_conservative = _unit_interval(
        candidate.get("conservative_qr_tolerance")
        if "conservative_qr_tolerance" in candidate
        else candidate.get("qr_verify_conservative_tolerance")
    )
    minimum_candidates = list(tolerances)
    if explicit_conservative is not None:
        # An intersection score can be lower than every individual run when
        # different presets succeed opportunistically.  It is the stricter
        # value and must participate in the delivery gate.
        minimum_candidates.append(explicit_conservative)

    return ConservativeQRScore(
        observation_count=len(observations),
        payload_exact_all=bool(observations)
        and len(successes) == len(observations)
        and all(successes),
        minimum_tolerance=(
            min(minimum_candidates)
            if len(tolerances) == len(observations)
            and (not explicit_present or explicit_conservative is not None)
            and minimum_candidates
            else None
        ),
        mean_tolerance=(sum(tolerances) / len(tolerances)) if tolerances else None,
        maximum_tolerance=max(tolerances) if tolerances else None,
        raster_hash_consistent=raster_hash_consistent,
    )


def _maximum_saturation_risk(candidate: Mapping[str, Any]) -> float | None:
    raw_values = []
    if "saturation_risk" in candidate:
        raw_values.append(candidate.get("saturation_risk"))
    for observation in _qr_observations(candidate):
        if "saturation_risk" in observation:
            raw_values.append(observation.get("saturation_risk"))
    if not raw_values:
        return None
    values = [_unit_interval(value) for value in raw_values]
    if any(value is None for value in values):
        return None
    return max(values) if values else None


def assess_stage2_candidate(
    candidate: Mapping[str, Any],
    gate: ConservativeDeliveryGate | None = None,
) -> Stage2CandidateAssessment:
    """Assess one Stage 2 raster without importing or rerunning a decoder."""

    gate = gate or ConservativeDeliveryGate()
    status = str(candidate.get("status") or "").casefold()
    generated = status in _GENERATED_STATUSES
    pipeline_state = str(candidate.get("pipeline_state") or "").casefold()
    output_variant = str(candidate.get("output_variant") or "").casefold()
    contract_valid = pipeline_state == "stage2" and output_variant in _STAGE2_OUTPUTS
    qr = conservative_qr_score(candidate)
    saturation = _maximum_saturation_risk(candidate)

    reasons = []
    if not generated:
        reasons.append("generation_failed")
    if not contract_valid:
        reasons.append("not_a_stage2_srpg_raster")
    if qr.observation_count < gate.minimum_qr_observations:
        reasons.append("insufficient_qr_observations")
    if not qr.payload_exact_all:
        reasons.append("payload_not_exact_on_every_observation")
    if qr.minimum_tolerance is None:
        reasons.append("missing_qr_tolerance")
    elif qr.minimum_tolerance < gate.qr_tolerance_threshold:
        reasons.append("conservative_qr_tolerance_below_threshold")
    if qr.raster_hash_consistent is not True:
        reasons.append("qr_observations_do_not_share_the_selected_raster")
    if saturation is None:
        reasons.append("missing_saturation_risk")
    elif saturation > gate.saturation_threshold:
        reasons.append("saturation_above_threshold")

    return Stage2CandidateAssessment(
        generated=generated,
        stage2_contract_valid=contract_valid,
        qr=qr,
        maximum_saturation_risk=saturation,
        deliverable=not reasons,
        rejection_reasons=tuple(reasons),
    )


def select_stage2_cascade(
    primary_stage2: Mapping[str, Any] | None,
    alternate_stage2: Mapping[str, Any] | None = None,
    *,
    gate: ConservativeDeliveryGate | None = None,
) -> Stage2CascadeDecision:
    """Run the fixed Stage 2 -> alternate Stage 2 delivery state machine.

    The primary recipe wins immediately when it passes.  The alternate recipe is
    requested only after the primary fails.  Failure of both candidates is a hard
    rejection; this policy deliberately never falls back to Stage 1 or SR-MPGD.
    """

    gate = gate or ConservativeDeliveryGate()
    if primary_stage2 is None:
        return Stage2CascadeDecision(
            next_action="generate_primary_stage2",
            selected_role=None,
            selected_candidate=None,
            primary=None,
            alternate=None,
            deliverable=False,
        )

    primary = assess_stage2_candidate(primary_stage2, gate)
    if primary.deliverable:
        return Stage2CascadeDecision(
            next_action="deliver_primary_stage2",
            selected_role="primary_stage2",
            selected_candidate=primary_stage2,
            primary=primary,
            alternate=None,
            deliverable=True,
        )

    if alternate_stage2 is None:
        return Stage2CascadeDecision(
            next_action="generate_alternate_stage2",
            selected_role=None,
            selected_candidate=None,
            primary=primary,
            alternate=None,
            deliverable=False,
        )

    alternate = assess_stage2_candidate(alternate_stage2, gate)
    if alternate.deliverable:
        return Stage2CascadeDecision(
            next_action="deliver_alternate_stage2",
            selected_role="alternate_stage2",
            selected_candidate=alternate_stage2,
            primary=primary,
            alternate=alternate,
            deliverable=True,
        )
    return Stage2CascadeDecision(
        next_action="reject",
        selected_role=None,
        selected_candidate=None,
        primary=primary,
        alternate=alternate,
        deliverable=False,
    )


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
