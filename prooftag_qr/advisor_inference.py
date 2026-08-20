from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .parameter_advisor import E026ParameterAdvisor, RecipeCandidate
from .qr import generate_diffqrcoder_qr
from .schemas import LabCampaignCreate, LabMethod, LabPrompt

TERMINAL_CAMPAIGN_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "interrupted",
}
SUCCESSFUL_CAMPAIGN_STATUSES = {"completed", "completed_with_errors"}
E026J_SELECTION_PROFILES = (
    "robust",
    "balanced",
    "aesthetic_scannable",
    "balanced_exploratory",
    "aesthetic_exploratory",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_identifier(value: Any, *, maximum: int = 70) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return (result or "method")[:maximum]


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "oui"}:
        return 1.0
    if lowered in {"false", "no", "non"}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_finite(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = _finite(row.get(name))
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class AdvisorInferencePlan:
    plan_id: str
    payload: str = field(repr=False)
    campaigns: tuple[dict[str, Any], ...]
    predictions: tuple[dict[str, Any], ...]
    public: dict[str, Any]


def _runtime_method(
    candidate: RecipeCandidate,
    *,
    runtime_id: str,
    display_name: str,
    adaptive_srmpgd: bool = False,
) -> dict[str, Any]:
    configuration = deepcopy(candidate.configuration)
    if "legacy_requested_parameters" in configuration:
        raise ValueError(
            f"candidate {candidate.method_id} is legacy-only and cannot be generated"
        )
    configuration.update(
        {
            "id": runtime_id,
            "name": display_name[:200],
            "enabled": True,
        }
    )
    if adaptive_srmpgd and (configuration.get("tools") or {}).get("srmpgd_enabled"):
        # ``auto`` lets the service retain the paired SRPG state whenever
        # SR-MPGD selects iteration zero.  A no-op refinement must not be
        # advertised as a new SR-MPGD image.
        configuration["output_variant"] = "auto"
        tools = deepcopy(configuration.get("tools") or {})
        settings = deepcopy(tools.get("settings") or {})
        settings["srmpgd_min_qr_tolerance"] = max(
            float(settings.get("srmpgd_min_qr_tolerance", 0.0)),
            0.80,
        )
        tools["settings"] = settings
        configuration["tools"] = tools
    return LabMethod.model_validate(configuration).model_dump(mode="json")


def effective_candidate_signature(candidate: RecipeCandidate) -> str:
    """Hash the generation state that exists *before* optional SR-MPGD.

    Gamma, LPIPS weight and iteration count cannot make three independent
    candidates when SR-MPGD keeps iteration zero: all of them return the same
    paired Stage-2 image.  Collapsing those settings here prevents the advisor
    from filling its top-K with aliases of one effective generation recipe.
    """

    configuration = deepcopy(candidate.configuration)
    for key in ("id", "name", "description", "enabled"):
        configuration.pop(key, None)
    tools = deepcopy(configuration.get("tools") or {})
    if tools.get("srmpgd_enabled"):
        tools["srmpgd_enabled"] = False
        tools["srpg_enabled"] = True
        tools["settings"] = {
            key: value
            for key, value in (tools.get("settings") or {}).items()
            if not str(key).startswith("srmpgd_")
        }
        configuration["tools"] = tools
        configuration["output_variant"] = "srpg"
    return hashlib.sha256(
        _canonical_json(configuration).encode("utf-8")
    ).hexdigest()


def _normalized_scores(
    recommendations: Sequence[Any],
    getter: Callable[[Any], float | None],
    *,
    lower_is_better: bool = False,
) -> dict[str, float]:
    available = [
        (item.candidate.signature, value)
        for item in recommendations
        if (value := getter(item)) is not None and math.isfinite(float(value))
    ]
    if not available:
        return {item.candidate.signature: 0.5 for item in recommendations}
    values = [float(value) for _, value in available]
    low, high = min(values), max(values)
    result = {}
    for signature, value in available:
        normalized = 0.5 if high == low else (float(value) - low) / (high - low)
        result[signature] = 1.0 - normalized if lower_is_better else normalized
    return {
        item.candidate.signature: result.get(item.candidate.signature, 0.0)
        for item in recommendations
    }


def _select_diversified_recommendations(
    ranked: Sequence[Any],
    *,
    top_k: int,
    excluded_method_ids: Sequence[str] = (),
) -> list[tuple[str, Any, str]]:
    """Select effective and objective-diverse recommendations.

    A scan-safe recommendation is mandatory and always occupies the robust slot.
    When the advisor exposes fewer than ``top_k`` distinct scan-safe recipes, the
    missing slots are filled with distinct exploratory recipes.  Their original
    ``scan_safe=False`` prediction is preserved so that neither the notebook nor
    downstream reports can present them as validated before QR-Verify runs.
    """

    excluded = set(excluded_method_ids)
    distinct: list[tuple[Any, str]] = []
    effective_seen: set[str] = set()
    for recommendation in ranked:
        if recommendation.candidate.method_id in excluded:
            continue
        effective = effective_candidate_signature(recommendation.candidate)
        if effective in effective_seen:
            continue
        effective_seen.add(effective)
        distinct.append((recommendation, effective))
    if len(distinct) < top_k:
        raise RuntimeError(
            f"only {len(distinct)} distinct effective generation recipes "
            f"for top_k={top_k}"
        )
    scan_safe = [item for item, _ in distinct if item.scan_safe]
    if not scan_safe:
        raise RuntimeError(
            "no distinct scan-safe generation recipe is available; "
            "refusing to build an inference plan without a robust anchor"
        )

    recommendations = [item for item, _ in distinct]
    tolerance = _normalized_scores(
        recommendations, lambda item: item.predicted_qr_tolerance
    )
    hps = _normalized_scores(recommendations, lambda item: item.predicted_hpsv2_1)
    aesthetic = _normalized_scores(
        recommendations, lambda item: item.predicted_clip_aesthetic
    )
    clip = _normalized_scores(recommendations, lambda item: item.predicted_clip_score)
    saturation = _normalized_scores(
        recommendations,
        lambda item: item.predicted_saturation_risk,
        lower_is_better=True,
    )

    def visual_score(item: Any) -> float:
        signature = item.candidate.signature
        return (
            0.50 * hps[signature]
            + 0.30 * aesthetic[signature]
            + 0.20 * clip[signature]
        )

    def qr_score(item: Any) -> float:
        signature = item.candidate.signature
        return (
            0.55 * float(item.qr_success_lower_bound)
            + 0.25 * float(item.predicted_qr_success)
            + 0.15 * tolerance[signature]
            + 0.05 * saturation[signature]
        )

    robust = max(
        scan_safe,
        key=lambda item: (
            qr_score(item),
            item.qr_success_lower_bound,
            item.predicted_qr_success,
            visual_score(item),
        ),
    )
    if top_k == 1:
        return [("robust", robust, effective_candidate_signature(robust.candidate))]
    remaining_safe = [item for item in scan_safe if item is not robust]
    exploratory = [item for item in recommendations if not item.scan_safe]

    def aesthetic_key(item: Any) -> tuple[float, float, float]:
        return (
            visual_score(item),
            qr_score(item),
            -(item.predicted_saturation_risk or 0.0),
        )

    def balanced_key(item: Any) -> tuple[float, float, float]:
        return (
            0.55 * qr_score(item) + 0.45 * visual_score(item),
            qr_score(item),
            visual_score(item),
        )

    if top_k == 2:
        if remaining_safe:
            second_profile = "aesthetic_scannable"
            second = max(remaining_safe, key=aesthetic_key)
        else:
            second_profile = "aesthetic_exploratory"
            second = max(exploratory, key=aesthetic_key)
        return [
            ("robust", robust, effective_candidate_signature(robust.candidate)),
            (
                second_profile,
                second,
                effective_candidate_signature(second.candidate),
            ),
        ]

    selected: list[tuple[str, Any]] = [("robust", robust)]
    if len(remaining_safe) >= 2:
        aesthetic_pick = max(remaining_safe, key=aesthetic_key)
        balanced_pool = [item for item in remaining_safe if item is not aesthetic_pick]
        balanced = max(balanced_pool, key=balanced_key)
        selected.extend(
            [
                ("balanced", balanced),
                ("aesthetic_scannable", aesthetic_pick),
            ]
        )
    elif len(remaining_safe) == 1:
        selected.append(("balanced", remaining_safe[0]))
        selected.append(
            ("aesthetic_exploratory", max(exploratory, key=aesthetic_key))
        )
    else:
        balanced = max(exploratory, key=balanced_key)
        selected.append(("balanced_exploratory", balanced))
        aesthetic_pool = [item for item in exploratory if item is not balanced]
        selected.append(
            ("aesthetic_exploratory", max(aesthetic_pool, key=aesthetic_key))
        )

    selected = selected[:top_k]
    used = {item.candidate.signature for _, item in selected}
    if top_k > len(selected):
        remaining = [
            item for item in recommendations if item.candidate.signature not in used
        ]
        remaining.sort(
            key=lambda item: (item.scan_safe, balanced_key(item)), reverse=True
        )
        for item in remaining:
            if item.candidate.signature in used:
                continue
            profile = (
                f"alternate_scan_safe_{len(selected) + 1}"
                if item.scan_safe
                else f"exploratory_{len(selected) + 1}"
            )
            selected.append((profile, item))
            used.add(item.candidate.signature)
            if len(selected) == top_k:
                break
    effective_by_signature = {
        item.candidate.signature: effective for item, effective in distinct
    }
    return [
        (profile, item, effective_by_signature[item.candidate.signature])
        for profile, item in selected
    ]


def _srpg_prerequisite(
    candidate: RecipeCandidate,
    *,
    runtime_id: str,
    display_name: str,
) -> tuple[dict[str, Any], str] | None:
    """Build the exact Stage 2 source required by an SR-MPGD candidate.

    The laboratory deliberately refuses to run SR-MPGD without an earlier,
    mathematically matching SRPG state in the same campaign.  Advisor inference
    therefore materializes that dependency explicitly instead of silently
    treating the failed SR-MPGD trial as a QR failure.
    """

    configuration = deepcopy(candidate.configuration)
    tools = configuration.get("tools") or {}
    if not tools.get("srmpgd_enabled"):
        return None
    if not tools.get("srpg_enabled"):
        raise ValueError(
            f"candidate {candidate.method_id} enables SR-MPGD without Stage 2 SRPG"
        )
    settings = {
        key: value
        for key, value in (tools.get("settings") or {}).items()
        if not str(key).startswith("srmpgd_")
    }
    configuration.update(
        {
            "id": runtime_id,
            "name": display_name[:200],
            "enabled": True,
            "output_variant": "srpg",
        }
    )
    configuration["tools"] = {
        **tools,
        "srpg_enabled": True,
        "srmpgd_enabled": False,
        "settings": settings,
    }
    signature = effective_candidate_signature(candidate)
    method = LabMethod.model_validate(configuration).model_dump(mode="json")
    return method, signature


def build_advisor_inference_plan(
    *,
    advisor: E026ParameterAdvisor,
    candidates: Sequence[RecipeCandidate],
    prompts: Sequence[Mapping[str, Any]],
    payload: str,
    advisor_sha256: str,
    prompt_embedding_provider: Callable[[str], Sequence[float]] | None = None,
    seen_prompt_texts: Sequence[str] = (),
    seeds: Sequence[int] = (413_001, 523_001, 631_001),
    top_k: int = 3,
    baseline_method_id: str | None = "diffqrcoder_stage1",
    scan_probability_threshold: float = 0.80,
    error_correction: str = "M",
    qr_context: Mapping[str, Any] | None = None,
) -> AdvisorInferencePlan:
    """Create deterministic per-prompt campaigns from E026 recommendations.

    The clear payload is kept only in the returned in-memory plan. The public plan persisted
    by the runner contains its hash and length, never the payload itself.
    """

    if not candidates:
        raise ValueError("at least one historical recipe candidate is required")
    if not prompts:
        raise ValueError("at least one unseen inference prompt is required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not advisor_sha256:
        raise ValueError("advisor_sha256 is required for a traceable plan")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("inference seeds must be non-empty and unique")

    context = {
        "qr_version": 3,
        "qr_mask_pattern": 4,
        "qr_module_size": 20,
        "qr_padding_px": 78,
        **dict(qr_context or {}),
    }
    generate_diffqrcoder_qr(
        payload,
        error_correction=error_correction,
        version=int(context["qr_version"]),
        mask_pattern=int(context["qr_mask_pattern"]),
        module_size=int(context["qr_module_size"]),
        border=4,
    )

    validated_prompts = [LabPrompt.model_validate(dict(prompt)) for prompt in prompts]
    if len({item.id for item in validated_prompts}) != len(validated_prompts):
        raise ValueError("inference prompt ids must be unique")
    seen = {_normalized_prompt(value) for value in seen_prompt_texts}
    overlap = [item.id for item in validated_prompts if _normalized_prompt(item.text) in seen]
    if overlap:
        raise ValueError(f"inference prompts already occur in training data: {overlap}")

    baseline = None
    if baseline_method_id:
        matching = [item for item in candidates if item.method_id == baseline_method_id]
        if not matching:
            raise ValueError(f"baseline candidate is absent: {baseline_method_id}")
        baseline = max(matching, key=lambda item: item.observations)

    drafts: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for prompt in validated_prompts:
        embedding = prompt_embedding_provider(prompt.text) if prompt_embedding_provider else None
        ranked = advisor.recommend(
            prompt=prompt.text,
            prompt_embedding=embedding,
            payload_length=len(payload),
            error_correction=error_correction,
            qr_context=context,
            candidates=candidates,
            scan_probability_threshold=scan_probability_threshold,
            limit=len(candidates),
        )
        selected = _select_diversified_recommendations(
            ranked,
            top_k=top_k,
            excluded_method_ids=(baseline_method_id,) if baseline_method_id else (),
        )

        methods = []
        prerequisite_signatures: set[str] = set()
        selected_signatures = set()
        for selection_rank, (
            selection_profile,
            recommendation,
            effective_signature,
        ) in enumerate(selected, start=1):
            candidate = recommendation.candidate
            selected_signatures.add(candidate.signature)
            prerequisite_id = (
                f"e026j_dep_{selection_rank:02d}_"
                f"{_safe_identifier(candidate.method_id, maximum=52)}"
            )[:100]
            prerequisite = _srpg_prerequisite(
                candidate,
                runtime_id=prerequisite_id,
                display_name=(
                    f"E026 paired SRPG prerequisite | {candidate.method_id}"
                ),
            )
            if prerequisite is not None:
                prerequisite_method, prerequisite_signature = prerequisite
                if prerequisite_signature not in prerequisite_signatures:
                    methods.append(prerequisite_method)
                    prerequisite_signatures.add(prerequisite_signature)
                    prediction_rows.append(
                        {
                            "prompt_id": prompt.id,
                            "prompt_text": prompt.text,
                            "plan_method_id": prerequisite_method["id"],
                            "source_method_id": candidate.method_id,
                            "role": "srmpgd_prerequisite",
                            "advisor_rank": selection_rank,
                            "model_rank": recommendation.rank,
                            "selection_profile": f"{selection_profile}_prerequisite",
                            "effective_candidate_signature": effective_signature,
                            "candidate_signature": (
                                f"srpg-prerequisite:{prerequisite_signature}"
                            ),
                            "candidate_observations": candidate.observations,
                            "requested_source_output_variant": candidate.configuration.get(
                                "output_variant"
                            ),
                            "runtime_output_variant": "srpg",
                            "scan_safe": None,
                            "predicted_qr_success": None,
                            "predicted_qr_success_lower_bound": None,
                            "predicted_qr_success_uncertainty": None,
                            "predicted_qr_tolerance": None,
                            "predicted_clip_aesthetic": None,
                            "predicted_clip_score": None,
                            "predicted_hpsv2_1": None,
                            "predicted_saturation_risk": None,
                            "predicted_duration_ms": None,
                        }
                    )
            runtime_id = (
                f"e026j_r{selection_rank:02d}_"
                f"{_safe_identifier(candidate.method_id)}"
            )[:100]
            methods.append(
                _runtime_method(
                    candidate,
                    runtime_id=runtime_id,
                    display_name=(
                        f"E026J {selection_profile} | {candidate.method_id}"
                    ),
                    adaptive_srmpgd=True,
                )
            )
            prediction_rows.append(
                {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.text,
                    "plan_method_id": runtime_id,
                    "source_method_id": candidate.method_id,
                    "role": "advisor_recommendation",
                    "advisor_rank": selection_rank,
                    "model_rank": recommendation.rank,
                    "selection_profile": selection_profile,
                    "effective_candidate_signature": effective_signature,
                    "candidate_signature": candidate.signature,
                    "candidate_observations": candidate.observations,
                    "requested_source_output_variant": candidate.configuration.get(
                        "output_variant"
                    ),
                    "runtime_output_variant": (
                        "auto"
                        if (candidate.configuration.get("tools") or {}).get(
                            "srmpgd_enabled"
                        )
                        else candidate.configuration.get("output_variant")
                    ),
                    "scan_safe": recommendation.scan_safe,
                    "predicted_qr_success": recommendation.predicted_qr_success,
                    "predicted_qr_success_lower_bound": (
                        recommendation.qr_success_lower_bound
                    ),
                    "predicted_qr_success_uncertainty": (
                        recommendation.qr_success_uncertainty
                    ),
                    "predicted_qr_tolerance": recommendation.predicted_qr_tolerance,
                    "predicted_clip_aesthetic": recommendation.predicted_clip_aesthetic,
                    "predicted_clip_score": recommendation.predicted_clip_score,
                    "predicted_hpsv2_1": recommendation.predicted_hpsv2_1,
                    "predicted_saturation_risk": (
                        recommendation.predicted_saturation_risk
                    ),
                    "predicted_duration_ms": recommendation.predicted_duration_ms,
                }
            )

        if baseline is not None and baseline.signature not in selected_signatures:
            baseline_prediction = next(
                item for item in ranked if item.candidate.signature == baseline.signature
            )
            runtime_id = f"e026j_baseline_{_safe_identifier(baseline.method_id)}"[:100]
            methods.append(
                _runtime_method(
                    baseline,
                    runtime_id=runtime_id,
                    display_name=f"E026 fixed baseline | {baseline.method_id}",
                )
            )
            prediction_rows.append(
                {
                    "prompt_id": prompt.id,
                    "prompt_text": prompt.text,
                    "plan_method_id": runtime_id,
                    "source_method_id": baseline.method_id,
                    "role": "fixed_baseline",
                    "advisor_rank": baseline_prediction.rank,
                    "model_rank": baseline_prediction.rank,
                    "selection_profile": "fixed_baseline",
                    "effective_candidate_signature": effective_candidate_signature(
                        baseline
                    ),
                    "candidate_signature": baseline.signature,
                    "candidate_observations": baseline.observations,
                    "requested_source_output_variant": baseline.configuration.get(
                        "output_variant"
                    ),
                    "runtime_output_variant": baseline.configuration.get(
                        "output_variant"
                    ),
                    "scan_safe": baseline_prediction.scan_safe,
                    "predicted_qr_success": baseline_prediction.predicted_qr_success,
                    "predicted_qr_success_lower_bound": (
                        baseline_prediction.qr_success_lower_bound
                    ),
                    "predicted_qr_success_uncertainty": (
                        baseline_prediction.qr_success_uncertainty
                    ),
                    "predicted_qr_tolerance": baseline_prediction.predicted_qr_tolerance,
                    "predicted_clip_aesthetic": (
                        baseline_prediction.predicted_clip_aesthetic
                    ),
                    "predicted_clip_score": baseline_prediction.predicted_clip_score,
                    "predicted_hpsv2_1": baseline_prediction.predicted_hpsv2_1,
                    "predicted_saturation_risk": (
                        baseline_prediction.predicted_saturation_risk
                    ),
                    "predicted_duration_ms": baseline_prediction.predicted_duration_ms,
                }
            )

        drafts.append(
            {
                "prompt": prompt.model_dump(mode="json"),
                "methods": methods,
            }
        )

    plan_core = {
        "protocol": "e026j-v2-scan-safe-exploratory-fallback",
        "advisor_sha256": advisor_sha256,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "payload_length": len(payload),
        "error_correction": error_correction,
        "qr_context": context,
        "scan_probability_threshold": scan_probability_threshold,
        "top_k": top_k,
        "baseline_method_id": baseline_method_id,
        "seeds": list(normalized_seeds),
        "prompts": [item.model_dump(mode="json") for item in validated_prompts],
        "selection": [
            {
                "prompt_id": item["prompt"]["id"],
                "methods": [
                    {
                        "id": method["id"],
                        "source_signature": next(
                            row["candidate_signature"]
                            for row in prediction_rows
                            if row["prompt_id"] == item["prompt"]["id"]
                            and row["plan_method_id"] == method["id"]
                        ),
                    }
                    for method in item["methods"]
                ],
            }
            for item in drafts
        ],
        "predictions": prediction_rows,
    }
    plan_id = hashlib.sha256(_canonical_json(plan_core).encode("utf-8")).hexdigest()[:16]
    campaigns = []
    public_campaigns = []
    for index, draft in enumerate(drafts, start=1):
        base_name = f"E026J {plan_id} {index:02d} {draft['prompt']['id']}"
        request = {
            "name": base_name,
            "payload": payload,
            "error_correction": error_correction,
            "prompts": [draft["prompt"]],
            "seeds": list(normalized_seeds),
            "methods": draft["methods"],
            "max_attempts": 1,
        }
        validated = LabCampaignCreate.model_validate(request).model_dump(mode="json")
        campaigns.append(validated)
        public_campaigns.append(
            {
                "name": base_name,
                "prompt": draft["prompt"],
                "seeds": list(normalized_seeds),
                "methods": draft["methods"],
                "trials": len(normalized_seeds) * len(draft["methods"]),
            }
        )
    public = {
        **plan_core,
        "plan_id": plan_id,
        "campaign_count": len(campaigns),
        "trial_count": sum(item["trials"] for item in public_campaigns),
        "comparison_trial_count": len(normalized_seeds)
        * sum(
            item["role"] in {"advisor_recommendation", "fixed_baseline"}
            for item in prediction_rows
        ),
        "prerequisite_trial_count": len(normalized_seeds)
        * sum(item["role"] == "srmpgd_prerequisite" for item in prediction_rows),
        "scan_safe_recommendation_count": sum(
            item["role"] == "advisor_recommendation" and item["scan_safe"] is True
            for item in prediction_rows
        ),
        "exploratory_recommendation_count": sum(
            item["role"] == "advisor_recommendation" and item["scan_safe"] is False
            for item in prediction_rows
        ),
        "campaigns": public_campaigns,
    }
    return AdvisorInferencePlan(
        plan_id=plan_id,
        payload=payload,
        campaigns=tuple(campaigns),
        predictions=tuple(prediction_rows),
        public=public,
    )


class AdvisorInferenceRunner:
    """Run advisor-selected laboratory campaigns with power-cut-safe checkpoints."""

    def __init__(
        self,
        *,
        plan: AdvisorInferencePlan,
        api_url: str,
        output_root: Path,
        poll_seconds: float = 15.0,
        maximum_campaign_attempts: int = 2,
        reject_campaigns_with_errors: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.plan = plan
        self.api_url = api_url.rstrip("/")
        self.poll_seconds = poll_seconds
        self.maximum_campaign_attempts = maximum_campaign_attempts
        self.reject_campaigns_with_errors = reject_campaigns_with_errors
        self.progress_callback = progress_callback
        self.output_dir = Path(output_root) / plan.plan_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.output_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        self.state_path = self.output_dir / "state.json"
        self.plan_path = self.output_dir / "plan-redacted.json"
        self.predictions_path = self.output_dir / "advisor-predictions.jsonl"
        self._write_or_verify_plan()
        self.state = self._load_state()

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _write_or_verify_plan(self) -> None:
        if self.plan_path.exists():
            saved = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if saved != self.plan.public:
                raise RuntimeError("stored inference plan differs from the requested plan")
        else:
            self._atomic_json(self.plan_path, self.plan.public)
        prediction_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in self.plan.predictions
        )
        if self.predictions_path.exists():
            if self.predictions_path.read_text(encoding="utf-8") != prediction_text:
                raise RuntimeError("stored advisor predictions differ from the plan")
        else:
            temporary = self.predictions_path.with_suffix(".jsonl.tmp")
            temporary.write_text(prediction_text, encoding="utf-8")
            temporary.replace(self.predictions_path)

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("plan_id") != self.plan.plan_id:
                raise RuntimeError("stored inference state belongs to another plan")
            stored_strict = bool(state.get("reject_campaigns_with_errors", False))
            if stored_strict != self.reject_campaigns_with_errors:
                raise RuntimeError(
                    "stored inference state uses a different campaign error policy"
                )
            return state
        state = {
            "version": 2,
            "plan_id": self.plan.plan_id,
            "reject_campaigns_with_errors": self.reject_campaigns_with_errors,
            "status": "running",
            "completed_campaigns": [],
            "failed_campaigns": [],
            "attempts": {},
            "active_campaign": None,
            "history": [],
            "created_at": time.time(),
        }
        self._atomic_json(self.state_path, state)
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = time.time()
        self._atomic_json(self.state_path, self.state)

    def _notify(self, event: str, **values: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(
                {"event": event, "timestamp": time.time(), **values}
            )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        raw: bool = False,
    ) -> Any:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.api_url}{path}", data=data, headers=headers, method=method
        )
        last_error: Exception | None = None
        for retry in range(8):
            try:
                with urlopen(request, timeout=120) as response:
                    body = response.read()
                    return body if raw else json.loads(body.decode("utf-8"))
            except HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(response_body)
                    detail = parsed.get("detail", parsed)
                except json.JSONDecodeError:
                    detail = response_body or exc.reason
                message = f"HTTP {exc.code} {method} {path}: {detail}"
                # A validation or authorization error is deterministic. Retrying
                # the exact same request only hides its useful response body and
                # delays the notebook by several minutes.
                if 400 <= exc.code < 500 and exc.code not in {408, 425, 429}:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
                time.sleep(min(60.0, 2.0 ** min(retry, 5)))
            except (URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(min(60.0, 2.0 ** min(retry, 5)))
        raise RuntimeError(f"inference API unavailable after retries: {last_error}")

    def _acquire_lock(self):
        handle = (self.output_dir / "runner.lock").open("a+", encoding="utf-8")
        if os.name != "nt":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"another inference runner owns plan {self.plan.plan_id}"
                ) from exc
        return handle

    @staticmethod
    def _release_lock(handle) -> None:
        if os.name != "nt":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def _campaign_name(self, base_name: str, attempt: int) -> str:
        """Return a remote campaign name that is unique to this immutable plan."""

        suffix = f" [plan-{self.plan.plan_id}] a{attempt:02d}"
        maximum_base_length = 200 - len(suffix)
        if maximum_base_length < 1:
            raise ValueError("plan id is too long for a laboratory campaign name")
        normalized_base = str(base_name).strip() or "advisor inference"
        return f"{normalized_base[:maximum_base_length].rstrip()}{suffix}"

    @staticmethod
    def _campaign_matches_request(
        campaign: dict[str, Any], request_payload: dict[str, Any]
    ) -> bool:
        """Refuse a same-name campaign whose payload or specification differs."""

        payload = str(request_payload.get("payload") or "")
        expected_specification = deepcopy(request_payload)
        expected_specification.pop("payload", None)
        expected_specification["payload_length"] = len(payload)
        return bool(
            campaign.get("name") == request_payload.get("name")
            and campaign.get("payload_hash")
            == hashlib.sha256(payload.encode("utf-8")).hexdigest()
            and campaign.get("specification") == expected_specification
        )

    def _find_campaign(
        self, request_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        campaigns = self._request("GET", "/v1/lab/campaigns?limit=500")
        matching = [
            item
            for item in campaigns
            if self._campaign_matches_request(item, request_payload)
        ]
        return max(matching, key=lambda item: item.get("created_at", "")) if matching else None

    def _wait_for_foreign_campaigns(self, own_id: str | None = None) -> None:
        while True:
            campaigns = self._request("GET", "/v1/lab/campaigns?limit=100")
            active = [
                item
                for item in campaigns
                if item.get("status") in {"queued", "running"}
                and item.get("id") != own_id
            ]
            if not active:
                return
            self._notify(
                "waiting_foreign_campaign",
                campaigns=[item.get("id") for item in active],
            )
            time.sleep(self.poll_seconds)

    def _export(self, index: int, attempt: int, campaign_id: str) -> Path:
        path = self.exports_dir / (
            f"prompt-{index + 1:02d}-attempt-{attempt:02d}-{campaign_id}.csv"
        )
        if path.is_file() and path.stat().st_size > 0:
            return path
        body = self._request(
            "GET", f"/v1/lab/campaigns/{campaign_id}/results.csv", raw=True
        )
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_bytes(body)
        temporary.replace(path)
        return path

    def _wait_campaign(self, index: int, attempt: int, campaign_id: str) -> dict[str, Any]:
        last_progress = None
        while True:
            campaign = self._request("GET", f"/v1/lab/campaigns/{campaign_id}")
            progress = (
                campaign.get("status"),
                campaign.get("completed_trials"),
                campaign.get("total_trials"),
                campaign.get("accepted_trials"),
            )
            if progress != last_progress:
                print(
                    f"inference prompt={index + 1:02d}/{len(self.plan.campaigns):02d} "
                    f"attempt={attempt} status={progress[0]} "
                    f"trials={progress[1]}/{progress[2]} accepted={progress[3]}"
                )
                last_progress = progress
            current_trial = next(
                (
                    item
                    for item in campaign.get("trials", [])
                    if item.get("status") == "running"
                ),
                None,
            )
            self._notify(
                "inference_progress",
                plan_id=self.plan.plan_id,
                prompt_number=index + 1,
                prompt_count=len(self.plan.campaigns),
                attempt=attempt,
                campaign_id=campaign_id,
                status=progress[0],
                completed_trials=progress[1],
                total_trials=progress[2],
                accepted_trials=progress[3],
                current_prompt_id=(current_trial or {}).get("prompt_id"),
                current_method_id=(current_trial or {}).get("method_id"),
                current_seed=(current_trial or {}).get("seed"),
            )
            if campaign.get("status") in TERMINAL_CAMPAIGN_STATUSES:
                export_path = self._export(index, attempt, campaign_id)
                campaign["export_path"] = str(export_path)
                return campaign
            time.sleep(self.poll_seconds)

    def run(self) -> dict[str, Any]:
        handle = self._acquire_lock()
        try:
            for index, base_request in enumerate(self.plan.campaigns):
                if index in self.state["completed_campaigns"]:
                    continue
                succeeded = False
                while (
                    int(self.state["attempts"].get(str(index), 0))
                    < self.maximum_campaign_attempts
                ):
                    active = self.state.get("active_campaign")
                    if active and int(active["index"]) == index:
                        attempt = int(active["attempt"])
                        campaign_id = str(active["campaign_id"])
                    else:
                        attempt = int(self.state["attempts"].get(str(index), 0)) + 1
                        request_payload = deepcopy(base_request)
                        request_payload["name"] = self._campaign_name(
                            str(base_request["name"]), attempt
                        )
                        existing = self._find_campaign(request_payload)
                        if existing is not None:
                            campaign = existing
                        else:
                            self._wait_for_foreign_campaigns()
                            campaign = self._request(
                                "POST", "/v1/lab/campaigns", request_payload
                            )
                        campaign_id = str(campaign["id"])
                        self.state["attempts"][str(index)] = attempt
                        self.state["active_campaign"] = {
                            "index": index,
                            "attempt": attempt,
                            "campaign_id": campaign_id,
                            "name": request_payload["name"],
                        }
                        self._save_state()
                    campaign = self._wait_campaign(index, attempt, campaign_id)
                    status = str(campaign["status"])
                    self.state["history"].append(
                        {
                            "index": index,
                            "attempt": attempt,
                            "campaign_id": campaign_id,
                            "status": status,
                            "export_path": campaign.get("export_path"),
                        }
                    )
                    self.state["active_campaign"] = None
                    campaign_succeeded = status in SUCCESSFUL_CAMPAIGN_STATUSES and not (
                        self.reject_campaigns_with_errors
                        and status == "completed_with_errors"
                    )
                    if campaign_succeeded:
                        self.state["completed_campaigns"].append(index)
                        succeeded = True
                    self._save_state()
                    if succeeded:
                        break
                if not succeeded and index not in self.state["failed_campaigns"]:
                    self.state["failed_campaigns"].append(index)
                    self._save_state()
            self.state["status"] = (
                "completed_with_errors"
                if self.state["failed_campaigns"]
                or (
                    not self.reject_campaigns_with_errors
                    and any(
                        item.get("status") == "completed_with_errors"
                        for item in self.state["history"]
                    )
                )
                else "completed"
            )
            self._save_state()
            return self.summary()
        finally:
            self._release_lock(handle)

    def summary(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan.plan_id,
            "status": self.state["status"],
            "campaigns": len(self.plan.campaigns),
            "completed_campaigns": len(self.state["completed_campaigns"]),
            "failed_campaigns": list(self.state["failed_campaigns"]),
            "exports": len(list(self.exports_dir.glob("*.csv"))),
            "state_path": str(self.state_path),
        }


def load_advisor_inference_results(output_dir: Path) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    predictions = {}
    prediction_path = output_dir / "advisor-predictions.jsonl"
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            predictions[(row["prompt_id"], row["plan_method_id"])] = row

    allowed_campaign_ids: set[str] | None = None
    state_path = output_dir / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed_indexes = {int(value) for value in state.get("completed_campaigns", [])}
        strict = bool(state.get("reject_campaigns_with_errors", False))
        successful_statuses = {"completed"} if strict else SUCCESSFUL_CAMPAIGN_STATUSES
        final_campaign_by_index: dict[int, str] = {}
        for item in state.get("history", []):
            index = int(item.get("index", -1))
            campaign_id = str(item.get("campaign_id") or "")
            if (
                index in completed_indexes
                and campaign_id
                and item.get("status") in successful_statuses
            ):
                final_campaign_by_index[index] = campaign_id
        if final_campaign_by_index:
            allowed_campaign_ids = set(final_campaign_by_index.values())

    trials: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "exports").glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if (
                    allowed_campaign_ids is not None
                    and str(row.get("campaign_id")) not in allowed_campaign_ids
                ):
                    continue
                row["_source_file"] = str(path)
                trials[str(row.get("trial_id"))] = row

    results = []
    for row in trials.values():
        key = (row.get("prompt_id"), row.get("method_id"))
        prediction = predictions.get(key)
        if prediction is None:
            continue
        results.append(
            {
                **prediction,
                "trial_id": row.get("trial_id"),
                "campaign_id": row.get("campaign_id"),
                "method_id": row.get("method_id"),
                "output_variant": row.get("selected_variant")
                or row.get("output_variant_requested"),
                "service_selected_variant": row.get("selected_variant") or None,
                "final_image_sha256": row.get("provenance_final_image_sha256")
                or row.get("final_image_sha256")
                or None,
                "srmpgd_selected_iteration": _first_finite(
                    row,
                    (
                        "quality_diffqrcoder_srmpgd_selected_iteration",
                        "diffqrcoder_srmpgd_selected_iteration",
                    ),
                ),
                "srmpgd_iteration_zero_exact": _first_finite(
                    row,
                    (
                        "quality_diffqrcoder_srmpgd_iteration_zero_exact",
                        "diffqrcoder_srmpgd_iteration_zero_exact",
                        "quality_srmpgd_iteration_zero_exact",
                        "srmpgd_iteration_zero_exact",
                    ),
                ),
                "seed": _finite(row.get("seed")),
                "status": row.get("status"),
                "generation_run_id": row.get("generation_run_id"),
                "stage1_reused": _finite(row.get("stage1_reused")),
                "stage1_source_run_id": row.get("stage1_source_run_id") or None,
                "stage1_image_sha256": row.get("provenance_stage1_image_sha256")
                or row.get("stage1_image_sha256")
                or None,
                "stage2_source_run_id": row.get("provenance_stage2_source_run_id")
                or row.get("stage2_source_run_id")
                or None,
                "stage2_source_method_id": row.get(
                    "provenance_stage2_source_method_id"
                )
                or row.get("stage2_source_method_id")
                or None,
                "stage2_source_latent_sha256": row.get(
                    "provenance_stage2_source_latent_sha256"
                )
                or row.get("stage2_source_latent_sha256")
                or None,
                "stage2_latent_sha256": row.get("provenance_stage2_latent_sha256")
                or row.get("stage2_latent_sha256")
                or None,
                "stage2_pairing_status": row.get("provenance_stage2_pairing_status")
                or row.get("stage2_pairing_status")
                or None,
                "stage2_pairing_exact": _first_finite(
                    row,
                    (
                        "quality_diffqrcoder_stage2_pairing_exact",
                        "diffqrcoder_stage2_pairing_exact",
                        "provenance_stage2_pairing_exact",
                    ),
                ),
                "srmpgd_stage2_image_sha256": row.get(
                    "provenance_srmpgd_stage2_image_sha256"
                )
                or row.get("srmpgd_stage2_image_sha256")
                or None,
                "srmpgd_selected_image_sha256": row.get(
                    "provenance_srmpgd_selected_image_sha256"
                )
                or row.get("srmpgd_selected_image_sha256")
                or None,
                "payload_length": _finite(row.get("payload_length")),
                "error_correction": row.get("error_correction") or None,
                "qr_success": _first_finite(
                    row, ("quality_qr_verify_any_exact", "exact_payload_match")
                ),
                "qr_tolerance": _first_finite(
                    row, ("quality_qr_verify_tolerance_score", "scan_pass_rate")
                ),
                "clip_aesthetic": _finite(row.get("quality_clip_aesthetic")),
                "clip_score": _finite(row.get("quality_clip_score")),
                "hpsv2_1": _finite(row.get("quality_hpsv2_1")),
                "saturation_risk": max(
                    value
                    for value in (
                        _finite(row.get("quality_high_saturation_pixel_ratio")),
                        _finite(row.get("quality_rgb_clipped_channel_ratio")),
                        0.0,
                    )
                    if value is not None
                ),
                "duration_ms": _first_finite(row, ("total_ms", "generation_ms")),
                "module_error_rate": _finite(row.get("module_error_rate")),
                "error": row.get("error"),
                "section": "advisor_inference",
                "predicted_qr_probability": prediction.get("predicted_qr_success"),
                "_source_file": row.get("_source_file"),
            }
        )
    for result in results:
        iteration = _finite(result.get("srmpgd_selected_iteration"))
        requested = result.get("requested_source_output_variant")
        result["srmpgd_effective"] = bool(
            requested in {"srmpgd", "auto"}
            and iteration is not None
            and iteration > 0
        )
        result["srmpgd_noop"] = bool(
            requested in {"srmpgd", "auto"}
            and iteration is not None
            and iteration == 0
        )
        if result["srmpgd_noop"]:
            # Older exports could label an iteration-zero backend result as
            # ``srmpgd``. Normalize only that legacy label. Never rewrite
            # ``raw``: it means the surrounding auto selector chose Stage 1.
            if result["output_variant"] == "srmpgd":
                result["output_variant"] = "srpg"
    return sorted(
        results,
        key=lambda item: (
            str(item["prompt_id"]),
            int(item.get("advisor_rank") or 999),
            str(item.get("role")),
            float(item.get("seed") or 0),
        ),
    )


def select_advisor_inference_winners(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    winners = []
    unique_entries = deduplicate_advisor_inference_results(entries)
    for prompt_id in sorted({str(item["prompt_id"]) for item in unique_entries}):
        candidates = [
            dict(item) for item in unique_entries if item["prompt_id"] == prompt_id
        ]
        if not candidates:
            continue
        winners.append(
            max(
                candidates,
                key=lambda item: (
                    item.get("qr_success") or 0.0,
                    item.get("qr_tolerance") or 0.0,
                    item.get("hpsv2_1") or -math.inf,
                    item.get("clip_aesthetic") or -math.inf,
                    item.get("clip_score") or -math.inf,
                    -(item.get("saturation_risk") or 0.0),
                    item.get("role") == "advisor_recommendation",
                    -(item.get("advisor_rank") or 999),
                ),
            )
        )
    return winners


def deduplicate_advisor_inference_results(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse byte-identical advisor outputs within each prompt and seed.

    Missing hashes are kept distinct.  This deliberately avoids claiming that
    two images are identical merely because their recipes share a Stage-2
    signature; only measured image provenance or downloaded bytes may collapse
    a result after generation.
    """

    groups: dict[tuple[str, float | None, str], list[dict[str, Any]]] = {}
    for source in entries:
        item = dict(source)
        digest = str(
            item.get("image_sha256") or item.get("final_image_sha256") or ""
        ).strip()
        if not digest:
            digest = f"run:{item.get('generation_run_id') or item.get('trial_id') or id(source)}"
        key = (str(item.get("prompt_id")), _finite(item.get("seed")), digest)
        groups.setdefault(key, []).append(item)

    unique = []
    for aliases in groups.values():
        def metric(item: Mapping[str, Any], name: str, default: float) -> float:
            value = _finite(item.get(name))
            return default if value is None else value

        representative = max(
            aliases,
            key=lambda item: (
                metric(item, "qr_success", 0.0),
                metric(item, "qr_tolerance", 0.0),
                metric(item, "hpsv2_1", -math.inf),
                metric(item, "clip_aesthetic", -math.inf),
                metric(item, "clip_score", -math.inf),
                -metric(item, "saturation_risk", 0.0),
                -metric(item, "advisor_rank", 999.0),
            ),
        )
        representative["duplicate_count"] = len(aliases) - 1
        representative["duplicate_method_ids"] = sorted(
            {
                str(item.get("source_method_id") or item.get("method_id"))
                for item in aliases
            }
        )
        unique.append(representative)
    return sorted(
        unique,
        key=lambda item: (
            str(item.get("prompt_id")),
            float(item.get("seed") or 0),
            int(_finite(item.get("advisor_rank")) or 999),
        ),
    )


def summarize_advisor_inference_results(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize advisor inference without hiding technical failures.

    Rates whose name does not end in ``_generated`` use every planned result as
    denominator.  A missing QR-Verify measurement therefore counts as failure,
    while the companion generated-only rate remains available for diagnosis.
    SRPG prerequisite trials are operational dependencies and are excluded from
    the advisor-versus-baseline comparison.
    """

    comparable = [
        dict(item)
        for item in entries
        if item.get("role") in {"advisor_recommendation", "fixed_baseline"}
    ]
    advisor = [item for item in comparable if item.get("role") == "advisor_recommendation"]
    baseline = [item for item in comparable if item.get("role") == "fixed_baseline"]
    generated = [item for item in comparable if _finite(item.get("qr_success")) is not None]
    generated_advisor = [
        item for item in advisor if _finite(item.get("qr_success")) is not None
    ]
    hashed_generated_advisor = [
        item
        for item in generated_advisor
        if str(item.get("final_image_sha256") or "").strip()
    ]
    unique_generated_advisor = deduplicate_advisor_inference_results(
        hashed_generated_advisor
    )
    generated_baseline = [
        item for item in baseline if _finite(item.get("qr_success")) is not None
    ]
    rank1 = [item for item in advisor if int(item.get("advisor_rank") or 0) == 1]
    generated_rank1 = [
        item for item in rank1 if _finite(item.get("qr_success")) is not None
    ]
    successful_advisor = [
        item for item in generated_advisor if (_finite(item.get("qr_success")) or 0.0) >= 0.5
    ]

    def success_rate(rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(
            1.0 if (_finite(item.get("qr_success")) or 0.0) >= 0.5 else 0.0
            for item in rows
        ) / len(rows)

    def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
        values = [
            value
            for item in rows
            if (value := _finite(item.get(field))) is not None
        ]
        return sum(values) / len(values) if values else None

    coverage: dict[tuple[str, float | None], float] = {}
    for item in advisor:
        key = (str(item.get("prompt_id")), _finite(item.get("seed")))
        coverage[key] = max(
            coverage.get(key, 0.0),
            1.0 if (_finite(item.get("qr_success")) or 0.0) >= 0.5 else 0.0,
        )

    return {
        "images_planned": len(comparable),
        "images_measured": len(generated),
        "technical_error_images": sum(
            1 for item in comparable if str(item.get("status")) == "error"
        ),
        "technical_completion_rate": (
            len(generated) / len(comparable) if comparable else None
        ),
        "prompts_measured": len({str(item.get("prompt_id")) for item in generated}),
        "rank1_images_planned": len(rank1),
        "rank1_images_measured": len(generated_rank1),
        "rank1_technical_completion_rate": (
            len(generated_rank1) / len(rank1) if rank1 else None
        ),
        "rank1_qr_verify_success_rate": success_rate(rank1),
        "rank1_qr_verify_success_rate_generated": success_rate(generated_rank1),
        "top_k_images_planned": len(advisor),
        "top_k_images_measured": len(generated_advisor),
        "top_k_images_with_provenance_hash": len(hashed_generated_advisor),
        "top_k_unique_images_measured": (
            len(unique_generated_advisor)
            if len(hashed_generated_advisor) == len(generated_advisor)
            else None
        ),
        "top_k_duplicate_images": (
            len(hashed_generated_advisor) - len(unique_generated_advisor)
            if len(hashed_generated_advisor) == len(generated_advisor)
            else None
        ),
        "top_k_technical_completion_rate": (
            len(generated_advisor) / len(advisor) if advisor else None
        ),
        "top_k_image_qr_verify_success_rate": success_rate(advisor),
        "top_k_image_qr_verify_success_rate_generated": success_rate(
            generated_advisor
        ),
        "top_k_prompt_seed_coverage": (
            sum(coverage.values()) / len(coverage) if coverage else None
        ),
        "srmpgd_requested_images": sum(
            item.get("requested_source_output_variant") == "srmpgd"
            for item in generated_advisor
        ),
        "srmpgd_effective_images": sum(
            bool(item.get("srmpgd_effective")) for item in generated_advisor
        ),
        "srmpgd_noop_images": sum(
            bool(item.get("srmpgd_noop")) for item in generated_advisor
        ),
        "baseline_qr_verify_success_rate": success_rate(baseline),
        "baseline_qr_verify_success_rate_generated": success_rate(
            generated_baseline
        ),
        "successful_advisor_clip_aesthetic": mean(
            successful_advisor, "clip_aesthetic"
        ),
        "successful_advisor_clip_score": mean(successful_advisor, "clip_score"),
        "successful_advisor_hpsv2_1": mean(successful_advisor, "hpsv2_1"),
    }
