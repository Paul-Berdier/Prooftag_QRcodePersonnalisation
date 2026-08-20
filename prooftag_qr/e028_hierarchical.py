from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .advisor_inference import AdvisorInferencePlan
from .e027_policy import build_e027_holdout_prompts
from .lab import laboratory_profiles
from .parameter_advisor import (
    AdvisorDataset,
    AdvisorRecord,
    ParameterRecommendation,
    RecipeCandidate,
)
from .qr import generate_diffqrcoder_qr
from .schemas import LabCampaignCreate, LabMethod, LabPrompt

E028_PIPELINE_STATES = ("stage1", "stage2", "srmpgd")
E028_POLICIES = ("fixed_cascade", "advisor_top1", "advisor_best_of_chains")
E028_QR_MONSTER = "monster-labs/control_v1p_sd15_qrcode_monster"


@dataclass(frozen=True, slots=True)
class E028CandidatePools:
    stage1: tuple[RecipeCandidate, ...]
    stage2: tuple[RecipeCandidate, ...]
    srmpgd: tuple[RecipeCandidate, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _signature(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()


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


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _safe_identifier(value: Any, maximum: int = 54) -> str:
    result = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return (result or "recipe")[:maximum]


def _recipe(
    configuration: Mapping[str, Any], method_id: str, observations: int = 0
) -> RecipeCandidate:
    material = deepcopy(dict(configuration))
    signature = _signature(material)
    return RecipeCandidate(
        id=f"recipe-{signature[:10]}",
        method_id=method_id,
        configuration=material,
        signature=signature,
        observations=observations,
    )


def _is_public_diffqrcoder_qr_monster(candidate: RecipeCandidate) -> bool:
    configuration = candidate.configuration
    if configuration.get("backend", "controlnet") != "controlnet":
        return False
    model = configuration.get("model") or {}
    return bool(
        model.get("diffqrcoder_upstream_enabled")
        and str(model.get("controlnet_model_id")) == E028_QR_MONSTER
        and str(model.get("controlnet_model_subfolder") or "v2") == "v2"
    )


def _candidate_stage(candidate: RecipeCandidate) -> str | None:
    if not _is_public_diffqrcoder_qr_monster(candidate):
        return None
    tools = candidate.configuration.get("tools") or {}
    if tools.get("srmpgd_enabled"):
        return "srmpgd"
    if tools.get("srpg_enabled"):
        return "stage2"
    if candidate.configuration.get("output_variant", "raw") == "raw":
        return "stage1"
    return None


def _fallback_candidates() -> list[RecipeCandidate]:
    wanted = {
        "diffqrcoder_stage1",
        "diffqrcoder_srpg",
        "diffqrcoder_srmpgd_robust",
    }
    base = [
        _recipe(profile, str(profile["id"]))
        for profile in laboratory_profiles()
        if profile["id"] in wanted
    ]
    by_stage = {_candidate_stage(candidate): candidate for candidate in base}
    expanded = list(base)

    # E026 mostly varied Stage 2 and SR-MPGD.  E028 must also let the advisor
    # choose the Stage 1 operating point instead of silently reusing one fixed
    # recipe for every prompt.  These candidates stay on the exact public
    # DiffQRCoder/QR-Monster stack; only documented generation controls vary.
    stage1 = by_stage["stage1"]
    for identifier, steps, guidance, control in (
        ("e028_s1_soft", 30, 5.5, 1.15),
        ("e028_s1_balanced", 40, 7.5, 1.35),
        ("e028_s1_structural", 50, 7.5, 1.55),
        ("e028_s1_detailed", 50, 9.0, 1.35),
    ):
        configuration = deepcopy(stage1.configuration)
        configuration["id"] = identifier
        configuration["generation"].update(
            {
                "steps": steps,
                "guidance_scale": guidance,
                "controlnet_scale": control,
            }
        )
        expanded.append(_recipe(configuration, identifier))

    stage2 = by_stage["stage2"]
    for strength in (0.35, 0.50, 0.65, 0.80):
        configuration = deepcopy(stage2.configuration)
        identifier = f"e028_s2_strength_{int(strength * 100):02d}"
        configuration["id"] = identifier
        configuration["tools"]["settings"][
            "diffqrcoder_stage2_strength"
        ] = strength
        expanded.append(_recipe(configuration, identifier))

    srmpgd = by_stage["srmpgd"]
    for gamma in (10.0, 30.0, 100.0, 300.0):
        configuration = deepcopy(srmpgd.configuration)
        identifier = f"e028_mpgd_gamma_{int(gamma):04d}"
        configuration["id"] = identifier
        settings = configuration["tools"]["settings"]
        settings.update(
            {
                "srmpgd_step_size": gamma,
                "srmpgd_max_iterations": 4,
                "srmpgd_lpips_weight": 0.10,
            }
        )
        expanded.append(_recipe(configuration, identifier))
    return expanded


def partition_e028_candidates(
    candidates: Sequence[RecipeCandidate],
) -> E028CandidatePools:
    """Keep only public DiffQRCoder + QR Monster recipes and split by stage."""

    unique: dict[str, RecipeCandidate] = {}
    for candidate in [*candidates, *_fallback_candidates()]:
        if _candidate_stage(candidate) is None:
            continue
        previous = unique.get(candidate.signature)
        if previous is None or candidate.observations > previous.observations:
            unique[candidate.signature] = candidate
    stages: dict[str, list[RecipeCandidate]] = {name: [] for name in E028_PIPELINE_STATES}
    for candidate in unique.values():
        stage = _candidate_stage(candidate)
        if stage is not None:
            stages[stage].append(candidate)
    for values in stages.values():
        values.sort(key=lambda item: (-item.observations, item.signature))
    missing = [name for name, values in stages.items() if not values]
    if missing:
        raise ValueError(f"missing E028 candidate stages: {missing}")
    return E028CandidatePools(
        stage1=tuple(stages["stage1"]),
        stage2=tuple(stages["stage2"]),
        srmpgd=tuple(stages["srmpgd"]),
    )


def build_e028_holdout_prompts(
    count: int = 30,
    *,
    seen_prompt_texts: Sequence[str] = (),
) -> list[dict[str, str]]:
    prompts = build_e027_holdout_prompts(count, seen_prompt_texts=seen_prompt_texts)
    return [
        {**item, "id": str(item["id"]).replace("e027h_", "e028h_", 1)}
        for item in prompts
    ]


def _runtime_method(
    configuration: Mapping[str, Any],
    *,
    runtime_id: str,
    name: str,
    output_variant: str,
) -> dict[str, Any]:
    method = deepcopy(dict(configuration))
    method.update(
        {
            "id": runtime_id[:100],
            "name": name[:200],
            "enabled": True,
            "output_variant": output_variant,
            "reuse_stage1": True,
            # E028/E029 are paired scientific campaigns.  A later stage may
            # never silently regenerate its own Stage 1 when the intended
            # source is missing from the in-memory campaign cache.
            "require_exact_stage1_reuse": output_variant != "raw",
        }
    )
    return LabMethod.model_validate(method).model_dump(mode="json")


def _compose_stage2(stage1: RecipeCandidate, template: RecipeCandidate) -> RecipeCandidate:
    configuration = deepcopy(template.configuration)
    configuration["generation"] = deepcopy(stage1.configuration.get("generation") or {})
    configuration["model"] = deepcopy(stage1.configuration.get("model") or {})
    configuration["reuse_stage1"] = True
    configuration["output_variant"] = "srpg"
    tools = deepcopy(configuration.get("tools") or {})
    tools["srpg_enabled"] = True
    tools["srmpgd_enabled"] = False
    tools["settings"] = {
        key: value
        for key, value in (tools.get("settings") or {}).items()
        if not str(key).startswith("srmpgd_")
    }
    configuration["tools"] = tools
    return _recipe(configuration, template.method_id, template.observations)


def _compose_srmpgd(stage2: RecipeCandidate, template: RecipeCandidate) -> RecipeCandidate:
    configuration = deepcopy(stage2.configuration)
    # Force the post-Stage-2 branch. In ``auto`` mode the delivery selector also
    # sees the raw Stage 1 candidate and can return it instead of the SR-MPGD
    # result, which breaks both the Stage-1 prohibition and raster pairing.
    configuration["output_variant"] = "srmpgd"
    tools = deepcopy(configuration.get("tools") or {})
    source_tools = template.configuration.get("tools") or {}
    source_settings = source_tools.get("settings") or {}
    settings = deepcopy(tools.get("settings") or {})
    settings.update(
        {
            key: value
            for key, value in source_settings.items()
            if str(key).startswith("srmpgd_")
        }
    )
    settings["srmpgd_min_qr_tolerance"] = max(
        float(settings.get("srmpgd_min_qr_tolerance", 0.0)),
        0.80,
    )
    tools.update({"srpg_enabled": True, "srmpgd_enabled": True, "settings": settings})
    configuration["tools"] = tools
    return _recipe(configuration, template.method_id, template.observations)


def _recommend(
    advisor: Any,
    *,
    prompt: str,
    candidates: Sequence[RecipeCandidate],
    prompt_embedding: Sequence[float] | np.ndarray | None,
    payload_length: int,
    error_correction: str,
    qr_context: Mapping[str, Any],
    threshold: float,
    context_features: Mapping[str, Any] | None = None,
) -> list[ParameterRecommendation]:
    return advisor.recommend(
        prompt=prompt,
        candidates=candidates,
        prompt_embedding=prompt_embedding,
        payload_length=payload_length,
        error_correction=error_correction,
        qr_context=qr_context,
        context_features=context_features,
        scan_probability_threshold=threshold,
        limit=len(candidates),
    )


def _optional(value: float | None, default: float = -math.inf) -> float:
    return default if value is None or not math.isfinite(float(value)) else float(value)


def _stage1_profiles(
    ranked: Sequence[ParameterRecommendation], count: int
) -> list[tuple[str, ParameterRecommendation]]:
    if not ranked:
        raise ValueError("the advisor returned no Stage 1 recommendation")
    structural = max(
        ranked,
        key=lambda item: (
            item.qr_success_lower_bound,
            _optional(item.predicted_qr_tolerance),
            -_optional(item.predicted_saturation_risk, 0.0),
            _optional(item.predicted_hpsv2_1),
        ),
    )
    aesthetic = max(
        ranked,
        key=lambda item: (
            _optional(item.predicted_saturation_risk, 0.0) <= 0.05,
            _optional(item.predicted_hpsv2_1),
            _optional(item.predicted_clip_aesthetic),
            _optional(item.predicted_clip_score),
            item.qr_success_lower_bound,
        ),
    )
    result: list[tuple[str, ParameterRecommendation]] = []
    candidates = [
        ("structural", structural),
        ("aesthetic", aesthetic),
        *[("ranked", item) for item in ranked],
    ]
    for profile, item in candidates:
        if any(existing.candidate.signature == item.candidate.signature for _, existing in result):
            continue
        result.append((profile, item))
        if len(result) == count:
            break
    if len(result) < count:
        raise ValueError(f"only {len(result)} distinct Stage 1 profiles for count={count}")
    return result


def _stage2_profiles(
    ranked: Sequence[ParameterRecommendation], count: int
) -> list[tuple[str, ParameterRecommendation]]:
    if not ranked:
        raise ValueError("the advisor returned no Stage 2 recommendation")
    robust = max(
        ranked,
        key=lambda item: (
            item.scan_safe,
            item.qr_success_lower_bound,
            _optional(item.predicted_qr_tolerance),
            _optional(item.predicted_hpsv2_1),
        ),
    )
    safe = [item for item in ranked if item.scan_safe] or list(ranked)
    aesthetic = max(
        safe,
        key=lambda item: (
            _optional(item.predicted_saturation_risk, 0.0) <= 0.05,
            _optional(item.predicted_hpsv2_1),
            _optional(item.predicted_clip_aesthetic),
            _optional(item.predicted_clip_score),
            item.qr_success_lower_bound,
        ),
    )
    result: list[tuple[str, ParameterRecommendation]] = []
    candidates = [
        ("robust", robust),
        ("aesthetic", aesthetic),
        *[("ranked", item) for item in ranked],
    ]
    for profile, item in candidates:
        if any(existing.candidate.signature == item.candidate.signature for _, existing in result):
            continue
        result.append((profile, item))
        if len(result) == count:
            break
    if len(result) < count:
        raise ValueError(f"only {len(result)} distinct Stage 2 profiles for count={count}")
    return result


def _prediction(
    recommendation: ParameterRecommendation | None,
    *,
    prompt: LabPrompt,
    runtime_method: Mapping[str, Any],
    source_method_id: str,
    role: str,
    pipeline_state: str,
    chain_id: str,
    profile: str,
    parent_stage1_method_id: str | None = None,
    parent_stage2_method_id: str | None = None,
    fixed: bool = False,
) -> dict[str, Any]:
    candidate = recommendation.candidate if recommendation is not None else None
    return {
        "prompt_id": prompt.id,
        "prompt_text": prompt.text,
        "plan_method_id": runtime_method["id"],
        "source_method_id": source_method_id,
        "role": role,
        "pipeline_state": pipeline_state,
        "chain_id": chain_id,
        "selection_profile": profile,
        "fixed_control": fixed,
        "parent_stage1_method_id": parent_stage1_method_id,
        "parent_stage2_method_id": parent_stage2_method_id,
        "candidate_signature": candidate.signature if candidate else _signature(runtime_method),
        "candidate_configuration": (
            deepcopy(candidate.configuration)
            if candidate
            else deepcopy(dict(runtime_method))
        ),
        "candidate_observations": candidate.observations if candidate else 0,
        "advisor_rank": recommendation.rank if recommendation else None,
        "scan_safe": recommendation.scan_safe if recommendation else None,
        "predicted_qr_success": recommendation.predicted_qr_success if recommendation else None,
        "predicted_qr_success_lower_bound": (
            recommendation.qr_success_lower_bound if recommendation else None
        ),
        "predicted_qr_success_uncertainty": (
            recommendation.qr_success_uncertainty if recommendation else None
        ),
        "predicted_qr_tolerance": recommendation.predicted_qr_tolerance if recommendation else None,
        "predicted_clip_aesthetic": (
            recommendation.predicted_clip_aesthetic if recommendation else None
        ),
        "predicted_clip_score": recommendation.predicted_clip_score if recommendation else None,
        "predicted_hpsv2_1": recommendation.predicted_hpsv2_1 if recommendation else None,
        "predicted_saturation_risk": (
            recommendation.predicted_saturation_risk if recommendation else None
        ),
        "predicted_duration_ms": recommendation.predicted_duration_ms if recommendation else None,
        "requested_source_output_variant": runtime_method["output_variant"],
        "runtime_output_variant": runtime_method["output_variant"],
    }


def build_e028_hierarchical_plan(
    *,
    advisor: Any,
    candidates: Sequence[RecipeCandidate],
    prompts: Sequence[Mapping[str, Any]],
    payload: str,
    advisor_sha256: str,
    prompt_embedding_provider: Callable[[str], Sequence[float]] | None = None,
    seen_prompt_texts: Sequence[str] = (),
    seeds: Sequence[int] = (1_083_001, 1_211_001, 1_327_001),
    stage1_top_k: int = 2,
    stage2_top_k: int = 2,
    scan_probability_threshold: float = 0.80,
    qr_tolerance_threshold: float = 0.80,
    saturation_threshold: float = 0.05,
    error_correction: str = "M",
    qr_context: Mapping[str, Any] | None = None,
    include_fixed_control: bool = True,
) -> AdvisorInferencePlan:
    """Build prompt-conditioned exact Stage 1 -> Stage 2 -> SR-MPGD chains."""

    if not advisor_sha256:
        raise ValueError("advisor_sha256 is required")
    if stage1_top_k < 1 or stage2_top_k < 1:
        raise ValueError("stage top-k values must be positive")
    if not 0.0 <= scan_probability_threshold <= 1.0:
        raise ValueError("scan_probability_threshold must be between zero and one")
    if not 0.0 <= qr_tolerance_threshold <= 1.0:
        raise ValueError("qr_tolerance_threshold must be between zero and one")
    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be between zero and one")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("E028 seeds must be non-empty and unique")
    validated_prompts = [LabPrompt.model_validate(dict(item)) for item in prompts]
    if not validated_prompts:
        raise ValueError("E028 requires prompts")
    seen = {_normalized_prompt(value) for value in seen_prompt_texts}
    overlap = [item.id for item in validated_prompts if _normalized_prompt(item.text) in seen]
    if overlap:
        raise ValueError(f"E028 prompts already occur in advisor training: {overlap}")
    pools = partition_e028_candidates(candidates)
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
    fixed_profiles = {item["id"]: item for item in laboratory_profiles()}
    campaigns: list[dict[str, Any]] = []
    public_campaigns: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(validated_prompts, start=1):
        embedding = prompt_embedding_provider(prompt.text) if prompt_embedding_provider else None
        stage1_ranked = _recommend(
            advisor,
            prompt=prompt.text,
            candidates=pools.stage1,
            prompt_embedding=embedding,
            payload_length=len(payload),
            error_correction=error_correction,
            qr_context=context,
            threshold=scan_probability_threshold,
        )
        stage1_selected = _stage1_profiles(stage1_ranked, stage1_top_k)
        stage1_methods: list[dict[str, Any]] = []
        stage2_methods: list[dict[str, Any]] = []
        srmpgd_methods: list[dict[str, Any]] = []
        prompt_predictions: list[dict[str, Any]] = []

        if include_fixed_control:
            fixed_stage1 = _runtime_method(
                fixed_profiles["diffqrcoder_stage1"],
                runtime_id="e028_fixed_stage1",
                name="E028 fixed Stage 1 source",
                output_variant="raw",
            )
            fixed_stage2 = _runtime_method(
                fixed_profiles["diffqrcoder_srpg"],
                runtime_id="e028_fixed_stage2",
                name="E028 fixed Stage 2",
                output_variant="srpg",
            )
            fixed_srmpgd = _runtime_method(
                fixed_profiles["diffqrcoder_srmpgd_robust"],
                runtime_id="e028_fixed_srmpgd",
                name="E028 fixed SR-MPGD",
                output_variant="srmpgd",
            )
            fixed_srmpgd["tools"]["settings"]["srmpgd_min_qr_tolerance"] = float(
                qr_tolerance_threshold
            )
            stage1_methods.append(fixed_stage1)
            stage2_methods.append(fixed_stage2)
            srmpgd_methods.append(fixed_srmpgd)
            for method, state, role, parent1, parent2 in [
                (fixed_stage1, "stage1", "e028_fixed_stage1", None, None),
                (fixed_stage2, "stage2", "e028_fixed_stage2", fixed_stage1["id"], None),
                (
                    fixed_srmpgd,
                    "srmpgd",
                    "e028_fixed_srmpgd",
                    fixed_stage1["id"],
                    fixed_stage2["id"],
                ),
            ]:
                prompt_predictions.append(
                    _prediction(
                        None,
                        prompt=prompt,
                        runtime_method=method,
                        source_method_id=method["id"],
                        role=role,
                        pipeline_state=state,
                        chain_id="fixed",
                        profile="fixed_control",
                        parent_stage1_method_id=parent1,
                        parent_stage2_method_id=parent2,
                        fixed=True,
                    )
                )

        for stage1_index, (stage1_profile, stage1_recommendation) in enumerate(
            stage1_selected, start=1
        ):
            stage1_runtime_id = f"e028_a_s1_{stage1_index:02d}"
            stage1_method = _runtime_method(
                stage1_recommendation.candidate.configuration,
                runtime_id=stage1_runtime_id,
                name=f"E028 advisor Stage 1 {stage1_profile}",
                output_variant="raw",
            )
            stage1_methods.append(stage1_method)
            prompt_predictions.append(
                _prediction(
                    stage1_recommendation,
                    prompt=prompt,
                    runtime_method=stage1_method,
                    source_method_id=stage1_recommendation.candidate.method_id,
                    role="e028_advisor_stage1",
                    pipeline_state="stage1",
                    chain_id=f"advisor-s1-{stage1_index:02d}",
                    profile=stage1_profile,
                )
            )
            stage2_candidates = [
                _compose_stage2(stage1_recommendation.candidate, item)
                for item in pools.stage2
            ]
            stage2_ranked = _recommend(
                advisor,
                prompt=prompt.text,
                candidates=stage2_candidates,
                prompt_embedding=embedding,
                payload_length=len(payload),
                error_correction=error_correction,
                qr_context=context,
                threshold=scan_probability_threshold,
                context_features={
                    "parent_qr_success": stage1_recommendation.predicted_qr_success,
                    "parent_qr_tolerance": stage1_recommendation.predicted_qr_tolerance,
                    "parent_clip_aesthetic": (
                        stage1_recommendation.predicted_clip_aesthetic
                    ),
                    "parent_clip_score": stage1_recommendation.predicted_clip_score,
                    "parent_hpsv2_1": stage1_recommendation.predicted_hpsv2_1,
                    "parent_saturation_risk": (
                        stage1_recommendation.predicted_saturation_risk
                    ),
                },
            )
            for stage2_index, (stage2_profile, stage2_recommendation) in enumerate(
                _stage2_profiles(stage2_ranked, stage2_top_k), start=1
            ):
                chain_id = f"advisor-s1-{stage1_index:02d}-s2-{stage2_index:02d}"
                stage2_runtime_id = f"e028_a_s1_{stage1_index:02d}_s2_{stage2_index:02d}"
                stage2_method = _runtime_method(
                    stage2_recommendation.candidate.configuration,
                    runtime_id=stage2_runtime_id,
                    name=f"E028 advisor Stage 2 {stage1_profile}/{stage2_profile}",
                    output_variant="srpg",
                )
                stage2_methods.append(stage2_method)
                prompt_predictions.append(
                    _prediction(
                        stage2_recommendation,
                        prompt=prompt,
                        runtime_method=stage2_method,
                        source_method_id=stage2_recommendation.candidate.method_id,
                        role="e028_advisor_stage2",
                        pipeline_state="stage2",
                        chain_id=chain_id,
                        profile=f"{stage1_profile}/{stage2_profile}",
                        parent_stage1_method_id=stage1_runtime_id,
                    )
                )
                srmpgd_candidates = [
                    _compose_srmpgd(stage2_recommendation.candidate, item)
                    for item in pools.srmpgd
                ]
                srmpgd_ranked = _recommend(
                    advisor,
                    prompt=prompt.text,
                    candidates=srmpgd_candidates,
                    prompt_embedding=embedding,
                    payload_length=len(payload),
                    error_correction=error_correction,
                    qr_context=context,
                    threshold=scan_probability_threshold,
                    context_features={
                        "parent_qr_success": (
                            stage2_recommendation.predicted_qr_success
                        ),
                        "parent_qr_tolerance": (
                            stage2_recommendation.predicted_qr_tolerance
                        ),
                        "parent_clip_aesthetic": (
                            stage2_recommendation.predicted_clip_aesthetic
                        ),
                        "parent_clip_score": (
                            stage2_recommendation.predicted_clip_score
                        ),
                        "parent_hpsv2_1": stage2_recommendation.predicted_hpsv2_1,
                        "parent_saturation_risk": (
                            stage2_recommendation.predicted_saturation_risk
                        ),
                    },
                )
                srmpgd_recommendation = srmpgd_ranked[0]
                srmpgd_runtime_id = f"{stage2_runtime_id}_mpgd"
                srmpgd_method = _runtime_method(
                    srmpgd_recommendation.candidate.configuration,
                    runtime_id=srmpgd_runtime_id,
                    name=f"E028 advisor SR-MPGD {stage1_profile}/{stage2_profile}",
                    output_variant="srmpgd",
                )
                srmpgd_method["tools"]["settings"]["srmpgd_min_qr_tolerance"] = float(
                    qr_tolerance_threshold
                )
                srmpgd_methods.append(srmpgd_method)
                prompt_predictions.append(
                    _prediction(
                        srmpgd_recommendation,
                        prompt=prompt,
                        runtime_method=srmpgd_method,
                        source_method_id=srmpgd_recommendation.candidate.method_id,
                        role="e028_advisor_srmpgd",
                        pipeline_state="srmpgd",
                        chain_id=chain_id,
                        profile=f"{stage1_profile}/{stage2_profile}/srmpgd",
                        parent_stage1_method_id=stage1_runtime_id,
                        parent_stage2_method_id=stage2_runtime_id,
                    )
                )
        methods = [*stage1_methods, *stage2_methods, *srmpgd_methods]
        if len(methods) > 25:
            raise ValueError(f"E028 prompt {prompt.id} has {len(methods)} methods > 25")
        request = LabCampaignCreate.model_validate(
            {
                "name": f"E028 hierarchical {prompt_index:03d} {prompt.id}",
                "payload": payload,
                "error_correction": error_correction,
                "prompts": [prompt.model_dump(mode="json")],
                "seeds": list(normalized_seeds),
                "methods": methods,
                "max_attempts": 1,
            }
        ).model_dump(mode="json")
        campaigns.append(request)
        prompt_predictions.sort(
            key=lambda row: (
                E028_PIPELINE_STATES.index(str(row["pipeline_state"])),
                str(row["plan_method_id"]),
            )
        )
        predictions.extend(prompt_predictions)
        public_campaigns.append(
            {
                "name": request["name"],
                "prompt_id": prompt.id,
                "method_count": len(methods),
                "trial_count": len(methods) * len(normalized_seeds),
            }
        )

    prediction_sha256 = hashlib.sha256(
        _canonical_json(predictions).encode("utf-8")
    ).hexdigest()
    plan_material = {
        "protocol": "e028-v4-plan-bound-strict-pairing-chain",
        "advisor_sha256": advisor_sha256,
        "prediction_sha256": prediction_sha256,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "payload_length": len(payload),
        "error_correction": error_correction,
        "qr_context": context,
        "seeds": list(normalized_seeds),
        "stage1_top_k": stage1_top_k,
        "stage2_top_k": stage2_top_k,
        "scan_probability_threshold": scan_probability_threshold,
        "qr_tolerance_threshold": qr_tolerance_threshold,
        "saturation_threshold": saturation_threshold,
        "prompts": [item.model_dump(mode="json") for item in validated_prompts],
        "selection": [
            {
                "prompt_id": prompt.id,
                "methods": [
                    {
                        "method_id": row["plan_method_id"],
                        "state": row["pipeline_state"],
                        "chain_id": row["chain_id"],
                        "candidate_signature": row["candidate_signature"],
                    }
                    for row in predictions
                    if row["prompt_id"] == prompt.id
                ],
            }
            for prompt in validated_prompts
        ],
    }
    plan_id = hashlib.sha256(_canonical_json(plan_material).encode("utf-8")).hexdigest()[:16]
    public = {
        **plan_material,
        "plan_id": plan_id,
        "prompt_count": len(validated_prompts),
        "seed_count": len(normalized_seeds),
        "context_count": len(validated_prompts) * len(normalized_seeds),
        "campaign_count": len(campaigns),
        "trial_count": sum(item["trial_count"] for item in public_campaigns),
        "candidate_pool_counts": {
            "stage1": len(pools.stage1),
            "stage2": len(pools.stage2),
            "srmpgd": len(pools.srmpgd),
        },
        "campaigns": public_campaigns,
    }
    return AdvisorInferencePlan(
        plan_id=plan_id,
        payload=payload,
        campaigns=tuple(campaigns),
        predictions=tuple(predictions),
        public=public,
    )


def _candidate_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
    success = _finite(row.get("qr_success")) or 0.0
    tolerance = _finite(row.get("qr_tolerance")) or 0.0
    saturation = _finite(row.get("saturation_risk")) or 0.0
    return (
        success >= 0.5,
        tolerance,
        saturation <= 0.05,
        _finite(row.get("hpsv2_1")) or -math.inf,
        _finite(row.get("clip_aesthetic")) or -math.inf,
        _finite(row.get("clip_score")) or -math.inf,
        -saturation,
    )


def _generated(row: Mapping[str, Any] | None) -> bool:
    return bool(
        row is not None
        and _finite(row.get("qr_success")) is not None
        and str(row.get("status")) not in {"error", "cancelled", "interrupted"}
    )


def _output_contract_valid(row: Mapping[str, Any] | None) -> bool:
    """Reject a raw Stage 1 raster masquerading as a later pipeline state."""

    if row is None:
        return False
    state = str(row.get("pipeline_state") or "")
    output = str(row.get("output_variant") or "")
    if state == "stage1":
        return output == "raw"
    if state == "stage2":
        return output == "srpg"
    if state == "srmpgd":
        return output in {"srpg", "srmpgd"}
    return False


def _deliverable(
    row: Mapping[str, Any] | None,
    threshold: float,
    saturation_threshold: float = 0.05,
) -> bool:
    return bool(
        _generated(row)
        and _output_contract_valid(row)
        and (_finite(row.get("qr_success")) or 0.0) >= 0.5
        and (_finite(row.get("qr_tolerance")) or 0.0) >= threshold
        and (_finite(row.get("saturation_risk")) or 0.0) <= saturation_threshold
    )


def _select(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    values = [
        dict(row)
        for row in rows
        if _generated(row) and _output_contract_valid(row)
    ]
    return max(values, key=_candidate_rank) if values else None


def audit_e028_pairing(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prove Stage 1 image and Stage 2 latent reuse from exported provenance."""

    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for source in entries:
        if str(source.get("role", "")).startswith("e028_"):
            key = (str(source.get("prompt_id")), _finite(source.get("seed")))
            groups.setdefault(key, []).append(dict(source))
    audits = []
    for (prompt_id, seed), rows in sorted(groups.items()):
        by_method = {str(row.get("method_id")): row for row in rows}
        by_run = {
            str(row.get("generation_run_id")): row
            for row in rows
            if row.get("generation_run_id")
        }
        for row in rows:
            state = str(row.get("pipeline_state"))
            if state == "stage1":
                continue
            technically_generated = _generated(row)
            output_contract_valid = _output_contract_valid(row)
            source_stage1 = by_run.get(str(row.get("stage1_source_run_id") or ""))
            source_stage1_hash = str(
                (source_stage1 or {}).get("final_image_sha256") or ""
            )
            current_stage1_hash = str(row.get("stage1_image_sha256") or "")
            expected_stage1 = str(row.get("parent_stage1_method_id") or "")
            stage1_source_found = source_stage1 is not None
            stage1_reused_marker = _finite(row.get("stage1_reused")) == 1.0
            stage1_method_match = bool(
                source_stage1
                and source_stage1.get("method_id") == expected_stage1
            )
            stage1_hash_match = bool(
                current_stage1_hash
                and source_stage1_hash
                and current_stage1_hash == source_stage1_hash
            )
            stage1_exact = bool(
                technically_generated
                and output_contract_valid
                and stage1_source_found
                and stage1_reused_marker
                and stage1_hash_match
            )
            stage2_exact = None
            stage2_source_found = None
            stage2_status_exact = None
            stage2_marker_exact = None
            stage2_latent_present = None
            stage2_method_match = None
            stage2_latent_match = None
            if state == "srmpgd":
                expected_stage2 = str(row.get("parent_stage2_method_id") or "")
                source_stage2 = by_run.get(str(row.get("stage2_source_run_id") or ""))
                source_latent = str(
                    (source_stage2 or {}).get("stage2_latent_sha256")
                    or (source_stage2 or {}).get("stage2_source_latent_sha256")
                    or ""
                )
                reused_latent = str(row.get("stage2_source_latent_sha256") or "")
                current_latent = str(row.get("stage2_latent_sha256") or "")
                stage2_source_found = source_stage2 is not None
                stage2_status_exact = (
                    str(row.get("stage2_pairing_status")) == "exact_reuse"
                )
                stage2_marker_exact = (
                    _finite(row.get("stage2_pairing_exact")) == 1.0
                )
                stage2_latent_present = bool(reused_latent)
                stage2_method_match = bool(
                    source_stage2
                    and source_stage2.get("method_id") == expected_stage2
                )
                stage2_latent_match = bool(
                    source_latent
                    and reused_latent
                    and current_latent
                    and source_latent == reused_latent == current_latent
                )
                stage2_exact = bool(
                    technically_generated
                    and output_contract_valid
                    and stage2_source_found
                    and stage2_status_exact
                    and stage2_marker_exact
                    and stage2_latent_present
                    and stage2_latent_match
                )
            failures = []
            for failed, name in [
                (not output_contract_valid, "output_contract"),
                (not stage1_source_found, "stage1_source_missing"),
                (not stage1_reused_marker, "stage1_reused_marker"),
                (
                    not current_stage1_hash or not source_stage1_hash,
                    "stage1_hash_missing",
                ),
                (
                    bool(current_stage1_hash and source_stage1_hash)
                    and not stage1_hash_match,
                    "stage1_hash_mismatch",
                ),
                (state == "srmpgd" and not stage2_source_found, "stage2_source_missing"),
                (state == "srmpgd" and not stage2_status_exact, "stage2_status"),
                (state == "srmpgd" and not stage2_marker_exact, "stage2_marker"),
                (state == "srmpgd" and not stage2_latent_present, "stage2_latent_missing"),
                (
                    state == "srmpgd"
                    and not stage2_latent_match,
                    "stage2_latent_mismatch",
                ),
            ]:
                if failed:
                    failures.append(name)
            audits.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "method_id": row.get("method_id"),
                    "pipeline_state": state,
                    "technically_generated": technically_generated,
                    "output_variant": row.get("output_variant"),
                    "output_contract_valid": output_contract_valid,
                    "stage1_exact_reuse": stage1_exact,
                    "stage1_source_found": stage1_source_found,
                    "stage1_reused_marker": stage1_reused_marker,
                    "stage1_method_match": stage1_method_match,
                    "stage1_hash_match": stage1_hash_match,
                    "stage1_image_sha256": current_stage1_hash or None,
                    "stage1_source_image_sha256": source_stage1_hash or None,
                    "stage2_exact_reuse": stage2_exact,
                    "stage2_source_found": stage2_source_found,
                    "stage2_status_exact": stage2_status_exact,
                    "stage2_marker_exact": stage2_marker_exact,
                    "stage2_latent_present": stage2_latent_present,
                    "stage2_method_match": stage2_method_match,
                    "stage2_latent_match": stage2_latent_match,
                    "failure_reasons": "|".join(failures),
                    "complete": (
                        stage1_exact and (stage2_exact is not False)
                        if technically_generated
                        else None
                    ),
                    "method_present": row.get("method_id") in by_method,
                }
            )
    return audits


def audit_srmpgd_iteration_zero_raster(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove that an SR-MPGD no-op returns the exact Stage 2 raster.

    The Stage 2 latent is not an exact representation of its PNG: decoding it again can
    introduce VAE reconstruction errors.  Therefore an SR-MPGD result that selects
    iteration zero must reuse the original Stage 2 raster, not a fresh VAE decode.
    """

    by_run = {
        str(row.get("generation_run_id")): dict(row)
        for row in entries
        if row.get("generation_run_id")
    }
    audits = []
    for source in entries:
        row = dict(source)
        if str(row.get("pipeline_state")) != "srmpgd":
            continue
        iteration = _finite(row.get("srmpgd_selected_iteration"))
        if iteration != 0.0:
            continue
        parent = by_run.get(str(row.get("stage2_source_run_id") or ""))
        parent_hash = str((parent or {}).get("final_image_sha256") or "")
        final_hash = str(row.get("final_image_sha256") or "")
        backend_stage2_hash = str(
            row.get("srmpgd_stage2_image_sha256") or ""
        )
        backend_selected_hash = str(
            row.get("srmpgd_selected_image_sha256") or ""
        )
        backend_marker = _finite(row.get("srmpgd_iteration_zero_exact"))
        backend_hashes_exact = bool(
            backend_stage2_hash
            and backend_selected_hash
            and final_hash
            and backend_stage2_hash == backend_selected_hash == final_hash
        )
        parent_crosscheck = bool(parent_hash and parent_hash == final_hash)
        exact = bool(
            _generated(row)
            and _output_contract_valid(row)
            and backend_marker == 1.0
            and backend_hashes_exact
        )
        audits.append(
            {
                "prompt_id": row.get("prompt_id"),
                "seed": _finite(row.get("seed")),
                "method_id": row.get("method_id"),
                "stage2_source_run_id": row.get("stage2_source_run_id"),
                "parent_stage2_method_id": (parent or {}).get("method_id"),
                "output_variant": row.get("output_variant"),
                "stage2_image_sha256": backend_stage2_hash or None,
                "srmpgd_image_sha256": backend_selected_hash or None,
                "final_image_sha256": final_hash or None,
                "parent_stage2_image_sha256": parent_hash or None,
                "backend_hashes_exact": backend_hashes_exact,
                "parent_crosscheck": parent_crosscheck,
                "backend_iteration_zero_exact": backend_marker,
                "exact": exact,
            }
        )
    return audits


def evaluate_e028_policies(
    entries: Sequence[Mapping[str, Any]],
    *,
    qr_tolerance_threshold: float = 0.80,
    saturation_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare fixed and advisor cascades; Stage 1 can never be selected."""

    if not 0.0 <= qr_tolerance_threshold <= 1.0:
        raise ValueError("qr_tolerance_threshold must be between zero and one")
    if not 0.0 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be between zero and one")

    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for source in entries:
        if str(source.get("role", "")).startswith("e028_"):
            key = (str(source.get("prompt_id")), _finite(source.get("seed")))
            groups.setdefault(key, []).append(dict(source))
    decisions: list[dict[str, Any]] = []
    for (prompt_id, seed), rows in sorted(groups.items()):
        fixed = [row for row in rows if row.get("fixed_control")]
        advisor_rows = [row for row in rows if not row.get("fixed_control")]
        fixed_stage2 = next((row for row in fixed if row.get("pipeline_state") == "stage2"), None)
        fixed_srmpgd = next((row for row in fixed if row.get("pipeline_state") == "srmpgd"), None)
        fixed_selected = (
            fixed_stage2
            if _deliverable(
                fixed_stage2, qr_tolerance_threshold, saturation_threshold
            )
            else _select([row for row in [fixed_stage2, fixed_srmpgd] if row])
        )
        chains: dict[str, list[dict[str, Any]]] = {}
        for row in advisor_rows:
            if row.get("pipeline_state") in {"stage2", "srmpgd"}:
                chains.setdefault(str(row.get("chain_id")), []).append(row)
        ordered_chains = sorted(chains)
        top1_rows = chains.get(ordered_chains[0], []) if ordered_chains else []
        top1_stage2 = next(
            (row for row in top1_rows if row.get("pipeline_state") == "stage2"), None
        )
        top1_srmpgd = next(
            (row for row in top1_rows if row.get("pipeline_state") == "srmpgd"), None
        )
        top1_selected = (
            top1_stage2
            if _deliverable(
                top1_stage2, qr_tolerance_threshold, saturation_threshold
            )
            else _select([row for row in [top1_stage2, top1_srmpgd] if row])
        )
        best_candidates = []
        best_units = len(
            {
                row.get("parent_stage1_method_id")
                for row in advisor_rows
                if row.get("pipeline_state") == "stage2"
            }
        )
        for chain_rows in chains.values():
            stage2 = next(
                (row for row in chain_rows if row.get("pipeline_state") == "stage2"), None
            )
            srmpgd = next(
                (row for row in chain_rows if row.get("pipeline_state") == "srmpgd"), None
            )
            best_units += 1
            if _deliverable(stage2, qr_tolerance_threshold, saturation_threshold):
                if stage2:
                    best_candidates.append(stage2)
            else:
                best_units += 1
                best_candidates.extend(row for row in [stage2, srmpgd] if row)
        best_selected = _select(best_candidates)
        selections = {
            "fixed_cascade": (
                fixed_selected,
                2
                + int(
                    not _deliverable(
                        fixed_stage2, qr_tolerance_threshold, saturation_threshold
                    )
                ),
            ),
            "advisor_top1": (
                top1_selected,
                2
                + int(
                    not _deliverable(
                        top1_stage2, qr_tolerance_threshold, saturation_threshold
                    )
                ),
            ),
            "advisor_best_of_chains": (best_selected, best_units),
        }
        for policy, (selected, units) in selections.items():
            decisions.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "policy": policy,
                    "selected": selected is not None,
                    "selected_state": selected.get("pipeline_state") if selected else None,
                    "chain_id": selected.get("chain_id") if selected else None,
                    "method_id": selected.get("method_id") if selected else None,
                    "generation_run_id": (
                        selected.get("generation_run_id") if selected else None
                    ),
                    "qr_success": _finite(selected.get("qr_success")) if selected else 0.0,
                    "qr_tolerance": (
                        _finite(selected.get("qr_tolerance")) if selected else 0.0
                    ),
                    "deliverable": _deliverable(
                        selected, qr_tolerance_threshold, saturation_threshold
                    ),
                    "clip_aesthetic": (
                        _finite(selected.get("clip_aesthetic")) if selected else None
                    ),
                    "clip_score": _finite(selected.get("clip_score")) if selected else None,
                    "hpsv2_1": _finite(selected.get("hpsv2_1")) if selected else None,
                    "saturation_risk": (
                        _finite(selected.get("saturation_risk")) if selected else None
                    ),
                    "estimated_generation_units": units,
                    "stage1_was_delivered": False,
                }
            )

    def average(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
        values = [value for row in rows if (value := _finite(row.get(name))) is not None]
        return sum(values) / len(values) if values else None

    policies = {}
    for policy in E028_POLICIES:
        rows = [row for row in decisions if row["policy"] == policy]
        delivered = [row for row in rows if row["deliverable"]]
        prompt_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            prompt_groups.setdefault(str(row["prompt_id"]), []).append(row)
        policies[policy] = {
            "contexts": len(rows),
            "exact_qr_successes": sum(
                (_finite(row.get("qr_success")) or 0.0) >= 0.5 for row in rows
            ),
            "delivery_gate_successes": len(delivered),
            "delivery_gate_success_rate": len(delivered) / len(rows) if rows else 0.0,
            "prompts": len(prompt_groups),
            "prompts_all_seeds_deliverable": sum(
                all(bool(row["deliverable"]) for row in values)
                for values in prompt_groups.values()
            ),
            "mean_qr_tolerance": average(rows, "qr_tolerance"),
            "mean_clip_aesthetic_delivered": average(delivered, "clip_aesthetic"),
            "mean_clip_score_delivered": average(delivered, "clip_score"),
            "mean_hpsv2_1_delivered": average(delivered, "hpsv2_1"),
            "mean_saturation_risk_delivered": average(delivered, "saturation_risk"),
            "estimated_generation_units": sum(row["estimated_generation_units"] for row in rows),
            "selected_state_counts": dict(Counter(row["selected_state"] for row in rows)),
            "stage1_deliveries": sum(bool(row["stage1_was_delivered"]) for row in rows),
        }
    return {
        "protocol": "e028-v4-plan-bound-strict-pairing-chain",
        "qr_tolerance_threshold": qr_tolerance_threshold,
        "saturation_threshold": saturation_threshold,
        "contexts": len(groups),
        "policies": policies,
        "decisions": decisions,
    }


def _prompt_features(prompt: str) -> dict[str, float]:
    words = re.findall(r"\w+", prompt, flags=re.UNICODE)
    return {
        "prompt_characters": float(len(prompt)),
        "prompt_words": float(len(words)),
        "prompt_unique_word_ratio": float(
            len({word.lower() for word in words}) / max(1, len(words))
        ),
        "prompt_commas": float(prompt.count(",")),
    }


def build_e028_conditional_datasets(
    entries: Sequence[Mapping[str, Any]],
    *,
    prompt_embedding_provider: Callable[[str], Sequence[float]] | None = None,
    qr_tolerance_threshold: float = 0.80,
    saturation_threshold: float = 0.05,
) -> dict[str, AdvisorDataset]:
    """Build next-generation Stage 2/SR-MPGD datasets with parent diagnostics."""

    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for source in entries:
        if str(source.get("role", "")).startswith("e028_"):
            key = (str(source.get("prompt_id")), _finite(source.get("seed")))
            groups.setdefault(key, []).append(dict(source))
    records = {"stage2": [], "srmpgd": []}
    candidates: dict[str, dict[str, RecipeCandidate]] = {"stage2": {}, "srmpgd": {}}
    for (prompt_id, seed), rows in groups.items():
        by_method = {str(row.get("method_id")): row for row in rows}
        prompt = str(next((row.get("prompt_text") for row in rows if row.get("prompt_text")), ""))
        if not prompt:
            continue
        embedding = prompt_embedding_provider(prompt) if prompt_embedding_provider else None
        for row in rows:
            state = str(row.get("pipeline_state"))
            if (
                state not in records
                or not _generated(row)
                or not _output_contract_valid(row)
            ):
                continue
            parent_id = (
                row.get("parent_stage1_method_id")
                if state == "stage2"
                else row.get("parent_stage2_method_id")
            )
            parent = by_method.get(str(parent_id))
            if not _generated(parent) or not _output_contract_valid(parent):
                continue
            context: dict[str, Any] = {
                **_prompt_features(prompt),
                "payload_length": _finite(row.get("payload_length")) or 0.0,
                "error_correction": row.get("error_correction") or "M",
                "parent_qr_success": _finite(parent.get("qr_success")) or 0.0,
                "parent_qr_tolerance": _finite(parent.get("qr_tolerance")) or 0.0,
                "parent_clip_aesthetic": _finite(parent.get("clip_aesthetic")) or 0.0,
                "parent_clip_score": _finite(parent.get("clip_score")) or 0.0,
                "parent_hpsv2_1": _finite(parent.get("hpsv2_1")) or 0.0,
                "parent_saturation_risk": _finite(parent.get("saturation_risk")) or 0.0,
                "parent_module_error_rate": _finite(parent.get("module_error_rate")) or 0.0,
            }
            if embedding is not None:
                context.update(
                    {
                        f"prompt_embedding_{index:03d}": float(value)
                        for index, value in enumerate(np.asarray(embedding).reshape(-1))
                    }
                )
            configuration = deepcopy(row.get("candidate_configuration") or {})
            if not configuration:
                continue
            target = {
                "qr_success": float(
                    _deliverable(
                        row, qr_tolerance_threshold, saturation_threshold
                    )
                ),
                "qr_tolerance": _finite(row.get("qr_tolerance")) or 0.0,
                "clip_aesthetic": _finite(row.get("clip_aesthetic")) or 0.0,
                "clip_score": _finite(row.get("clip_score")) or 0.0,
                "hpsv2_1": _finite(row.get("hpsv2_1")) or 0.0,
                "saturation_risk": _finite(row.get("saturation_risk")) or 0.0,
                "duration_ms": _finite(row.get("duration_ms")) or 0.0,
            }
            group_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            record = AdvisorRecord(
                trial_id=str(row.get("trial_id")),
                prompt_id=prompt_id,
                prompt_text=prompt,
                group_id=group_id,
                context_features=context,
                parameters=configuration,
                targets=target,
                metadata={
                    "seed": seed,
                    "generation_run_id": row.get("generation_run_id"),
                    "parent_generation_run_id": parent.get("generation_run_id"),
                    "chain_id": row.get("chain_id"),
                },
            )
            records[state].append(record)
            candidate = _recipe(
                configuration,
                str(row.get("source_method_id") or row.get("method_id")),
            )
            candidates[state][candidate.signature] = candidate
    result = {}
    for state in ("stage2", "srmpgd"):
        values = records[state]
        successes = sum(record.targets["qr_success"] >= 0.5 for record in values)
        result[state] = AdvisorDataset(
            records=values,
            candidates=list(candidates[state].values()),
            audit={
                "stage": state,
                "usable_rows": len(values),
                "prompt_groups": len({record.group_id for record in values}),
                "recipes": len(candidates[state]),
                "robust_successes": successes,
                "robust_failures": len(values) - successes,
                "parent_diagnostics_in_features": True,
            },
        )
    return result
