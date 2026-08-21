from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from .advisor_inference import AdvisorInferencePlan
from .e028_hierarchical import partition_e028_candidates
from .lab import DIFFQRCODER_MODEL_SETTINGS, laboratory_profiles
from .parameter_advisor import ParameterRecommendation, RecipeCandidate
from .policy import (
    ConservativeDeliveryGate,
    Stage2CandidateAssessment,
    assess_stage2_candidate,
)
from .schemas import LabCampaignCreate, LabMethod, LabPrompt

E031_EXPERIMENT = "e031-prospective-stage2-holdout-v1"
E031_PROTOCOL = "e031-v1-static-paired-three-stage2-branches"
E031_PRIMARY_SEED = 1_310_001
E031_RETRY_SEED = 1_310_002
E031_QR_TOLERANCE_THRESHOLD = 0.80
E031_STRICT_QR_TOLERANCE_THRESHOLD = 0.95
E031_STRICT_EXACT_PRESETS = 36
E031_QR_VERIFY_PRESET_COUNT = 37
E031_QR_VERIFY_REPETITIONS = 5
E031_SATURATION_THRESHOLD = 0.05
E031_BRANCHES = (
    "fixed_seed_a",
    "advisor_seed_a",
    "fixed_seed_b",
)
E031_POLICIES: dict[str, tuple[str, ...]] = {
    "fixed_seed_a": ("fixed_seed_a",),
    "advisor_seed_a": ("advisor_seed_a",),
    "fixed_then_advisor_seed_a": (
        "fixed_seed_a",
        "advisor_seed_a",
    ),
    "fixed_seed_retry": (
        "fixed_seed_a",
        "fixed_seed_b",
    ),
    "fixed_advisor_then_seed_retry": E031_BRANCHES,
    "best_of_three": E031_BRANCHES,
}

_CASCADE_POLICIES = frozenset(E031_POLICIES) - {"best_of_three"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_E031_MODEL_SETTINGS = dict(DIFFQRCODER_MODEL_SETTINGS)
_E031_BRANCH_SEEDS = {
    "fixed_seed_a": E031_PRIMARY_SEED,
    "advisor_seed_a": E031_PRIMARY_SEED,
    "fixed_seed_b": E031_RETRY_SEED,
}
_RECIPE_IDENTITY_FIELDS = frozenset({"id", "name"})
_DEFAULT_NEGATIVE_PROMPT = (
    "text, letters, watermark, logo, barcode, oversaturated colors, "
    "blown highlights, muddy details"
)

# This is deliberately a finite, frozen bank.  If one of these texts has already
# been used, the caller must stop and version a new bank instead of silently
# substituting another prompt and changing the experiment after the fact.
_SIMPLE_PROMPTS = (
    (
        "A single amber glass chess knight on pale travertine, restrained product "
        "photograph, soft north light, generous negative space, no text or typography."
    ),
    (
        "A folded indigo umbrella beside a rain-speckled window, quiet editorial "
        "photograph, natural materials, no text or typography."
    ),
    (
        "A copper watering can on a weathered garden bench, balanced daylight "
        "photograph, one clear subject, no text or typography."
    ),
    (
        "A pair of ivory ice skates on a muted blue floor, museum-like still life, "
        "controlled contrast, no text or typography."
    ),
    (
        "A charcoal fountain pen resting on handmade cream paper, precise macro "
        "photograph, subtle shadows, no text or typography."
    ),
    (
        "A yellow camping lantern inside a calm canvas tent, believable evening light, "
        "uncluttered composition, no text or typography."
    ),
    (
        "A carved wooden duck on a grey linen shelf, understated catalogue photograph, "
        "coherent proportions, no text or typography."
    ),
    (
        "A violet laboratory flask containing one fern leaf, clean studio photograph, "
        "soft reflections, no text or typography."
    ),
    (
        "A small bronze handbell on dark green velvet, classic still life, focused warm "
        "light, no text or typography."
    ),
    (
        "A coral-red table radio in a pale plywood alcove, modern editorial photograph, "
        "simple geometry, no text or typography."
    ),
    (
        "A white porcelain mortar and pestle on black slate, calm culinary photograph, "
        "crisp material detail, no text or typography."
    ),
    (
        "A moss-green binocular case on a sand-colored stool, outdoor equipment "
        "photograph, diffused daylight, no text or typography."
    ),
    (
        "A translucent blue marble in a shallow wooden dish, minimal close-up "
        "photograph, realistic caustics, no text or typography."
    ),
    (
        "A handwoven straw sunhat hanging on a limewashed wall, quiet summer "
        "photograph, gentle shadows, no text or typography."
    ),
    (
        "A plum-colored violin bow in an open maple case, refined still life, controlled "
        "warm illumination, no text or typography."
    ),
    (
        "A compact stainless steel moka pot on a terracotta tile, natural kitchen "
        "photograph, coherent reflections, no text or typography."
    ),
    (
        "A single sea-green typewriter key displayed under glass, archival museum "
        "photograph, ample empty space, no text or typography."
    ),
    (
        "A paper kite with orange tails leaning against a pale concrete wall, airy "
        "editorial photograph, no text or typography."
    ),
    (
        "A black ceramic moon jar on a low oak plinth, gallery photograph, balanced "
        "symmetry, no text or typography."
    ),
    (
        "A raspberry-red climbing helmet on folded canvas, equipment catalogue "
        "photograph, clean edges, no text or typography."
    ),
)

_ATYPICAL_PROMPTS = (
    (
        "A miniature desert suspended inside a transparent cello, cinematic impossible "
        "still life, coherent glass and sand, intricate detail, no text or typography."
    ),
    (
        "A library grown from interlocking seashells beneath green water, believable "
        "impossible architecture, controlled light, no text or typography."
    ),
    (
        "A midnight carousel made of folded auroras orbiting a stone seed, detailed "
        "gouache scene, coherent geometry, no text or typography."
    ),
    (
        "A silent brass orchestra nesting inside a giant pomegranate, surreal editorial "
        "photograph, natural material detail, no text or typography."
    ),
    (
        "A glacier shaped as an open mechanical pocketbook crossing a lavender plain, "
        "cinematic scene, believable depth, no text or typography."
    ),
    (
        "A translucent fox carrying a greenhouse of red moss on its back, museum "
        "diorama, coherent anatomy and reflections, no text or typography."
    ),
    (
        "An underwater observatory built from stacked teacups and basalt, architectural "
        "visualization, controlled contrast, no text or typography."
    ),
    (
        "A flock of ceramic umbrellas migrating through a candlelit tunnel, surreal "
        "photograph, consistent perspective, no text or typography."
    ),
    (
        "A tiny railway station turning slowly inside a polished pearl, macro fantasy "
        "photograph, intricate coherent detail, no text or typography."
    ),
    (
        "A canyon of blue fabric sewing itself around a levitating compass, cinematic "
        "impossible workshop, no text or typography."
    ),
    (
        "A winter garden folded into the shadow of a copper key, poetic museum "
        "installation, believable lighting, no text or typography."
    ),
    (
        "A glass lighthouse illuminating an inverted coral mountain above the clouds, "
        "restrained fantasy painting, no text or typography."
    ),
    (
        "A clockwork heron assembling a river from silver ribbons, detailed editorial "
        "illustration, coherent motion, no text or typography."
    ),
    (
        "A volcanic reading room balanced inside a hollow snowflake, architectural "
        "fantasy, controlled palette, no text or typography."
    ),
    (
        "A procession of luminous mushrooms carrying a miniature suspension bridge, "
        "nocturnal diorama, natural textures, no text or typography."
    ),
    (
        "An origami submarine cultivating an orchard of tiny moons, cinematic "
        "underwater scene, coherent impossible geometry, no text or typography."
    ),
    (
        "A marble beehive projecting constellations into an empty theatre, museum "
        "installation photograph, no text or typography."
    ),
    (
        "A velvet tornado carefully sorting porcelain fruit in a quiet pantry, surreal "
        "editorial scene, controlled contrast, no text or typography."
    ),
    (
        "A floating observatory carved from black ice and inhabited by butterflies, "
        "atmospheric architectural rendering, no text or typography."
    ),
    (
        "A mechanical lotus unfolding an entire rainy street from its petals, detailed "
        "cinematic scene, believable depth, no text or typography."
    ),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _safe_identifier(value: Any, maximum: int = 52) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return (normalized or "recipe")[:maximum]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _strict_bool(value: Any) -> bool | None:
    if value is True or value == 1 or str(value).strip().casefold() in {"true", "1"}:
        return True
    if value is False or value == 0 or str(value).strip().casefold() in {"false", "0"}:
        return False
    return None


def _configuration(value: Any) -> dict[str, Any]:
    candidate = getattr(value, "candidate", None)
    if candidate is not None:
        value = candidate
    configuration = getattr(value, "configuration", None)
    if configuration is not None:
        return deepcopy(dict(configuration))
    if isinstance(value, Mapping) and isinstance(value.get("configuration"), Mapping):
        return deepcopy(dict(value["configuration"]))
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise TypeError(f"recipe configuration must be a mapping, got {type(value)!r}")


def _e031_model_mismatches(configuration: Mapping[str, Any]) -> list[str]:
    model = configuration.get("model")
    if not isinstance(model, Mapping):
        return ["model"]
    mismatches = [
        f"model.{key}"
        for key, expected in _E031_MODEL_SETTINGS.items()
        if key not in model or model.get(key) != expected
    ]
    mismatches.extend(
        f"model.{key} (unexpected)"
        for key in sorted(set(model) - set(_E031_MODEL_SETTINGS))
    )
    return mismatches


def _require_e031_model(configuration: Mapping[str, Any], *, context: str) -> None:
    mismatches = _e031_model_mismatches(configuration)
    if mismatches:
        raise ValueError(
            f"{context} does not use the frozen E031 DiffQRCoder stack: "
            f"{', '.join(sorted(mismatches))}"
        )


def _e031_candidates(values: Sequence[RecipeCandidate]) -> tuple[RecipeCandidate, ...]:
    """Keep only recipes carrying every frozen model and geometry value."""

    return tuple(
        candidate
        for candidate in values
        if not _e031_model_mismatches(candidate.configuration)
    )


def _recipe_identity(configuration: Any) -> str | None:
    """Fingerprint generation math while ignoring display identifiers only."""

    if not isinstance(configuration, Mapping):
        return None
    material = deepcopy(dict(configuration))
    for key in _RECIPE_IDENTITY_FIELDS:
        material.pop(key, None)
    return _sha256_json(material)


def _source_method_id(value: Any, fallback: str) -> str:
    candidate = getattr(value, "candidate", None)
    if candidate is not None:
        value = candidate
    method_id = getattr(value, "method_id", None)
    if method_id:
        return str(method_id)
    if isinstance(value, Mapping):
        return str(value.get("source_method_id") or value.get("method_id") or fallback)
    return fallback


def _prediction_values(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        raw = value.to_dict()
    elif is_dataclass(value):
        raw = asdict(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError(f"prediction must be a mapping, got {type(value)!r}")
    candidate = raw.pop("candidate", None)
    if isinstance(candidate, Mapping):
        raw.setdefault("candidate_signature", candidate.get("signature"))
        raw.setdefault("candidate_observations", candidate.get("observations"))
    return raw


def build_e031_holdout_prompts(
    count: int = 40,
    *,
    excluded_prompt_texts: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Return the frozen, balanced E031 prompt bank.

    The experiment is stratified one-for-one between ordinary and atypical
    prompts.  An overlap is a hard error: replacing a prompt dynamically would
    change the holdout without changing the protocol name.
    """

    if count < 2 or count > 40 or count % 2:
        raise ValueError("E031 prompt count must be even and between 2 and 40")
    excluded = {_normalized_prompt(value) for value in excluded_prompt_texts}
    per_family = count // 2
    prompts: list[dict[str, str]] = []
    for index in range(per_family):
        ordinal = index * 2 + 1
        prompts.append(
            {
                "id": f"e031h_simple_{ordinal:03d}",
                "text": _SIMPLE_PROMPTS[index],
                "negative_prompt": _DEFAULT_NEGATIVE_PROMPT,
            }
        )
        prompts.append(
            {
                "id": f"e031h_atypical_{ordinal + 1:03d}",
                "text": _ATYPICAL_PROMPTS[index],
                "negative_prompt": _DEFAULT_NEGATIVE_PROMPT,
            }
        )
    normalized = [_normalized_prompt(item["text"]) for item in prompts]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("the frozen E031 bank contains duplicate prompt texts")
    overlap = [item["id"] for item in prompts if _normalized_prompt(item["text"]) in excluded]
    if overlap:
        raise ValueError(f"E031 holdout overlaps prior/training prompts: {overlap}")
    return prompts


def _effective_e031_candidate_material(
    candidates: Sequence[RecipeCandidate],
) -> dict[str, Any]:
    pools = partition_e028_candidates(candidates)
    stage1 = _e031_candidates(pools.stage1)
    stage2 = _e031_candidates(pools.stage2)
    if not stage1 or not stage2:
        raise ValueError("the frozen E031 candidate pool requires Stage 1 and Stage 2 recipes")

    def serialized(values: Sequence[RecipeCandidate]) -> list[dict[str, Any]]:
        return [
            {
                "signature": candidate.signature,
                "method_id": candidate.method_id,
                "configuration": candidate.configuration,
                "observations": candidate.observations,
            }
            for candidate in sorted(values, key=lambda item: item.signature)
        ]

    # SR-MPGD is deliberately absent: changing a recipe which E031 can never
    # execute must not alter the fingerprint of the effective candidate pool.
    return {
        "protocol": E031_PROTOCOL,
        "stage1": serialized(stage1),
        "stage2": serialized(stage2),
    }


def candidate_pool_sha256(candidates: Sequence[RecipeCandidate]) -> str:
    """Fingerprint the effective E031 Stage 1 and Stage 2 advisor pools."""

    return _sha256_json(_effective_e031_candidate_material(candidates))


def _compose_advisor_stage2_candidate(
    stage1: RecipeCandidate,
    template: RecipeCandidate,
) -> RecipeCandidate:
    configuration = deepcopy(template.configuration)
    configuration["generation"] = deepcopy(stage1.configuration.get("generation") or {})
    configuration["model"] = deepcopy(stage1.configuration.get("model") or {})
    configuration["output_variant"] = "srpg"
    configuration["reuse_stage1"] = True
    configuration["require_exact_stage1_reuse"] = True
    tools = deepcopy(configuration.get("tools") or {})
    settings = {
        key: value
        for key, value in (tools.get("settings") or {}).items()
        if not str(key).startswith("srmpgd_")
    }
    settings["diffqrcoder_stage2_target_mode"] = "binary_exact"
    tools.update(
        {
            "srpg_enabled": True,
            "srmpgd_enabled": False,
            "settings": settings,
        }
    )
    configuration["tools"] = tools
    signature = _sha256_json(configuration)
    return RecipeCandidate(
        id=f"e031-stage2-{signature[:12]}",
        method_id=template.method_id,
        configuration=configuration,
        signature=signature,
        observations=template.observations,
    )


def _recommendation_score(
    recommendation: ParameterRecommendation,
) -> tuple[Any, ...]:
    saturation = _finite(recommendation.predicted_saturation_risk)
    duration = _finite(recommendation.predicted_duration_ms)

    def optional(value: Any) -> float:
        result = _finite(value)
        return result if result is not None else -math.inf

    # Both stages are selected conservatively.  Candidate signature is the
    # final tie-breaker so advisor return order cannot change the experiment.
    return (
        bool(recommendation.scan_safe),
        optional(recommendation.qr_success_lower_bound),
        optional(recommendation.predicted_qr_tolerance),
        -saturation if saturation is not None else -math.inf,
        optional(recommendation.predicted_hpsv2_1),
        optional(recommendation.predicted_clip_score),
        optional(recommendation.predicted_clip_aesthetic),
        -duration if duration is not None else -math.inf,
        recommendation.candidate.signature,
    )


def _recommend_one(
    advisor: Any,
    *,
    prompt: str,
    candidates: Sequence[RecipeCandidate],
    prompt_embedding: Sequence[float] | None,
    payload_length: int,
    error_correction: str,
    qr_context: Mapping[str, Any],
    scan_probability_threshold: float,
    context_features: Mapping[str, Any] | None = None,
) -> ParameterRecommendation:
    ranked = advisor.recommend(
        prompt=prompt,
        candidates=candidates,
        prompt_embedding=prompt_embedding,
        payload_length=payload_length,
        error_correction=error_correction,
        qr_context=qr_context,
        context_features=context_features,
        scan_probability_threshold=scan_probability_threshold,
        limit=len(candidates),
    )
    allowed = {candidate.signature for candidate in candidates}
    valid = [item for item in ranked if item.candidate.signature in allowed]
    if not valid:
        raise ValueError("the advisor returned no recommendation from the allowed pool")
    return max(valid, key=_recommendation_score)


def recommend_e031_advisor_chains(
    advisor: Any,
    candidates: Sequence[RecipeCandidate],
    prompts: Sequence[Mapping[str, Any]],
    payload: str,
    prompt_embedding_provider: Callable[[str], Sequence[float]] | None,
    error_correction: str = "M",
    qr_context: Mapping[str, Any] | None = None,
    scan_probability_threshold: float = E031_QR_TOLERANCE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Select one deterministic, prompt-conditioned Stage 1 -> Stage 2 chain.

    Stage 1 is selected conservatively from the public E028 pool.  Stage 2 is
    then recomposed with that exact Stage 1 model/generation configuration and
    recommended with the Stage 1 predictions supplied as parent features.
    Recommendation objects are retained so the plan fingerprint includes the
    complete prediction evidence rather than just the chosen parameters.
    """

    if not payload:
        raise ValueError("payload is required")
    if not 0.0 <= scan_probability_threshold <= 1.0:
        raise ValueError("scan_probability_threshold must be between zero and one")
    validated_prompts = [LabPrompt.model_validate(dict(item)) for item in prompts]
    if not validated_prompts:
        raise ValueError("E031 requires prompts")
    prompt_ids = [prompt.id for prompt in validated_prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("E031 prompt identifiers must be unique")
    pools = partition_e028_candidates(candidates)
    stage1_pool = _e031_candidates(pools.stage1)
    stage2_pool = _e031_candidates(pools.stage2)
    if not stage1_pool or not stage2_pool:
        raise ValueError("the frozen E031 advisor pool requires Stage 1 and Stage 2 recipes")
    context = {
        "qr_version": 3,
        "qr_mask_pattern": 4,
        "qr_module_size": 20,
        "qr_padding_px": 78,
        **dict(qr_context or {}),
    }
    chains: dict[str, dict[str, Any]] = {}
    for prompt in validated_prompts:
        embedding = (
            prompt_embedding_provider(prompt.text)
            if prompt_embedding_provider is not None
            else None
        )
        stage1 = _recommend_one(
            advisor,
            prompt=prompt.text,
            candidates=stage1_pool,
            prompt_embedding=embedding,
            payload_length=len(payload),
            error_correction=error_correction,
            qr_context=context,
            scan_probability_threshold=scan_probability_threshold,
        )
        stage2_candidates = tuple(
            _compose_advisor_stage2_candidate(stage1.candidate, template)
            for template in stage2_pool
        )
        stage2 = _recommend_one(
            advisor,
            prompt=prompt.text,
            candidates=stage2_candidates,
            prompt_embedding=embedding,
            payload_length=len(payload),
            error_correction=error_correction,
            qr_context=context,
            scan_probability_threshold=scan_probability_threshold,
            context_features={
                "parent_qr_success": stage1.predicted_qr_success,
                "parent_qr_tolerance": stage1.predicted_qr_tolerance,
                "parent_clip_aesthetic": stage1.predicted_clip_aesthetic,
                "parent_clip_score": stage1.predicted_clip_score,
                "parent_hpsv2_1": stage1.predicted_hpsv2_1,
                "parent_saturation_risk": stage1.predicted_saturation_risk,
            },
        )
        chains[prompt.id] = {
            "stage1": stage1.candidate,
            "stage2": stage2.candidate,
            "stage1_prediction": stage1,
            "stage2_prediction": stage2,
        }
    return chains


def _runtime_method(
    source: Any,
    *,
    identifier: str,
    name: str,
    output_variant: str,
) -> dict[str, Any]:
    method = _configuration(source)
    method.pop("description", None)
    tools = deepcopy(method.get("tools") or {})
    settings = deepcopy(tools.get("settings") or {})
    if output_variant == "raw":
        if tools.get("srpg_enabled") or tools.get("srmpgd_enabled"):
            raise ValueError("an E031 Stage 1 recipe must be raw and contain no SRPG/SR-MPGD")
        tools.update({"srpg_enabled": False, "srmpgd_enabled": False, "settings": settings})
    elif output_variant == "srpg":
        tools.update({"srpg_enabled": True, "srmpgd_enabled": False})
        settings = {
            key: value
            for key, value in settings.items()
            if not str(key).startswith("srmpgd_")
        }
        settings["diffqrcoder_stage2_target_mode"] = "binary_exact"
        tools["settings"] = settings
    else:
        raise ValueError(f"unsupported E031 output variant: {output_variant}")
    method.update(
        {
            "id": _safe_identifier(identifier, maximum=100),
            "name": name[:200],
            "backend": "controlnet",
            "enabled": True,
            "output_variant": output_variant,
            "reuse_stage1": True,
            "require_exact_stage1_reuse": output_variant == "srpg",
            "tools": tools,
        }
    )
    validated = LabMethod.model_validate(method).model_dump(mode="json")
    _require_e031_model(validated, context=f"E031 method {identifier!r}")
    return validated


def _compose_stage2(stage1: Any, stage2: Any) -> dict[str, Any]:
    """Bind the Stage 2 template to the exact Stage 1 model and generation math."""

    stage1_configuration = _configuration(stage1)
    configuration = _configuration(stage2)
    configuration["generation"] = deepcopy(stage1_configuration.get("generation") or {})
    configuration["model"] = deepcopy(stage1_configuration.get("model") or {})
    return configuration


def _prediction_row(
    *,
    prompt: LabPrompt,
    method: Mapping[str, Any],
    branch_id: str,
    pipeline_state: str,
    source_method_id: str,
    parent_stage1_method_id: str | None,
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = dict(prediction or {})
    values.pop("candidate", None)
    return {
        **values,
        "prompt_id": prompt.id,
        "prompt_text": prompt.text,
        "plan_method_id": method["id"],
        "source_method_id": source_method_id,
        "role": f"e031_{branch_id}_{pipeline_state}",
        "pipeline_state": pipeline_state,
        "branch_id": branch_id,
        "chain_id": branch_id,
        "selection_profile": branch_id,
        "parent_stage1_method_id": parent_stage1_method_id,
        "candidate_signature": _sha256_json(method),
        "candidate_configuration": deepcopy(dict(method)),
        "requested_source_output_variant": method["output_variant"],
        "runtime_output_variant": method["output_variant"],
    }


def _advisor_chain(chain: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    if "stage1" not in chain or "stage2" not in chain:
        raise ValueError("each advisor chain requires 'stage1' and 'stage2'")
    stage1 = chain["stage1"]
    stage2 = chain["stage2"]
    stage1_prediction = _prediction_values(chain.get("stage1_prediction"))
    stage2_prediction = _prediction_values(chain.get("stage2_prediction"))
    return stage1, stage2, stage1_prediction, stage2_prediction


def build_e031_prospective_plan(
    *,
    prompts: Sequence[Mapping[str, Any]],
    payload: str,
    advisor_chains: Mapping[str, Mapping[str, Any]],
    advisor_sha256: str,
    candidate_pool_sha256: str,
    fixed_stage1: Mapping[str, Any] | None = None,
    fixed_stage2: Mapping[str, Any] | None = None,
    primary_seed: int = E031_PRIMARY_SEED,
    retry_seed: int = E031_RETRY_SEED,
    error_correction: str = "M",
) -> AdvisorInferencePlan:
    """Build the immutable three-branch E031 campaign plan.

    Campaigns are grouped by effective recipe.  The fixed primary, advisor
    primary and fixed retry branches each contain an exact Stage 1/Stage 2 pair,
    yielding exactly six API trials per prompt and no SR-MPGD trial.
    """

    validated_prompts = [LabPrompt.model_validate(dict(item)) for item in prompts]
    if not validated_prompts:
        raise ValueError("E031 requires prompts")
    if len({item.id for item in validated_prompts}) != len(validated_prompts):
        raise ValueError("E031 prompt ids must be unique")
    if not payload:
        raise ValueError("E031 payload must be non-empty")
    if not _SHA256_PATTERN.fullmatch(str(advisor_sha256).casefold()):
        raise ValueError("advisor_sha256 must be a SHA-256")
    if not _SHA256_PATTERN.fullmatch(str(candidate_pool_sha256).casefold()):
        raise ValueError("candidate_pool_sha256 must be a SHA-256")
    seeds = (int(primary_seed), int(retry_seed))
    if len(set(seeds)) != 2 or any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError("E031 primary and retry seeds must be distinct uint32 values")
    missing = sorted({item.id for item in validated_prompts} - set(advisor_chains))
    extra = sorted(set(advisor_chains) - {item.id for item in validated_prompts})
    if missing or extra:
        raise ValueError(f"advisor chain coverage mismatch: missing={missing}, extra={extra}")

    profiles = {item["id"]: item for item in laboratory_profiles()}
    fixed_stage1 = fixed_stage1 or profiles["diffqrcoder_stage1"]
    fixed_stage2 = fixed_stage2 or profiles["diffqrcoder_srpg"]

    campaigns: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    public_campaigns: list[dict[str, Any]] = []

    def add_campaign(
        *,
        branch_id: str,
        group_id: str,
        group_prompts: Sequence[LabPrompt],
        seed: int,
        stage1_source: Any,
        stage2_source: Any,
        stage1_predictions: Mapping[str, Mapping[str, Any]] | None = None,
        stage2_predictions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        suffix = _safe_identifier(group_id, maximum=24)
        stage1_id = f"e031_{branch_id}_{suffix}_stage1"
        stage2_id = f"e031_{branch_id}_{suffix}_stage2"
        _require_e031_model(
            _configuration(stage1_source),
            context=f"E031 Stage 1 source {stage1_id!r}",
        )
        _require_e031_model(
            _configuration(stage2_source),
            context=f"E031 Stage 2 source {stage2_id!r}",
        )
        stage1 = _runtime_method(
            stage1_source,
            identifier=stage1_id,
            name=f"E031 {branch_id} Stage 1",
            output_variant="raw",
        )
        stage2 = _runtime_method(
            _compose_stage2(stage1_source, stage2_source),
            identifier=stage2_id,
            name=f"E031 {branch_id} Stage 2",
            output_variant="srpg",
        )
        request = LabCampaignCreate.model_validate(
            {
                "name": f"E031 {branch_id} {suffix}",
                "payload": payload,
                "error_correction": error_correction,
                "prompts": [item.model_dump(mode="json") for item in group_prompts],
                "seeds": [seed],
                "methods": [stage1, stage2],
                "max_attempts": 1,
            }
        ).model_dump(mode="json")
        campaigns.append(request)
        public_campaigns.append(
            {
                "name": request["name"],
                "branch_id": branch_id,
                "recipe_group": group_id,
                "prompt_count": len(group_prompts),
                "trial_count": len(group_prompts) * 2,
                "seed": seed,
            }
        )
        for prompt in group_prompts:
            predictions.extend(
                [
                    _prediction_row(
                        prompt=prompt,
                        method=stage1,
                        branch_id=branch_id,
                        pipeline_state="stage1",
                        source_method_id=_source_method_id(stage1_source, stage1_id),
                        parent_stage1_method_id=None,
                        prediction=(stage1_predictions or {}).get(prompt.id),
                    ),
                    _prediction_row(
                        prompt=prompt,
                        method=stage2,
                        branch_id=branch_id,
                        pipeline_state="stage2",
                        source_method_id=_source_method_id(stage2_source, stage2_id),
                        parent_stage1_method_id=stage1_id,
                        prediction=(stage2_predictions or {}).get(prompt.id),
                    ),
                ]
            )

    add_campaign(
        branch_id="fixed_seed_a",
        group_id="fixed",
        group_prompts=validated_prompts,
        seed=seeds[0],
        stage1_source=fixed_stage1,
        stage2_source=fixed_stage2,
    )

    advisor_groups: dict[str, dict[str, Any]] = {}
    for prompt in validated_prompts:
        chain = advisor_chains[prompt.id]
        stage1, stage2, stage1_prediction, stage2_prediction = _advisor_chain(chain)
        effective_stage1 = _configuration(stage1)
        effective_stage2 = _compose_stage2(stage1, stage2)
        signature = _sha256_json(
            {"stage1": effective_stage1, "stage2": effective_stage2}
        )
        group = advisor_groups.setdefault(
            signature,
            {
                "stage1": stage1,
                "stage2": stage2,
                "prompts": [],
                "stage1_predictions": {},
                "stage2_predictions": {},
            },
        )
        group["prompts"].append(prompt)
        group["stage1_predictions"][prompt.id] = stage1_prediction
        group["stage2_predictions"][prompt.id] = stage2_prediction

    for index, (signature, group) in enumerate(sorted(advisor_groups.items()), start=1):
        add_campaign(
            branch_id="advisor_seed_a",
            group_id=f"advisor_{index:02d}_{signature[:8]}",
            group_prompts=group["prompts"],
            seed=seeds[0],
            stage1_source=group["stage1"],
            stage2_source=group["stage2"],
            stage1_predictions=group["stage1_predictions"],
            stage2_predictions=group["stage2_predictions"],
        )

    add_campaign(
        branch_id="fixed_seed_b",
        group_id="fixed",
        group_prompts=validated_prompts,
        seed=seeds[1],
        stage1_source=fixed_stage1,
        stage2_source=fixed_stage2,
    )

    predictions.sort(
        key=lambda item: (
            E031_BRANCHES.index(str(item["branch_id"])),
            str(item["prompt_id"]),
            0 if item["pipeline_state"] == "stage1" else 1,
            str(item["plan_method_id"]),
        )
    )
    prediction_sha256 = _sha256_json(predictions)
    plan_material = {
        "protocol": E031_PROTOCOL,
        "advisor_sha256": str(advisor_sha256).casefold(),
        "candidate_pool_sha256": str(candidate_pool_sha256).casefold(),
        "prediction_sha256": prediction_sha256,
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "payload_length": len(payload),
        "error_correction": error_correction,
        "primary_seed": seeds[0],
        "retry_seed": seeds[1],
        "branches": list(E031_BRANCHES),
        "qr_verify_repetitions": E031_QR_VERIFY_REPETITIONS,
        "qr_verify_presets": E031_QR_VERIFY_PRESET_COUNT,
        "delivery_threshold": E031_QR_TOLERANCE_THRESHOLD,
        "strict_threshold_requested": E031_STRICT_QR_TOLERANCE_THRESHOLD,
        "strict_exact_presets": E031_STRICT_EXACT_PRESETS,
        "strict_threshold_effective": (
            E031_STRICT_EXACT_PRESETS / E031_QR_VERIFY_PRESET_COUNT
        ),
        "saturation_threshold": E031_SATURATION_THRESHOLD,
        "prompts": [item.model_dump(mode="json") for item in validated_prompts],
        "campaigns": public_campaigns,
    }
    plan_id = _sha256_json(plan_material)[:16]
    public = {
        **plan_material,
        "experiment": E031_EXPERIMENT,
        "plan_id": plan_id,
        "prompt_count": len(validated_prompts),
        "campaign_count": len(campaigns),
        "stage1_trial_count": len(validated_prompts) * 3,
        "stage2_trial_count": len(validated_prompts) * 3,
        "trial_count": len(validated_prompts) * 6,
        "advisor_recipe_groups": len(advisor_groups),
        "srmpgd_trial_count": 0,
    }
    actual_trials = sum(
        len(campaign["prompts"]) * len(campaign["seeds"]) * len(campaign["methods"])
        for campaign in campaigns
    )
    if actual_trials != public["trial_count"]:
        raise RuntimeError(f"E031 plan trial mismatch: {actual_trials} != {public['trial_count']}")
    return AdvisorInferencePlan(
        plan_id=plan_id,
        payload=payload,
        campaigns=tuple(campaigns),
        predictions=tuple(predictions),
        public=public,
    )


def _fixed_recipe_audit_reasons(
    rows: Sequence[dict[str, Any]],
    by_run: Mapping[str, Mapping[str, Any]],
) -> dict[int, list[str]]:
    """Prove that fixed A/B differ by their registered seed, not their recipe."""

    failures: dict[int, list[str]] = defaultdict(list)
    stage2_by_key = {
        (str(row.get("prompt_id") or ""), str(row.get("branch_id") or "")): row
        for row in rows
        if str(row.get("pipeline_state")) == "stage2"
        and row.get("branch_id") in {"fixed_seed_a", "fixed_seed_b"}
    }
    prompt_ids = sorted(prompt_id for prompt_id, _ in stage2_by_key if prompt_id)
    for prompt_id in sorted(set(prompt_ids)):
        fixed_rows = {
            branch: stage2_by_key.get((prompt_id, branch))
            for branch in ("fixed_seed_a", "fixed_seed_b")
        }
        present = [row for row in fixed_rows.values() if row is not None]
        missing = [branch for branch, row in fixed_rows.items() if row is None]
        if missing:
            reason = f"fixed_recipe_peer_missing:{','.join(missing)}"
            for row in present:
                failures[id(row)].append(reason)
            continue

        fixed_a = fixed_rows["fixed_seed_a"]
        fixed_b = fixed_rows["fixed_seed_b"]
        assert fixed_a is not None and fixed_b is not None
        for phase, left, right in (
            (
                "stage1",
                by_run.get(str(fixed_a.get("stage1_source_run_id") or "")),
                by_run.get(str(fixed_b.get("stage1_source_run_id") or "")),
            ),
            ("stage2", fixed_a, fixed_b),
        ):
            configurations = {
                "fixed_seed_a": (left or {}).get("candidate_configuration"),
                "fixed_seed_b": (right or {}).get("candidate_configuration"),
            }
            missing_configurations = [
                branch
                for branch, configuration in configurations.items()
                if _recipe_identity(configuration) is None
            ]
            if missing_configurations:
                reason = (
                    f"fixed_{phase}_recipe_configuration_missing:"
                    f"{','.join(missing_configurations)}"
                )
                failures[id(fixed_a)].append(reason)
                failures[id(fixed_b)].append(reason)
                continue
            identities = {
                branch: _recipe_identity(configuration)
                for branch, configuration in configurations.items()
            }
            if identities["fixed_seed_a"] != identities["fixed_seed_b"]:
                reason = f"fixed_{phase}_recipe_mismatch"
                failures[id(fixed_a)].append(reason)
                failures[id(fixed_b)].append(reason)
    return failures


def audit_e031_pairing(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prove the exact Stage 1 parent of every generated E031 Stage 2 raster."""

    rows = [dict(item) for item in entries]
    by_run = {
        str(row.get("generation_run_id")): row
        for row in rows
        if row.get("generation_run_id")
    }
    fixed_recipe_failures = _fixed_recipe_audit_reasons(rows, by_run)
    audits: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("pipeline_state")) != "stage2":
            continue
        reasons: list[str] = list(fixed_recipe_failures.get(id(row), ()))
        branch_id = str(row.get("branch_id") or "")
        expected_seed = _E031_BRANCH_SEEDS.get(branch_id)
        if expected_seed is None:
            reasons.append("branch_unknown")
        elif _finite(row.get("seed")) != float(expected_seed):
            reasons.append(f"branch_seed_unexpected:{expected_seed}")
        status = str(row.get("status") or "").casefold()
        technically_generated = status in {"accepted", "rejected"}
        if not technically_generated:
            reasons.append("stage2_generation_failed")
        if str(row.get("output_variant") or "").casefold() != "srpg":
            reasons.append("stage2_output_is_not_srpg")
        if "srmpgd" in str(row.get("requested_source_output_variant") or "").casefold():
            reasons.append("srmpgd_forbidden")

        stage2_configuration = row.get("candidate_configuration")
        if not isinstance(stage2_configuration, Mapping):
            reasons.append("stage2_recipe_configuration_missing")
        else:
            reasons.extend(
                f"stage2_model_stack_mismatch:{mismatch}"
                for mismatch in _e031_model_mismatches(stage2_configuration)
            )

        source_run_id = str(row.get("stage1_source_run_id") or "")
        source = by_run.get(source_run_id)
        if source is None:
            reasons.append("stage1_source_run_missing")
        else:
            if str(source.get("pipeline_state")) != "stage1":
                reasons.append("stage1_source_state_invalid")
            if str(source.get("output_variant") or "").casefold() != "raw":
                reasons.append("stage1_source_output_is_not_raw")
            if source.get("prompt_id") != row.get("prompt_id"):
                reasons.append("stage1_prompt_mismatch")
            if source.get("branch_id") != row.get("branch_id"):
                reasons.append("stage1_branch_mismatch")
            if _finite(source.get("seed")) != _finite(row.get("seed")):
                reasons.append("stage1_seed_mismatch")
            if source.get("campaign_id") != row.get("campaign_id"):
                reasons.append("stage1_campaign_mismatch")
            expected_method = str(row.get("parent_stage1_method_id") or "")
            if expected_method and str(source.get("method_id")) != expected_method:
                reasons.append("stage1_method_mismatch")
            stage1_configuration = source.get("candidate_configuration")
            if not isinstance(stage1_configuration, Mapping):
                reasons.append("stage1_recipe_configuration_missing")
            else:
                reasons.extend(
                    f"stage1_model_stack_mismatch:{mismatch}"
                    for mismatch in _e031_model_mismatches(stage1_configuration)
                )

        if _strict_bool(row.get("stage1_reused")) is not True:
            reasons.append("stage1_reuse_marker_missing")
        source_hash = str((source or {}).get("final_image_sha256") or "").casefold()
        reused_hash = str(row.get("stage1_image_sha256") or "").casefold()
        if not _SHA256_PATTERN.fullmatch(source_hash):
            reasons.append("stage1_source_hash_missing")
        if not _SHA256_PATTERN.fullmatch(reused_hash):
            reasons.append("stage1_reused_hash_missing")
        if source_hash and reused_hash and source_hash != reused_hash:
            reasons.append("stage1_raster_hash_mismatch")
        final_hash = str(row.get("final_image_sha256") or "").casefold()
        if technically_generated and not _SHA256_PATTERN.fullmatch(final_hash):
            reasons.append("stage2_raster_hash_missing")

        audits.append(
            {
                "prompt_id": row.get("prompt_id"),
                "branch_id": row.get("branch_id"),
                "seed": _finite(row.get("seed")),
                "expected_seed": expected_seed,
                "campaign_id": row.get("campaign_id"),
                "stage2_method_id": row.get("method_id"),
                "stage2_run_id": row.get("generation_run_id"),
                "stage1_source_run_id": row.get("stage1_source_run_id"),
                "stage1_recipe_identity_sha256": _recipe_identity(
                    (source or {}).get("candidate_configuration")
                ),
                "stage2_recipe_identity_sha256": _recipe_identity(
                    row.get("candidate_configuration")
                ),
                "technically_generated": technically_generated,
                "complete": not reasons,
                "failure_reasons": ";".join(reasons),
            }
        )
    return sorted(
        audits,
        key=lambda item: (
            str(item.get("prompt_id")),
            E031_BRANCHES.index(str(item.get("branch_id")))
            if item.get("branch_id") in E031_BRANCHES
            else 999,
        ),
    )


def _score_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    raise TypeError(f"unsupported conservative score: {type(value)!r}")


def enrich_e031_stage2_results(
    entries: Sequence[Mapping[str, Any]],
    scores_by_raster: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind repeated QR-Verify evidence to each exact Stage 2 raster."""

    enriched: list[dict[str, Any]] = []
    for source in entries:
        if str(source.get("pipeline_state")) != "stage2":
            continue
        row = dict(source)
        raster_hash = str(row.get("final_image_sha256") or "").casefold()
        generated = str(row.get("status") or "").casefold() in {"accepted", "rejected"}
        if not generated:
            row.update(
                {
                    "qr_verify_observations": [],
                    "conservative_qr_success": False,
                    "conservative_qr_tolerance": None,
                    "conservative_exact_presets": 0,
                    "qr_verify_repetitions": 0,
                    "qr_verify_preset_count": 0,
                }
            )
            enriched.append(row)
            continue
        if not _SHA256_PATTERN.fullmatch(raster_hash):
            raise ValueError(f"invalid E031 Stage 2 raster hash: {raster_hash!r}")
        downloaded_hash = str(row.get("downloaded_raster_sha256") or raster_hash).casefold()
        if downloaded_hash != raster_hash:
            raise ValueError(
                f"downloaded raster differs from provenance for {row.get('generation_run_id')}"
            )
        if raster_hash not in scores_by_raster:
            raise KeyError(f"missing conservative QR-Verify score for raster {raster_hash}")
        score = _score_dict(scores_by_raster[raster_hash])
        if str(score.get("image_sha256") or "").casefold() != raster_hash:
            raise ValueError(f"QR-Verify score is bound to another raster: {raster_hash}")
        repetitions = int(score.get("repetitions") or 0)
        preset_count = int(score.get("preset_count") or 0)
        runs = list(score.get("runs") or [])
        if repetitions != E031_QR_VERIFY_REPETITIONS or len(runs) != repetitions:
            raise ValueError(
                f"E031 requires {E031_QR_VERIFY_REPETITIONS} QR-Verify repetitions"
            )
        if preset_count != E031_QR_VERIFY_PRESET_COUNT:
            raise ValueError(
                f"E031 requires {E031_QR_VERIFY_PRESET_COUNT} QR-Verify presets"
            )
        saturation = _finite(row.get("saturation_risk"))
        observations = [
            {
                "repetition": int(run.get("repetition") or index),
                # The conservative gate deliberately rejects Python booleans
                # as ambiguous numeric input.  Persist an explicit binary
                # measurement, exactly like E030.
                "qr_success": float(bool(run.get("any_exact"))),
                "qr_tolerance": _finite(run.get("tolerance_score")),
                "image_sha256": raster_hash,
                "saturation_risk": saturation,
            }
            for index, run in enumerate(runs, start=1)
        ]
        row.update(
            {
                "qr_verify_observations": observations,
                "conservative_qr_success": bool(score.get("each_repetition_any_exact")),
                "conservative_qr_tolerance": float(
                    score.get("conservative_tolerance_score") or 0.0
                ),
                "conservative_exact_presets": int(
                    score.get("conservative_exact_presets") or 0
                ),
                "qr_verify_repetitions": repetitions,
                "qr_verify_preset_count": preset_count,
                "qr_verify_unstable_presets": int(score.get("unstable_preset_count") or 0),
                "qr_verify_cache_key": score.get("cache_key"),
                "qr_verify_engine_version": score.get("engine_version"),
                "qr_verify_scoring_version": score.get("scoring_version"),
                "qr_verify_implementation_sha256": score.get("implementation_sha256"),
            }
        )
        enriched.append(row)
    return sorted(
        enriched,
        key=lambda item: (
            str(item.get("prompt_id")),
            E031_BRANCHES.index(str(item.get("branch_id")))
            if item.get("branch_id") in E031_BRANCHES
            else 999,
        ),
    )


def e031_candidate_rank(
    candidate: Mapping[str, Any],
    assessment: Stage2CandidateAssessment | None = None,
) -> tuple[Any, ...]:
    """Rank only after the gate: QR, saturation, HPS, CLIPScore, then AES."""

    assessment = assessment or assess_stage2_candidate(candidate)

    def score(name: str) -> float:
        value = _finite(candidate.get(name))
        return value if value is not None else -math.inf

    return (
        assessment.deliverable,
        assessment.qr.minimum_tolerance
        if assessment.qr.minimum_tolerance is not None
        else -math.inf,
        -assessment.maximum_saturation_risk
        if assessment.maximum_saturation_risk is not None
        else -math.inf,
        score("hpsv2_1"),
        score("clip_score"),
        score("clip_aesthetic"),
        -E031_BRANCHES.index(str(candidate.get("branch_id")))
        if candidate.get("branch_id") in E031_BRANCHES
        else -999,
    )


def select_e031_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    gate: ConservativeDeliveryGate,
) -> Mapping[str, Any] | None:
    assessed = [(candidate, assess_stage2_candidate(candidate, gate)) for candidate in candidates]
    passing = [(candidate, result) for candidate, result in assessed if result.deliverable]
    if not passing:
        return None
    return max(passing, key=lambda item: e031_candidate_rank(*item))[0]


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Two-sided Wilson score interval, clamped for plotting safety."""

    if total < 1:
        raise ValueError("Wilson interval requires a positive total")
    if successes < 0 or successes > total:
        raise ValueError("Wilson successes must be between zero and total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _gate(name: str) -> ConservativeDeliveryGate:
    threshold = (
        E031_QR_TOLERANCE_THRESHOLD
        if name == "standard"
        else E031_STRICT_QR_TOLERANCE_THRESHOLD
    )
    return ConservativeDeliveryGate(
        qr_tolerance_threshold=threshold,
        saturation_threshold=E031_SATURATION_THRESHOLD,
        minimum_qr_observations=E031_QR_VERIFY_REPETITIONS,
    )


def _decision(
    *,
    prompt_id: str,
    family: str,
    policy: str,
    gate_name: str,
    candidates_by_branch: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    branches = E031_POLICIES[policy]
    gate = _gate(gate_name)
    attempted: list[tuple[Mapping[str, Any], Stage2CandidateAssessment]] = []
    selected: Mapping[str, Any] | None = None
    selected_assessment: Stage2CandidateAssessment | None = None
    if policy == "best_of_three":
        for branch in branches:
            candidate = candidates_by_branch.get(branch)
            if candidate is not None:
                attempted.append((candidate, assess_stage2_candidate(candidate, gate)))
        selected = select_e031_candidate([item[0] for item in attempted], gate=gate)
        if selected is not None:
            selected_assessment = assess_stage2_candidate(selected, gate)
    else:
        for branch in branches:
            candidate = candidates_by_branch.get(branch)
            if candidate is None:
                continue
            assessment = assess_stage2_candidate(candidate, gate)
            attempted.append((candidate, assessment))
            if assessment.deliverable:
                selected = candidate
                selected_assessment = assessment
                break

    attempts_used = len(attempted)
    selected_branch = selected.get("branch_id") if selected else None
    selected_index = (
        next(
            (
                index
                for index, (candidate, _) in enumerate(attempted, start=1)
                if candidate is selected
            ),
            None,
        )
        if selected is not None
        else None
    )
    return {
        "prompt_id": prompt_id,
        "prompt_family": family,
        "policy": policy,
        "gate": gate_name,
        "gate_requested_threshold": gate.qr_tolerance_threshold,
        "gate_effective_exact_presets": (
            math.ceil(gate.qr_tolerance_threshold * E031_QR_VERIFY_PRESET_COUNT)
        ),
        "deliverable": selected is not None,
        "selected_branch": selected_branch,
        "selected_attempt": selected_index,
        "selected_generation_run_id": selected.get("generation_run_id") if selected else None,
        "stage2_attempts_used": attempts_used,
        "api_trials_used": attempts_used * 2,
        "planned_stage2_rasters": len(branches),
        "planned_api_trials": len(branches) * 2,
        "conservative_qr_tolerance": (
            selected_assessment.qr.minimum_tolerance if selected_assessment else None
        ),
        "conservative_exact_presets": (
            int(selected.get("conservative_exact_presets") or 0) if selected else 0
        ),
        "maximum_saturation_risk": (
            selected_assessment.maximum_saturation_risk if selected_assessment else None
        ),
        "hpsv2_1": _finite(selected.get("hpsv2_1")) if selected else None,
        "clip_score": _finite(selected.get("clip_score")) if selected else None,
        "clip_aesthetic": _finite(selected.get("clip_aesthetic")) if selected else None,
        "attempted_branches": [str(item[0].get("branch_id")) for item in attempted],
        "rejection_reasons": {
            str(candidate.get("branch_id")): list(assessment.rejection_reasons)
            for candidate, assessment in attempted
            if not assessment.deliverable
        },
    }


def evaluate_e031_policies(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Replay E031 cascades under the E030 and strict 36/37 gates."""

    stage2 = [dict(row) for row in entries if str(row.get("pipeline_state")) == "stage2"]
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    families: dict[str, str] = {}
    for row in stage2:
        prompt_id = str(row.get("prompt_id") or "")
        branch = str(row.get("branch_id") or "")
        if not prompt_id or branch not in E031_BRANCHES:
            raise ValueError(f"invalid E031 Stage 2 identity: {prompt_id!r}/{branch!r}")
        if branch in groups[prompt_id]:
            raise ValueError(f"duplicate E031 branch for {prompt_id}/{branch}")
        groups[prompt_id][branch] = row
        family = "simple" if "_simple_" in prompt_id else "atypical"
        families[prompt_id] = family
    incomplete = {
        prompt_id: sorted(set(E031_BRANCHES) - set(branches))
        for prompt_id, branches in groups.items()
        if set(branches) != set(E031_BRANCHES)
    }
    if incomplete:
        raise ValueError(f"incomplete E031 branch matrix: {incomplete}")
    if not groups:
        raise ValueError("no E031 Stage 2 results")

    decisions = [
        _decision(
            prompt_id=prompt_id,
            family=families[prompt_id],
            policy=policy,
            gate_name=gate_name,
            candidates_by_branch=groups[prompt_id],
        )
        for gate_name in ("standard", "strict")
        for policy in E031_POLICIES
        for prompt_id in sorted(groups)
    ]

    summaries: list[dict[str, Any]] = []
    for gate_name in ("standard", "strict"):
        for policy in E031_POLICIES:
            selected = [
                row
                for row in decisions
                if row["gate"] == gate_name and row["policy"] == policy
            ]
            delivered = [row for row in selected if row["deliverable"]]
            successes = len(delivered)
            low, high = wilson_interval(successes, len(selected))

            def mean(name: str, values: Sequence[Mapping[str, Any]]) -> float | None:
                numbers = [_finite(item.get(name)) for item in values]
                numbers = [item for item in numbers if item is not None]
                return sum(numbers) / len(numbers) if numbers else None

            summaries.append(
                {
                    "gate": gate_name,
                    "policy": policy,
                    "prompts": len(selected),
                    "delivered": successes,
                    "delivery_rate": successes / len(selected),
                    "wilson_95_low": low,
                    "wilson_95_high": high,
                    "mean_stage2_attempts": mean("stage2_attempts_used", selected),
                    "maximum_stage2_attempts": max(
                        int(item["stage2_attempts_used"]) for item in selected
                    ),
                    "total_effective_api_trials": sum(
                        int(item["api_trials_used"]) for item in selected
                    ),
                    "mean_qr_tolerance_delivered": mean(
                        "conservative_qr_tolerance", delivered
                    ),
                    "mean_saturation_delivered": mean(
                        "maximum_saturation_risk", delivered
                    ),
                    "mean_hpsv2_1_delivered": mean("hpsv2_1", delivered),
                    "mean_clip_score_delivered": mean("clip_score", delivered),
                    "mean_clip_aesthetic_delivered": mean(
                        "clip_aesthetic", delivered
                    ),
                    "selected_branch_counts": dict(
                        Counter(
                            str(item["selected_branch"])
                            for item in delivered
                            if item.get("selected_branch")
                        )
                    ),
                    "stage1_was_delivered": False,
                    "srmpgd_was_requested": False,
                }
            )
    return {"decisions": decisions, "summary": summaries}
