from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .advisor_inference import AdvisorInferencePlan
from .lab import laboratory_profiles
from .qr import generate_diffqrcoder_qr
from .schemas import LabCampaignCreate, LabMethod, LabPrompt

E027_PIPELINE_STATES = ("stage1", "stage2", "srmpgd")
E027_POLICIES = ("cascade", "full_lexicographic", "forced_srmpgd")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_e027_holdout_prompts(
    count: int = 100,
    *,
    seen_prompt_texts: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build a deterministic prompt bank disjoint from the E026 week bank.

    The bank deliberately alternates ordinary compositions and unusual scenes.
    It does not call a text model, so the exact holdout can be rebuilt and audited.
    """

    if count < 1 or count > 500:
        raise ValueError("E027 prompt count must be between 1 and 500")
    simple_subjects = [
        "linen armchair",
        "green enamel kettle",
        "basket of apricots",
        "handmade leather satchel",
        "white racing bicycle",
        "cedar perfume bottle",
        "blue ceramic teapot",
        "pair of hiking boots",
        "brass desk fan",
        "small lemon tree",
        "wooden toy sailboat",
        "red espresso machine",
        "folded wool blanket",
        "glass carafe",
        "vintage field camera",
        "terracotta planter",
        "silver pocket watch",
        "lacquer jewelry box",
        "cream acoustic guitar",
        "stone garden fountain",
    ]
    unusual_subjects = [
        "gravity-defying observatory woven from reeds",
        "transparent manta ray carrying a miniature theatre",
        "spiral greenhouse growing mechanical clouds",
        "porcelain labyrinth inhabited by fireflies",
        "floating tailoring workshop inside a soap bubble",
        "crystalline tram crossing an upside-down forest",
        "folded-paper volcano containing a quiet library",
        "clockwork tide pool orbiting a black pearl",
        "velvet canyon shaped like a musical instrument",
        "bioluminescent bakery beneath a frozen lake",
        "recursive balcony overlooking its own interior",
        "ceramic storm assembled from botanical specimens",
        "miniature opera performed inside a raindrop",
        "woven moon descending into an underground station",
        "glass orchard reflected through impossible mirrors",
        "moss-covered machine translating shadows into birds",
        "levitating museum of unfinished constellations",
        "origami harbor sailing through an abandoned ballroom",
        "coral printing press producing luminous maps",
        "marble staircase curling around a sleeping whale",
    ]
    settings = [
        "in a quiet limewashed room",
        "on a dark walnut table",
        "inside a restrained modern gallery",
        "beneath soft skylight",
        "in a rain-washed courtyard",
        "against a textured plaster wall",
        "under warm evening window light",
        "inside a pale stone pavilion",
        "on a muted indigo backdrop",
        "within a calm winter garden",
    ]
    treatments = [
        "editorial photograph with coherent natural materials",
        "cinematic scene with controlled contrast",
        "detailed gouache illustration with a limited palette",
        "museum display with balanced geometry",
        "architectural visualization with believable depth",
    ]
    seen = {_normalized_prompt(value) for value in seen_prompt_texts}
    prompts: list[dict[str, str]] = []
    candidate_index = 0
    maximum_candidates = 20_000
    while len(prompts) < count and candidate_index < maximum_candidates:
        family = "simple" if candidate_index % 2 == 0 else "atypical"
        subjects = simple_subjects if family == "simple" else unusual_subjects
        subject = subjects[(candidate_index * 7 + candidate_index // 3) % len(subjects)]
        setting = settings[(candidate_index * 3 + candidate_index // 7) % len(settings)]
        treatment = treatments[(candidate_index * 11 + candidate_index // 5) % len(treatments)]
        detail = (
            "single clear subject, generous negative space, no text or typography"
            if family == "simple"
            else "coherent impossible geometry, intricate detail, no text or typography"
        )
        text = f"A {subject} {setting}, {treatment}, {detail}."
        normalized = _normalized_prompt(text)
        candidate_index += 1
        if normalized in seen or any(
            _normalized_prompt(item["text"]) == normalized for item in prompts
        ):
            continue
        prompts.append(
            {
                "id": f"e027h_{family}_{len(prompts) + 1:03d}",
                "text": text,
                "negative_prompt": "text, letters, watermark, logo, barcode, oversaturated colors",
            }
        )
    if len(prompts) != count:
        raise RuntimeError(f"only {len(prompts)} unseen E027 prompts available for count={count}")
    return prompts


def _runtime_profile(profile: Mapping[str, Any], identifier: str, name: str) -> dict[str, Any]:
    result = deepcopy(dict(profile))
    result.pop("description", None)
    result.update({"id": identifier, "name": name, "enabled": True})
    return LabMethod.model_validate(result).model_dump(mode="json")


def build_e027_holdout_plan(
    *,
    payload: str,
    prompts: Sequence[Mapping[str, Any]] | None = None,
    prompt_count: int = 100,
    seen_prompt_texts: Sequence[str] = (),
    seeds: Sequence[int] = (743_001, 857_001, 971_001),
    prompts_per_campaign: int = 20,
    error_correction: str = "M",
    qr_tolerance_threshold: float = 0.80,
) -> AdvisorInferencePlan:
    """Create the paired Stage 1 / Stage 2 / SR-MPGD E027 holdout plan."""

    if not 0.0 <= qr_tolerance_threshold <= 1.0:
        raise ValueError("QR tolerance threshold must be between 0 and 1")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("E027 seeds must be non-empty and unique")
    if prompts_per_campaign < 1 or prompts_per_campaign > 50:
        raise ValueError("prompts_per_campaign must be between 1 and 50")
    selected_prompts = [
        LabPrompt.model_validate(dict(item))
        for item in (
            prompts
            if prompts is not None
            else build_e027_holdout_prompts(
                prompt_count,
                seen_prompt_texts=seen_prompt_texts,
            )
        )
    ]
    if not selected_prompts:
        raise ValueError("E027 requires at least one holdout prompt")
    if len({item.id for item in selected_prompts}) != len(selected_prompts):
        raise ValueError("E027 prompt ids must be unique")
    seen = {_normalized_prompt(value) for value in seen_prompt_texts}
    overlap = [item.id for item in selected_prompts if _normalized_prompt(item.text) in seen]
    if overlap:
        raise ValueError(f"E027 prompts already occur in training data: {overlap}")

    generate_diffqrcoder_qr(
        payload,
        error_correction=error_correction,
        version=3,
        mask_pattern=4,
        module_size=20,
        border=4,
    )
    profiles = {item["id"]: item for item in laboratory_profiles()}
    stage1 = _runtime_profile(
        profiles["diffqrcoder_stage1"],
        "e027_stage1",
        "E027 — Stage 1 apparié",
    )
    stage2 = _runtime_profile(
        profiles["diffqrcoder_srpg"],
        "e027_stage2",
        "E027 — Stage 2 SRPG apparié",
    )
    srmpgd_profile = deepcopy(profiles["diffqrcoder_srmpgd_robust"])
    srmpgd_profile["tools"]["settings"]["srmpgd_min_qr_tolerance"] = float(
        qr_tolerance_threshold
    )
    srmpgd = _runtime_profile(
        srmpgd_profile,
        "e027_srmpgd",
        "E027 — Stage 2 + SR-MPGD robuste apparié",
    )
    methods = [stage1, stage2, srmpgd]

    plan_material = {
        "protocol": "e027-v1-paired-cascade-full-forced-srmpgd",
        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "error_correction": error_correction,
        "prompts": [item.model_dump(mode="json") for item in selected_prompts],
        "seeds": normalized_seeds,
        "methods": methods,
        "qr_tolerance_threshold": qr_tolerance_threshold,
    }
    plan_id = hashlib.sha256(_canonical_json(plan_material).encode("utf-8")).hexdigest()[:16]
    campaigns: list[dict[str, Any]] = []
    public_campaigns: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    state_by_method = {
        "e027_stage1": "stage1",
        "e027_stage2": "stage2",
        "e027_srmpgd": "srmpgd",
    }
    for offset in range(0, len(selected_prompts), prompts_per_campaign):
        batch = selected_prompts[offset : offset + prompts_per_campaign]
        batch_number = len(campaigns) + 1
        request = {
            "name": f"E027 {plan_id} paired batch {batch_number:02d}",
            "payload": payload,
            "error_correction": error_correction,
            "prompts": [item.model_dump(mode="json") for item in batch],
            "seeds": list(normalized_seeds),
            "methods": methods,
            "max_attempts": 1,
        }
        validated = LabCampaignCreate.model_validate(request).model_dump(mode="json")
        campaigns.append(validated)
        public_campaigns.append(
            {
                "batch": batch_number,
                "name": request["name"],
                "prompt_ids": [item.id for item in batch],
                "trials": len(batch) * len(normalized_seeds) * len(methods),
            }
        )
        for prompt in batch:
            for method in methods:
                state = state_by_method[method["id"]]
                predictions.append(
                    {
                        "prompt_id": prompt.id,
                        "prompt_text": prompt.text,
                        "plan_method_id": method["id"],
                        "source_method_id": method["id"],
                        "role": "e027_pipeline_state",
                        "pipeline_state": state,
                        "selection_profile": state,
                        "candidate_signature": hashlib.sha256(
                            _canonical_json(method).encode("utf-8")
                        ).hexdigest(),
                        "requested_source_output_variant": method["output_variant"],
                        "runtime_output_variant": method["output_variant"],
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

    public = {
        "protocol": plan_material["protocol"],
        "plan_id": plan_id,
        "payload_sha256": plan_material["payload_sha256"],
        "payload_length": len(payload),
        "error_correction": error_correction,
        "prompt_count": len(selected_prompts),
        "prompt_bank_sha256": hashlib.sha256(
            _canonical_json([item.text for item in selected_prompts]).encode("utf-8")
        ).hexdigest(),
        "seed_count": len(normalized_seeds),
        "seeds": list(normalized_seeds),
        "context_count": len(selected_prompts) * len(normalized_seeds),
        "states": list(E027_PIPELINE_STATES),
        "policies": list(E027_POLICIES),
        "qr_tolerance_threshold": qr_tolerance_threshold,
        "campaign_count": len(campaigns),
        "trial_count": sum(item["trials"] for item in public_campaigns),
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
    hps = _finite(row.get("hpsv2_1"))
    aesthetic = _finite(row.get("clip_aesthetic"))
    clip = _finite(row.get("clip_score"))
    duration = _finite(row.get("duration_ms"))
    state = str(row.get("pipeline_state") or "")
    return (
        success >= 0.5,
        tolerance,
        saturation <= 0.05,
        hps if hps is not None else -math.inf,
        aesthetic if aesthetic is not None else -math.inf,
        clip if clip is not None else -math.inf,
        -saturation,
        -(duration if duration is not None else math.inf),
        state == "stage1",
    )


def select_e027_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    qr_tolerance_threshold: float = 0.80,
) -> dict[str, Any] | None:
    """Select QR validity first, tolerance second and visual quality third."""

    generated = [
        dict(row)
        for row in rows
        if _finite(row.get("qr_success")) is not None
        and str(row.get("status")) not in {"error", "cancelled", "interrupted"}
    ]
    if not generated:
        return None
    selected = max(generated, key=_candidate_rank)
    selected["deliverable"] = bool(
        (_finite(selected.get("qr_success")) or 0.0) >= 0.5
        and (_finite(selected.get("qr_tolerance")) or 0.0) >= qr_tolerance_threshold
    )
    return selected


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return [max(0.0, centre - margin / denominator), min(1.0, centre + margin / denominator)]


def evaluate_e027_policies(
    entries: Sequence[Mapping[str, Any]],
    *,
    qr_tolerance_threshold: float = 0.80,
) -> dict[str, Any]:
    """Replay cascade, full selection and forced SR-MPGD on paired results."""

    if not 0.0 <= qr_tolerance_threshold <= 1.0:
        raise ValueError("QR tolerance threshold must be between 0 and 1")
    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for source in entries:
        if source.get("role") != "e027_pipeline_state":
            continue
        key = (str(source.get("prompt_id")), _finite(source.get("seed")))
        groups.setdefault(key, []).append(dict(source))
    decisions: list[dict[str, Any]] = []
    group_audit: list[dict[str, Any]] = []
    for (prompt_id, seed), rows in sorted(groups.items()):
        by_state = {str(row.get("pipeline_state")): row for row in rows}
        missing = [state for state in E027_PIPELINE_STATES if state not in by_state]
        group_audit.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "complete": not missing,
                "missing_states": missing,
            }
        )
        stage1 = by_state.get("stage1")
        stage1_deliverable = bool(
            stage1 is not None
            and (_finite(stage1.get("qr_success")) or 0.0) >= 0.5
            and (_finite(stage1.get("qr_tolerance")) or 0.0) >= qr_tolerance_threshold
        )
        full_selected = select_e027_candidate(
            rows,
            qr_tolerance_threshold=qr_tolerance_threshold,
        )
        cascade_selected = (
            select_e027_candidate(
                [stage1],
                qr_tolerance_threshold=qr_tolerance_threshold,
            )
            if stage1_deliverable and stage1 is not None
            else full_selected
        )
        forced_selected = (
            select_e027_candidate(
                [by_state["srmpgd"]],
                qr_tolerance_threshold=qr_tolerance_threshold,
            )
            if "srmpgd" in by_state
            else None
        )
        selections = {
            "cascade": (cascade_selected, 1 if stage1_deliverable else 3),
            "full_lexicographic": (full_selected, 3),
            "forced_srmpgd": (forced_selected, 3),
        }
        for policy, (selected, generation_units) in selections.items():
            row: dict[str, Any] = {
                "prompt_id": prompt_id,
                "seed": seed,
                "policy": policy,
                "technical_complete": not missing,
                "missing_states": ",".join(missing),
                "estimated_generation_units": generation_units,
                "cascade_stopped_after_stage1": bool(
                    policy == "cascade" and stage1_deliverable
                ),
                "selected": selected is not None,
            }
            if selected is None:
                row.update(
                    {
                        "selected_state": None,
                        "selected_method_id": None,
                        "generation_run_id": None,
                        "qr_success": 0.0,
                        "qr_tolerance": 0.0,
                        "deliverable": False,
                        "clip_aesthetic": None,
                        "clip_score": None,
                        "hpsv2_1": None,
                        "saturation_risk": None,
                        "duration_ms": None,
                    }
                )
            else:
                row.update(
                    {
                        "selected_state": selected.get("pipeline_state"),
                        "selected_method_id": selected.get("method_id"),
                        "generation_run_id": selected.get("generation_run_id"),
                        "qr_success": _finite(selected.get("qr_success")) or 0.0,
                        "qr_tolerance": _finite(selected.get("qr_tolerance")) or 0.0,
                        "deliverable": bool(selected.get("deliverable")),
                        "clip_aesthetic": _finite(selected.get("clip_aesthetic")),
                        "clip_score": _finite(selected.get("clip_score")),
                        "hpsv2_1": _finite(selected.get("hpsv2_1")),
                        "saturation_risk": _finite(selected.get("saturation_risk")),
                        "duration_ms": _finite(selected.get("duration_ms")),
                        "output_variant": selected.get("output_variant"),
                        "srmpgd_selected_iteration": _finite(
                            selected.get("srmpgd_selected_iteration")
                        ),
                    }
                )
            decisions.append(row)

    def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
        values = [value for row in rows if (value := _finite(row.get(field))) is not None]
        return sum(values) / len(values) if values else None

    summaries: dict[str, Any] = {}
    for policy in E027_POLICIES:
        rows = [row for row in decisions if row["policy"] == policy]
        exact = sum((_finite(row.get("qr_success")) or 0.0) >= 0.5 for row in rows)
        delivered = sum(bool(row.get("deliverable")) for row in rows)
        delivered_rows = [row for row in rows if bool(row.get("deliverable"))]
        prompt_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            prompt_groups.setdefault(str(row["prompt_id"]), []).append(row)
        prompts_all_seeds_deliverable = sum(
            all(bool(row.get("deliverable")) for row in prompt_rows)
            for prompt_rows in prompt_groups.values()
        )
        summaries[policy] = {
            "contexts": len(rows),
            "technical_complete_contexts": sum(bool(row["technical_complete"]) for row in rows),
            "exact_qr_successes": exact,
            "exact_qr_success_rate": exact / len(rows) if rows else None,
            "exact_qr_wilson_95": _wilson(exact, len(rows)),
            "delivery_gate_successes": delivered,
            "delivery_gate_success_rate": delivered / len(rows) if rows else None,
            "delivery_gate_wilson_95": _wilson(delivered, len(rows)),
            "prompts": len(prompt_groups),
            "prompts_all_seeds_deliverable": prompts_all_seeds_deliverable,
            "prompt_all_seed_success_rate": (
                prompts_all_seeds_deliverable / len(prompt_groups)
                if prompt_groups
                else None
            ),
            "prompt_all_seed_wilson_95": _wilson(
                prompts_all_seeds_deliverable,
                len(prompt_groups),
            ),
            "mean_qr_tolerance": mean(rows, "qr_tolerance"),
            "mean_clip_aesthetic": mean(delivered_rows, "clip_aesthetic"),
            "mean_clip_score": mean(delivered_rows, "clip_score"),
            "mean_hpsv2_1": mean(delivered_rows, "hpsv2_1"),
            "mean_saturation_risk": mean(delivered_rows, "saturation_risk"),
            "estimated_generation_units": sum(
                int(row["estimated_generation_units"]) for row in rows
            ),
            "selected_state_counts": dict(Counter(row.get("selected_state") for row in rows)),
        }

    by_key = {
        (row["prompt_id"], row["seed"], row["policy"]): row for row in decisions
    }
    comparisons: dict[str, Any] = {}
    for left, right in (
        ("full_lexicographic", "forced_srmpgd"),
        ("cascade", "full_lexicographic"),
    ):
        pairs = []
        for prompt_id, seed in groups:
            left_row = by_key[(prompt_id, seed, left)]
            right_row = by_key[(prompt_id, seed, right)]
            pairs.append((left_row, right_row))
        comparisons[f"{left}_vs_{right}"] = {
            "contexts": len(pairs),
            "qr_improvements": sum(
                bool(a["deliverable"]) and not bool(b["deliverable"]) for a, b in pairs
            ),
            "qr_regressions": sum(
                not bool(a["deliverable"]) and bool(b["deliverable"]) for a, b in pairs
            ),
            "same_selected_run": sum(
                bool(a.get("generation_run_id"))
                and a.get("generation_run_id") == b.get("generation_run_id")
                for a, b in pairs
            ),
            "mean_qr_tolerance_delta": (
                sum(float(a["qr_tolerance"]) - float(b["qr_tolerance"]) for a, b in pairs)
                / len(pairs)
                if pairs
                else None
            ),
        }

    return {
        "protocol": "e027-v1-paired-cascade-full-forced-srmpgd",
        "qr_tolerance_threshold": qr_tolerance_threshold,
        "contexts": len(groups),
        "complete_contexts": sum(bool(row["complete"]) for row in group_audit),
        "decisions": decisions,
        "group_audit": group_audit,
        "policies": summaries,
        "paired_comparisons": comparisons,
    }


def e027_policy_winner_entries(
    entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return source generation rows selected by each policy for visual reports."""

    sources = {
        str(row.get("generation_run_id")): dict(row)
        for row in entries
        if row.get("generation_run_id")
    }
    winners = []
    for decision in decisions:
        run_id = str(decision.get("generation_run_id") or "")
        if not run_id or run_id not in sources:
            continue
        item = dict(sources[run_id])
        item.update(
            {
                "section": f"e027_{decision['policy']}",
                "selection_profile": decision["policy"],
                "policy": decision["policy"],
            }
        )
        winners.append(item)
    return winners
