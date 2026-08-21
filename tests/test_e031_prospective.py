import hashlib
from copy import deepcopy

import pytest

from prooftag_qr.e031_prospective import (
    E031_BRANCHES,
    E031_PRIMARY_SEED,
    E031_RETRY_SEED,
    audit_e031_pairing,
    build_e031_holdout_prompts,
    build_e031_prospective_plan,
    candidate_pool_sha256,
    e031_candidate_rank,
    enrich_e031_stage2_results,
    evaluate_e031_policies,
    recommend_e031_advisor_chains,
    select_e031_candidate,
    wilson_interval,
)
from prooftag_qr.lab import DIFFQRCODER_MODEL_SETTINGS, laboratory_profiles
from prooftag_qr.parameter_advisor import ParameterRecommendation
from prooftag_qr.policy import ConservativeDeliveryGate, assess_stage2_candidate


def _profiles():
    return {item["id"]: deepcopy(item) for item in laboratory_profiles()}


def _advisor_chain(*, strength: float = 0.80, prediction: float = 0.75):
    profiles = _profiles()
    stage1 = profiles["diffqrcoder_stage1"]
    stage1["generation"]["controlnet_scale"] = 1.55
    stage2 = profiles["diffqrcoder_srpg"]
    stage2["tools"]["settings"]["diffqrcoder_stage2_strength"] = strength
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage1_prediction": {"predicted_qr_success": prediction - 0.1},
        "stage2_prediction": {
            "predicted_qr_success": prediction,
            "predicted_hpsv2_1": 0.2,
        },
    }


def _small_plan(*, second_strength: float = 0.80, second_prediction: float = 0.76):
    prompts = build_e031_holdout_prompts(2)
    return build_e031_prospective_plan(
        prompts=prompts,
        payload="https://ptag.io/t/e031",
        advisor_chains={
            prompts[0]["id"]: _advisor_chain(strength=0.80, prediction=0.75),
            prompts[1]["id"]: _advisor_chain(
                strength=second_strength,
                prediction=second_prediction,
            ),
        },
        advisor_sha256="a" * 64,
        candidate_pool_sha256="b" * 64,
    )


def test_e031_frozen_prompt_bank_is_balanced_unique_and_fail_closed_on_overlap():
    prompts = build_e031_holdout_prompts()

    assert len(prompts) == 40
    assert sum("_simple_" in item["id"] for item in prompts) == 20
    assert sum("_atypical_" in item["id"] for item in prompts) == 20
    assert len({" ".join(item["text"].split()).casefold() for item in prompts}) == 40
    assert all(item["negative_prompt"] for item in prompts)

    with pytest.raises(ValueError, match="overlaps"):
        build_e031_holdout_prompts(
            excluded_prompt_texts=[prompts[7]["text"]],
        )
    with pytest.raises(ValueError, match="even"):
        build_e031_holdout_prompts(3)


def test_e031_plan_has_exactly_three_paired_stage2_branches_and_no_srmpgd():
    plan = _small_plan()

    assert plan.public["protocol"] == "e031-v1-static-paired-three-stage2-branches"
    assert plan.public["prompt_count"] == 2
    assert plan.public["stage1_trial_count"] == 6
    assert plan.public["stage2_trial_count"] == 6
    assert plan.public["trial_count"] == 12
    assert plan.public["srmpgd_trial_count"] == 0
    assert plan.public["branches"] == list(E031_BRANCHES)
    assert plan.public["primary_seed"] == E031_PRIMARY_SEED
    assert plan.public["retry_seed"] == E031_RETRY_SEED
    assert len(plan.campaigns) == 3

    for campaign in plan.campaigns:
        assert len(campaign["methods"]) == 2
        stage1, stage2 = campaign["methods"]
        assert stage1["model"] == DIFFQRCODER_MODEL_SETTINGS
        assert stage2["model"] == DIFFQRCODER_MODEL_SETTINGS
        assert stage1["output_variant"] == "raw"
        assert stage2["output_variant"] == "srpg"
        assert stage2["reuse_stage1"] is True
        assert stage2["require_exact_stage1_reuse"] is True
        assert stage2["tools"]["srpg_enabled"] is True
        assert stage2["tools"]["srmpgd_enabled"] is False
        assert not any(
            key.startswith("srmpgd_")
            for key in stage2["tools"]["settings"]
        )
        assert (
            stage2["tools"]["settings"]["diffqrcoder_stage2_target_mode"]
            == "binary_exact"
        )

    stage2_predictions = [
        item for item in plan.predictions if item["pipeline_state"] == "stage2"
    ]
    assert len(stage2_predictions) == 6
    assert {
        (item["prompt_id"], item["branch_id"])
        for item in stage2_predictions
    } == {
        (prompt["id"], branch)
        for prompt in build_e031_holdout_prompts(2)
        for branch in E031_BRANCHES
    }


def test_e031_plan_rejects_any_model_revision_or_geometry_drift():
    prompts = build_e031_holdout_prompts(2)
    fixed_stage1 = _profiles()["diffqrcoder_stage1"]
    fixed_stage1["model"]["controlnet_model_revision"] = "0" * 40

    with pytest.raises(
        ValueError,
        match=r"model\.controlnet_model_revision",
    ):
        build_e031_prospective_plan(
            prompts=prompts,
            payload="https://ptag.io/t/e031-drift",
            advisor_chains={prompt["id"]: _advisor_chain() for prompt in prompts},
            advisor_sha256="a" * 64,
            candidate_pool_sha256="b" * 64,
            fixed_stage1=fixed_stage1,
        )

    advisor_chains = {prompt["id"]: _advisor_chain() for prompt in prompts}
    advisor_chains[prompts[0]["id"]]["stage2"]["model"][
        "diffqrcoder_qr_mask_pattern"
    ] = 5
    with pytest.raises(
        ValueError,
        match=r"model\.diffqrcoder_qr_mask_pattern",
    ):
        build_e031_prospective_plan(
            prompts=prompts,
            payload="https://ptag.io/t/e031-drift",
            advisor_chains=advisor_chains,
            advisor_sha256="a" * 64,
            candidate_pool_sha256="b" * 64,
        )


def test_e031_full_frozen_plan_is_exactly_120_stage1_plus_120_stage2_trials():
    prompts = build_e031_holdout_prompts()
    plan = build_e031_prospective_plan(
        prompts=prompts,
        payload="https://ptag.io/t/e031-full",
        advisor_chains={prompt["id"]: _advisor_chain() for prompt in prompts},
        advisor_sha256="a" * 64,
        candidate_pool_sha256="b" * 64,
    )

    assert plan.public["prompt_count"] == 40
    assert plan.public["stage1_trial_count"] == 120
    assert plan.public["stage2_trial_count"] == 120
    assert plan.public["trial_count"] == 240
    assert sum(
        len(campaign["prompts"])
        * len(campaign["seeds"])
        * len(campaign["methods"])
        for campaign in plan.campaigns
    ) == 240


def test_e031_plan_groups_advisor_prompts_by_effective_recipe():
    grouped = _small_plan(second_strength=0.80)
    split = _small_plan(second_strength=0.50)

    assert grouped.public["advisor_recipe_groups"] == 1
    assert grouped.public["campaign_count"] == 3
    assert split.public["advisor_recipe_groups"] == 2
    assert split.public["campaign_count"] == 4
    assert split.public["trial_count"] == grouped.public["trial_count"] == 12


def test_e031_plan_id_is_bound_to_full_advisor_predictions():
    first = _small_plan(second_prediction=0.76)
    changed = _small_plan(second_prediction=0.77)

    assert first.public["prediction_sha256"] != changed.public["prediction_sha256"]
    assert first.plan_id != changed.plan_id


class _DeterministicFakeAdvisor:
    def __init__(self, *, reverse: bool):
        self.reverse = reverse
        self.calls = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        recommendations = []
        for position, candidate in enumerate(kwargs["candidates"], start=1):
            fraction = int(candidate.signature[:8], 16) / 0xFFFFFFFF
            probability = 0.70 + 0.25 * fraction
            recommendations.append(
                ParameterRecommendation(
                    rank=position,
                    candidate=candidate,
                    scan_safe=probability >= kwargs["scan_probability_threshold"],
                    predicted_qr_success=probability,
                    qr_success_uncertainty=0.01,
                    qr_success_lower_bound=probability - 0.01,
                    predicted_qr_tolerance=probability,
                    predicted_clip_aesthetic=4.0 + fraction,
                    predicted_clip_score=0.5 + fraction / 4,
                    predicted_hpsv2_1=0.2 + fraction / 10,
                    predicted_human_aesthetic=None,
                    predicted_human_prompt_fidelity=None,
                    predicted_human_qr_discretion=None,
                    predicted_human_overall=None,
                    predicted_duration_ms=1000.0 + 10.0 * fraction,
                    predicted_saturation_risk=0.01 + 0.02 * fraction,
                )
            )
        return list(reversed(recommendations)) if self.reverse else recommendations


def test_e031_recommends_one_conditioned_chain_per_prompt_deterministically():
    prompts = build_e031_holdout_prompts(4)
    first_advisor = _DeterministicFakeAdvisor(reverse=False)
    reversed_advisor = _DeterministicFakeAdvisor(reverse=True)
    arguments = {
        "candidates": [],
        "prompts": prompts,
        "payload": "https://ptag.io/t/e031-advisor",
        "prompt_embedding_provider": lambda text: [float(len(text))],
    }

    first = recommend_e031_advisor_chains(first_advisor, **arguments)
    reordered = recommend_e031_advisor_chains(reversed_advisor, **arguments)

    assert set(first) == {prompt["id"] for prompt in prompts}
    assert {
        prompt_id: (
            chain["stage1"].signature,
            chain["stage2"].signature,
        )
        for prompt_id, chain in first.items()
    } == {
        prompt_id: (
            chain["stage1"].signature,
            chain["stage2"].signature,
        )
        for prompt_id, chain in reordered.items()
    }
    assert len(first_advisor.calls) == 2 * len(prompts)
    assert all(
        call["context_features"] is None
        for call in first_advisor.calls[0::2]
    )
    assert all(
        call["context_features"]["parent_qr_success"] is not None
        for call in first_advisor.calls[1::2]
    )
    for chain in first.values():
        assert chain["stage1_prediction"].candidate is chain["stage1"]
        assert chain["stage2_prediction"].candidate is chain["stage2"]
        assert chain["stage2"].configuration["output_variant"] == "srpg"
        tools = chain["stage2"].configuration["tools"]
        assert tools["srpg_enabled"] is True
        assert tools["srmpgd_enabled"] is False
        assert not any(key.startswith("srmpgd_") for key in tools["settings"])

    pool_hash = candidate_pool_sha256([])
    assert len(pool_hash) == 64
    assert pool_hash == candidate_pool_sha256([])

    plan = build_e031_prospective_plan(
        prompts=prompts,
        payload=arguments["payload"],
        advisor_chains=first,
        advisor_sha256="a" * 64,
        candidate_pool_sha256=pool_hash,
    )
    assert plan.public["stage2_trial_count"] == 3 * len(prompts)
    assert plan.public["srmpgd_trial_count"] == 0


def _paired_rows():
    rows = []
    prompt_id = "e031h_simple_001"
    for branch, seed in (
        ("fixed_seed_a", E031_PRIMARY_SEED),
        ("advisor_seed_a", E031_PRIMARY_SEED),
        ("fixed_seed_b", E031_RETRY_SEED),
    ):
        stage1_method = f"{branch}-stage1"
        stage2_method = f"{branch}-stage2"
        stage1_run = f"{branch}-stage1-run"
        stage1_hash = hashlib.sha256(f"{branch}-stage1".encode()).hexdigest()
        stage2_hash = hashlib.sha256(f"{branch}-stage2".encode()).hexdigest()
        campaign_id = f"campaign-{branch}"
        stage1_configuration = _profiles()["diffqrcoder_stage1"]
        stage1_configuration.update(
            {"id": stage1_method, "name": f"{branch} Stage 1"}
        )
        stage2_configuration = _profiles()["diffqrcoder_srpg"]
        stage2_configuration.update(
            {"id": stage2_method, "name": f"{branch} Stage 2"}
        )
        rows.extend(
            [
                {
                    "prompt_id": prompt_id,
                    "branch_id": branch,
                    "pipeline_state": "stage1",
                    "output_variant": "raw",
                    "seed": float(seed),
                    "status": "rejected",
                    "method_id": stage1_method,
                    "campaign_id": campaign_id,
                    "generation_run_id": stage1_run,
                    "final_image_sha256": stage1_hash,
                    "candidate_configuration": stage1_configuration,
                },
                {
                    "prompt_id": prompt_id,
                    "branch_id": branch,
                    "pipeline_state": "stage2",
                    "output_variant": "srpg",
                    "requested_source_output_variant": "srpg",
                    "seed": float(seed),
                    "status": "accepted",
                    "method_id": stage2_method,
                    "parent_stage1_method_id": stage1_method,
                    "campaign_id": campaign_id,
                    "generation_run_id": f"{branch}-stage2-run",
                    "stage1_reused": 1.0,
                    "stage1_source_run_id": stage1_run,
                    "stage1_image_sha256": stage1_hash,
                    "final_image_sha256": stage2_hash,
                    "saturation_risk": 0.01,
                    "hpsv2_1": 0.2,
                    "clip_score": 0.7,
                    "clip_aesthetic": 5.0,
                    "candidate_configuration": stage2_configuration,
                },
            ]
        )
    return rows


def test_e031_pairing_audit_proves_exact_parent_and_explains_mismatch():
    rows = _paired_rows()
    audit = audit_e031_pairing(rows)

    assert len(audit) == 3
    assert all(item["complete"] for item in audit)
    assert {item["seed"] for item in audit} == {
        float(E031_PRIMARY_SEED),
        float(E031_RETRY_SEED),
    }
    assert all(item["seed"] == item["expected_seed"] for item in audit)
    fixed = [item for item in audit if item["branch_id"].startswith("fixed_")]
    assert len({item["stage1_recipe_identity_sha256"] for item in fixed}) == 1
    assert len({item["stage2_recipe_identity_sha256"] for item in fixed}) == 1

    broken = deepcopy(rows)
    stage2 = next(
        item
        for item in broken
        if item["pipeline_state"] == "stage2"
        and item["branch_id"] == "advisor_seed_a"
    )
    stage2["stage1_image_sha256"] = "f" * 64
    failed = audit_e031_pairing(broken)
    advisor = next(item for item in failed if item["branch_id"] == "advisor_seed_a")
    assert advisor["complete"] is False
    assert "stage1_raster_hash_mismatch" in advisor["failure_reasons"]


def test_e031_pairing_audit_rejects_wrong_branch_seed_even_when_parent_matches():
    rows = _paired_rows()
    for row in rows:
        if row["branch_id"] == "fixed_seed_a":
            row["seed"] = 42.0

    audit = audit_e031_pairing(rows)
    fixed = next(item for item in audit if item["branch_id"] == "fixed_seed_a")

    assert fixed["complete"] is False
    assert f"branch_seed_unexpected:{E031_PRIMARY_SEED}" in fixed["failure_reasons"]
    assert "stage1_seed_mismatch" not in fixed["failure_reasons"]


@pytest.mark.parametrize("phase", ["stage1", "stage2"])
def test_e031_pairing_audit_rejects_fixed_recipe_drift(phase):
    rows = _paired_rows()
    target = next(
        row
        for row in rows
        if row["branch_id"] == "fixed_seed_b"
        and row["pipeline_state"] == phase
    )
    target["candidate_configuration"]["generation"]["guidance_scale"] = 6.25

    audit = audit_e031_pairing(rows)
    fixed = [
        item
        for item in audit
        if item["branch_id"] in {"fixed_seed_a", "fixed_seed_b"}
    ]

    assert all(item["complete"] is False for item in fixed)
    assert all(
        f"fixed_{phase}_recipe_mismatch" in item["failure_reasons"]
        for item in fixed
    )


def _score(raster_hash: str, exact_presets: int):
    tolerance = exact_presets / 37
    return {
        "image_sha256": raster_hash,
        "payload_sha256": "c" * 64,
        "cache_key": hashlib.sha256((raster_hash + "cache").encode()).hexdigest(),
        "engine_version": "qr-verify@0.2.0",
        "implementation_sha256": "d" * 64,
        "scoring_version": "qr-verify-conservative-v1",
        "repetitions": 5,
        "preset_count": 37,
        "each_repetition_any_exact": exact_presets > 0,
        "conservative_exact_presets": exact_presets,
        "conservative_tolerance_score": tolerance,
        "unstable_preset_count": 0,
        "runs": [
            {
                "repetition": repetition,
                "any_exact": exact_presets > 0,
                "tolerance_score": tolerance,
            }
            for repetition in range(1, 6)
        ],
    }


def _enriched_rows():
    rows = _paired_rows()
    stage2 = [item for item in rows if item["pipeline_state"] == "stage2"]
    presets = {
        "fixed_seed_a": 30,
        "advisor_seed_a": 36,
        "fixed_seed_b": 37,
    }
    scores = {
        item["final_image_sha256"]: _score(
            item["final_image_sha256"], presets[item["branch_id"]]
        )
        for item in stage2
    }
    return enrich_e031_stage2_results(rows, scores)


def test_e031_enrichment_binds_five_repetitions_to_the_exact_raster():
    enriched = _enriched_rows()

    assert len(enriched) == 3
    assert all(len(item["qr_verify_observations"]) == 5 for item in enriched)
    assert all(item["qr_verify_preset_count"] == 37 for item in enriched)
    assert all(
        {
            observation["image_sha256"]
            for observation in item["qr_verify_observations"]
        }
        == {item["final_image_sha256"]}
        for item in enriched
    )

    rows = _paired_rows()
    stage2 = next(item for item in rows if item["pipeline_state"] == "stage2")
    stage2["downloaded_raster_sha256"] = "e" * 64
    scores = {stage2["final_image_sha256"]: _score(stage2["final_image_sha256"], 37)}
    with pytest.raises(ValueError, match="downloaded raster differs"):
        enrich_e031_stage2_results([stage2], scores)


def test_e031_policy_replay_uses_standard_and_effective_36_of_37_strict_gates():
    report = evaluate_e031_policies(_enriched_rows())

    decisions = report["decisions"]
    standard = next(
        item
        for item in decisions
        if item["gate"] == "standard"
        and item["policy"] == "fixed_advisor_then_seed_retry"
    )
    strict = next(
        item
        for item in decisions
        if item["gate"] == "strict"
        and item["policy"] == "fixed_advisor_then_seed_retry"
    )
    oracle = next(
        item
        for item in decisions
        if item["gate"] == "strict" and item["policy"] == "best_of_three"
    )

    assert standard["selected_branch"] == "fixed_seed_a"
    assert standard["stage2_attempts_used"] == 1
    assert standard["api_trials_used"] == 2
    assert strict["gate_effective_exact_presets"] == 36
    assert strict["selected_branch"] == "advisor_seed_a"
    assert strict["stage2_attempts_used"] == 2
    assert strict["api_trials_used"] == 4
    assert oracle["selected_branch"] == "fixed_seed_b"
    assert oracle["stage2_attempts_used"] == 3

    strict_summary = next(
        item
        for item in report["summary"]
        if item["gate"] == "strict"
        and item["policy"] == "fixed_advisor_then_seed_retry"
    )
    assert strict_summary["delivered"] == 1
    assert strict_summary["delivery_rate"] == 1.0
    assert strict_summary["total_effective_api_trials"] == 4
    assert strict_summary["stage1_was_delivered"] is False
    assert strict_summary["srmpgd_was_requested"] is False


def test_e031_lexicographic_selection_is_qr_then_saturation_then_quality():
    rows = _enriched_rows()
    gate = ConservativeDeliveryGate(
        qr_tolerance_threshold=0.80,
        saturation_threshold=0.05,
        minimum_qr_observations=5,
    )
    fixed = next(item for item in rows if item["branch_id"] == "fixed_seed_b")
    advisor = next(item for item in rows if item["branch_id"] == "advisor_seed_a")
    advisor["conservative_qr_tolerance"] = fixed["conservative_qr_tolerance"]
    advisor["conservative_exact_presets"] = fixed["conservative_exact_presets"]
    for observation in advisor["qr_verify_observations"]:
        observation["qr_tolerance"] = fixed["conservative_qr_tolerance"]
    advisor["saturation_risk"] = 0.005
    for observation in advisor["qr_verify_observations"]:
        observation["saturation_risk"] = 0.005
    advisor["hpsv2_1"] = 0.01
    fixed["hpsv2_1"] = 0.99

    selected = select_e031_candidate([fixed, advisor], gate=gate)

    assert selected is advisor
    assert e031_candidate_rank(advisor, assess_stage2_candidate(advisor, gate)) > (
        e031_candidate_rank(fixed, assess_stage2_candidate(fixed, gate))
    )


def test_e031_wilson_interval_is_ordered_and_never_negative():
    low, high = wilson_interval(10, 10)
    assert 0.0 <= low <= 1.0
    assert high == pytest.approx(1.0)

    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert 0.0 <= high <= 1.0

    with pytest.raises(ValueError):
        wilson_interval(1, 0)
