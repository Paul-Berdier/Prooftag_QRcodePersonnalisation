"""Group-split, training-contract and determinism guards for E046 large."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

import pytest

from prooftag_qr.e046_large_campaign import (
    ADVISOR_FEATURE_COLUMNS,
    ADVISOR_TARGET_COLUMNS,
    GROUP_KFOLD_COUNT,
    SPLIT_GROUP_KEYS,
    advisor_training_contract,
    build_group_splits,
    prompt_group_fold,
    validate_group_split,
)
from prooftag_qr.e046_large_catalog import FAMILIES, load_prompts

ROOT = Path(__file__).resolve().parents[1]


def _fake_rows(prompts, *, configs=2, trajectories=1, iterations=2):
    rows = []
    for prompt in prompts:
        payload_sha = hashlib.sha256(prompt.payload.encode("utf-8")).hexdigest()
        for config in range(configs):
            candidate_id = f"p{prompt.ordinal:04d}_k{config:02d}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "prompt_id": prompt.prompt_id,
                    "prompt_family": prompt.family,
                    "payload_sha256": payload_sha,
                    "generation_group_id": candidate_id,
                    "trajectory_id": None,
                    "image_sha256": f"parent-{candidate_id}",
                    "observation_independence_class": "independent_parent",
                }
            )
            for trajectory in range(trajectories):
                trajectory_id = f"{candidate_id}__recipe{trajectory}"
                for iteration in range(iterations + 1):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "prompt_id": prompt.prompt_id,
                            "prompt_family": prompt.family,
                            "payload_sha256": payload_sha,
                            "generation_group_id": candidate_id,
                            "trajectory_id": trajectory_id,
                            "image_sha256": f"{trajectory_id}-i{iteration}",
                            "observation_independence_class": (
                                "correlated_srmpgd_checkpoint"
                            ),
                        }
                    )
    return rows


def test_group_splits_keep_prompt_payload_generation_and_trajectory_together() -> None:
    prompts = load_prompts()
    subset = prompts[:40]
    rows = _fake_rows(subset)
    splits = build_group_splits(rows, prompts)

    assert splits["random_row_split_allowed"] is False
    assert splits["fold_unit"] == "prompt_id"
    assert splits["group_keys"] == list(SPLIT_GROUP_KEYS)
    assert len(splits["row_assignments"]) == len(rows)

    by_prompt = splits["group_kfold_by_prompt"]["prompt_folds"]
    for item in splits["row_assignments"]:
        assert item["group_kfold_by_prompt"] == by_prompt[item["prompt_id"]]
        assert item["leave_family_out"] == next(
            prompt.family for prompt in subset if prompt.prompt_id == item["prompt_id"]
        )

    # Every checkpoint of a trajectory shares the fold of its parent row.
    folds_by_trajectory: dict[str, set[int]] = {}
    parent_fold = {}
    for item in splits["row_assignments"]:
        if item["trajectory_id"] is None:
            parent_fold[item["generation_group_id"]] = item["group_kfold_by_prompt"]
        else:
            folds_by_trajectory.setdefault(item["trajectory_id"], set()).add(
                item["group_kfold_by_prompt"]
            )
    assert folds_by_trajectory
    for trajectory_id, folds in folds_by_trajectory.items():
        assert len(folds) == 1
        assert folds == {parent_fold[trajectory_id.split("__")[0]]}


def test_validate_group_split_rejects_a_trajectory_split_across_folds() -> None:
    assignments = [
        {
            "prompt_id": "p",
            "payload_sha256": "s",
            "generation_group_id": "g",
            "trajectory_id": "t",
            "fold": 0,
        },
        {
            "prompt_id": "p",
            "payload_sha256": "s",
            "generation_group_id": "g",
            "trajectory_id": "t",
            "fold": 1,
        },
    ]
    with pytest.raises(ValueError, match="span several folds"):
        validate_group_split(assignments, "fold")

    # Different prompts in different folds is the intended shape.
    validate_group_split(
        [
            {
                **assignments[0],
                "prompt_id": "a",
                "generation_group_id": "ga",
                "trajectory_id": "ta",
                "payload_sha256": "sa",
            },
            {
                **assignments[1],
                "prompt_id": "b",
                "generation_group_id": "gb",
                "trajectory_id": "tb",
                "payload_sha256": "sb",
            },
        ],
        "fold",
    )


def test_group_kfold_by_prompt_is_family_stratified_over_the_full_catalog() -> None:
    prompts = load_prompts()
    rows = _fake_rows(prompts, configs=1, trajectories=0)
    splits = build_group_splits(rows, prompts)
    kfold = splits["group_kfold_by_prompt"]
    assert kfold["fold_count"] == GROUP_KFOLD_COUNT == 5
    assert kfold["prompt_counts_per_fold"] == {"0": 64, "1": 48, "2": 48, "3": 48, "4": 48}
    for family in FAMILIES:
        folds = Counter(
            prompt_group_fold(prompt.design_row)
            for prompt in prompts
            if prompt.family == family.family
        )
        assert set(folds) == set(range(GROUP_KFOLD_COUNT))
    leave = splits["leave_family_out"]
    assert leave["fold_count"] == 16
    assert set(leave["prompt_folds"].values()) == {family.family for family in FAMILIES}


def test_advisor_training_contract_declares_interfaces_but_no_trained_model() -> None:
    prompts = load_prompts()
    splits = build_group_splits(_fake_rows(prompts[:8]), prompts)
    contract = advisor_training_contract(
        {
            "plan_id": "0123456789abcdef",
            "profile": "smoke",
            "source_commit": "0" * 40,
            "runtime_image_digest": "sha256:" + "0" * 64,
        },
        splits=splits,
        dataset_hashes={"parent-observations.jsonl": "0" * 64},
    )
    assert contract["final_advisor_trained"] is False
    assert contract["automatic_training_authorized"] is False
    assert contract["sufficient_for_final_advisor"] is False
    assert contract["row_random_split_forbidden"] is True
    assert contract["same_trajectory_cross_split_forbidden"] is True
    assert contract["feature_columns"] == list(ADVISOR_FEATURE_COLUMNS)
    assert contract["target_columns"] == list(ADVISOR_TARGET_COLUMNS)
    for target in (
        "wechat_exact_presets",
        "wechat_original_exact",
        "clip_score",
        "hpsv2_1",
        "clip_aesthetic",
        "module_error_rate",
        "multiobjective_prompt_score",
    ):
        assert target in contract["target_columns"]
    for feature in (
        "stage1_steps",
        "stage1_guidance_scale",
        "stage1_controlnet_scale",
        "control_guidance_start",
        "control_guidance_end",
        "stage2_initialization",
        "stage2_strength",
        "stage2_steps",
        "stage2_controlnet_scale",
        "stage2_qr_weight",
        "stage2_perceptual_weight",
        "error_correction",
        "qr_mask_pattern",
        "payload_length",
        "gamma",
        "latent_radius_rms",
    ):
        assert feature in contract["feature_columns"]
    assert contract["multiobjective_score_is_derived_not_primary"] is True


def test_large_modules_contain_no_unseeded_randomness() -> None:
    """Scientific identities (catalog, DOE, seeds, candidates) never depend on
    process randomness. ``uuid4`` is tolerated only for attempt/staging paths."""
    forbidden = (
        r"\brandom\.random\(",
        r"\brandom\.shuffle\(",
        r"\brandom\.choice",
        r"\brandom\.randint\(",
        r"\brandom\.sample\(",
        r"\bnp\.random\.",
        r"\bnumpy\.random\.",
        r"\btorch\.rand",
        r"\bsecrets\.",
        r"\btime\.time\(\)",
    )
    for name in (
        "e046_large_catalog.py",
        "e046_large_prompts.py",
        "e046_large_doe.py",
        "e046_large_campaign.py",
    ):
        text = (ROOT / "prooftag_qr" / name).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text), (name, pattern)
        for line in text.splitlines():
            if "uuid" in line and "import uuid" not in line:
                allowed = ("attempt", "staging", "temporary", "uuid4().hex")
                assert any(token in line for token in allowed), (name, line)
    doe_text = (ROOT / "prooftag_qr/e046_large_doe.py").read_text(encoding="utf-8")
    generator_uses = re.findall(r"random\.\w+\(", doe_text)
    assert set(generator_uses) == {"random.Random("}
    for name in ("e046_large_catalog.py", "e046_large_prompts.py"):
        assert "import random" not in (ROOT / "prooftag_qr" / name).read_text(encoding="utf-8")
