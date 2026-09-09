from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from prooftag_qr.e046_large_catalog import (
    DOE_PATH,
    PROFILE_PROMPT_COUNTS,
    load_doe,
    load_prompts,
    prompts_for_profile,
)
from prooftag_qr.e046_large_doe import (
    ANCHOR_CONFIG,
    ANCHOR_ID,
    CONFIG_COUNT,
    DIMENSIONS,
    PROFILE_CONFIG_COUNTS,
    actual_mask,
    build_doe,
    configs_for_profile,
    parent_seed,
)
from prooftag_qr.resilient_experiment import stable_hash


def _canonical_text(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_doe_is_deterministic_canonical_and_contains_historical_anchor() -> None:
    stored = load_doe()
    rebuilt = build_doe()
    assert rebuilt == stored
    assert _canonical_text(rebuilt) == DOE_PATH.read_text(encoding="utf-8")
    assert stored["config_count"] == CONFIG_COUNT == 12
    assert len(stored["dimensions"]) == len(DIMENSIONS) == 13
    anchor = stored["configs"][0]
    assert anchor["config_id"] == ANCHOR_ID
    assert anchor["role"] == "historical_anchor"
    assert {key: anchor[key] for key in ANCHOR_CONFIG} == ANCHOR_CONFIG
    assert stored["min_pairwise_distance_unit"] >= 1.20
    assert stored["max_abs_correlation"] < 0.60


def test_doe_covers_each_dimension_and_both_ecc_levels() -> None:
    configs = load_doe()["configs"]
    assert Counter(config["error_correction"] for config in configs) == {"M": 6, "Q": 6}
    assert {config["stage2_initialization"] for config in configs} == {
        "paper_stage1_noise",
        "public_random",
    }
    for dimension in DIMENSIONS:
        values = {config[dimension.name] for config in configs}
        if dimension.kind == "categorical":
            assert values == set(dimension.choices or ())
        else:
            assert len(values) >= 6, dimension.name


def test_profile_counts_mask_rotation_and_ecc_coverage() -> None:
    prompts = load_prompts()
    doe = load_doe()
    expected_parents = {"smoke": 8, "pilot": 192, "full": 3072}

    for profile, expected in expected_parents.items():
        profile_prompts = prompts_for_profile(prompts, profile)
        configs = configs_for_profile(doe, profile)
        assert len(profile_prompts) == PROFILE_PROMPT_COUNTS[profile]
        assert len(configs) == PROFILE_CONFIG_COUNTS[profile]
        assert len(profile_prompts) * len(configs) == expected
        masks = Counter(
            actual_mask(config["mask_slot"], prompt.mask_rotation)
            for prompt in profile_prompts
            for config in configs
        )
        assert set(masks) == set(range(8))
        assert max(masks.values()) == min(masks.values())
        ecc = Counter(
            config["error_correction"] for _prompt in profile_prompts for config in configs
        )
        assert set(ecc) == {"M", "Q"}

    full_masks = Counter(
        actual_mask(config["mask_slot"], prompt.mask_rotation)
        for prompt in prompts_for_profile(prompts, "full")
        for config in configs_for_profile(doe, "full")
    )
    assert full_masks == {mask: 384 for mask in range(8)}
    full_ecc = Counter(
        config["error_correction"]
        for _prompt in prompts_for_profile(prompts, "full")
        for config in configs_for_profile(doe, "full")
    )
    assert full_ecc == {"M": 1536, "Q": 1536}


def test_parent_seeds_are_unique_and_reproducible_for_all_3072_parents() -> None:
    seeds = [
        parent_seed(prompt_ordinal, config_index)
        for prompt_ordinal in range(1, 257)
        for config_index in range(12)
    ]
    assert len(seeds) == len(set(seeds)) == 3072
    assert seeds == [
        parent_seed(prompt_ordinal, config_index)
        for prompt_ordinal in range(1, 257)
        for config_index in range(12)
    ]
    with pytest.raises(ValueError):
        parent_seed(0, 0)
    with pytest.raises(ValueError):
        parent_seed(1, 12)


def test_doe_rejects_semantic_tampering_even_with_recomputed_hash(tmp_path) -> None:
    tampered = copy.deepcopy(load_doe())
    tampered["configs"][0]["stage1_steps"] = 41
    tampered["doe_sha256"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "doe_sha256"}
    )
    path = tmp_path / "tampered-doe.json"
    path.write_text(_canonical_text(tampered), encoding="utf-8")
    with pytest.raises(AssertionError):
        load_doe(path)


def test_unknown_profile_cannot_expand_to_full() -> None:
    with pytest.raises(ValueError, match="unknown E046 large profile"):
        configs_for_profile(load_doe(), "default")
