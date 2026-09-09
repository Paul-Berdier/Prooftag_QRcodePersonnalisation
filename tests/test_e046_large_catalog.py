from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import pytest
import qrcode
from qrcode.constants import ERROR_CORRECT_M, ERROR_CORRECT_Q

from prooftag_qr.e046_large_catalog import (
    CATALOG_PATH,
    FAMILIES,
    FAMILY_COUNT,
    GEOMETRY_TYPES,
    PROFILE_PROMPT_COUNTS,
    PROMPT_COUNT,
    PROMPTS_PER_FAMILY,
    TAG_AXES,
    build_prompt_specs,
    catalog_document,
    catalog_file_sha256,
    load_catalog,
    load_prompts,
    payload_fits,
    prompt_similarity,
    prompts_for_profile,
)
from prooftag_qr.e046_large_prompts import FAMILY_PROMPTS
from prooftag_qr.resilient_experiment import stable_hash


def _canonical_text(document: dict) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def test_canonical_catalog_has_exact_cardinality_and_unique_content() -> None:
    document = load_catalog()
    prompts = load_prompts()

    assert document["prompt_count"] == PROMPT_COUNT == 256
    assert document["family_count"] == FAMILY_COUNT == 16
    assert document["prompts_per_family"] == PROMPTS_PER_FAMILY == 16
    assert len({item.prompt_id for item in prompts}) == 256
    assert len({item.prompt for item in prompts}) == 256
    assert len({item.subject for item in prompts}) == 256
    assert len({item.payload for item in prompts}) == 256
    assert Counter(item.family for item in prompts) == {family.family: 16 for family in FAMILIES}
    assert [item.payload for item in prompts] == [
        f"https://ptag.io/d2/{ordinal:04d}" for ordinal in range(1, 257)
    ]
    assert {item.payload_length for item in prompts} == {23}
    assert len(document["catalog_sha256"]) == 64
    assert catalog_file_sha256() == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def test_catalog_rebuild_is_byte_identical_to_versioned_json() -> None:
    rebuilt = catalog_document(build_prompt_specs(FAMILY_PROMPTS))
    assert rebuilt == load_catalog()
    assert _canonical_text(rebuilt) == CATALOG_PATH.read_text(encoding="utf-8")


def test_structural_grid_is_balanced_inside_every_family() -> None:
    prompts = load_prompts()
    for family in FAMILIES:
        members = [item for item in prompts if item.family == family.family]
        assert {item.design_row for item in members} == set(range(1, 17))
        assert {item.geometry_type for item in members} == {family.geometry_type}
        for axis, levels in TAG_AXES.items():
            counts = Counter(getattr(item, axis) for item in members)
            assert set(counts) == set(levels)
            assert max(counts.values()) - min(counts.values()) <= 1

    assert {item.geometry_type for item in prompts} == set(GEOMETRY_TYPES)


def test_subject_clauses_have_no_manifest_near_duplicates() -> None:
    prompts = load_prompts()
    worst = max(
        (
            prompt_similarity(left.subject, right.subject),
            left.prompt_id,
            right.prompt_id,
        )
        for index, left in enumerate(prompts)
        for right in prompts[index + 1 :]
    )
    assert worst[0] < 0.60, worst


def test_all_payloads_fit_real_qr_version_3_m_and_q() -> None:
    corrections = (ERROR_CORRECT_M, ERROR_CORRECT_Q)
    for prompt in load_prompts():
        assert payload_fits(prompt.payload, "M")
        assert payload_fits(prompt.payload, "Q")
        for correction in corrections:
            qr = qrcode.QRCode(version=3, error_correction=correction, box_size=1, border=4)
            qr.add_data(prompt.payload)
            qr.make(fit=False)
            assert qr.version == 3


def test_profile_prompt_subsets_are_deterministic_and_balanced() -> None:
    prompts = load_prompts()
    selected = {profile: prompts_for_profile(prompts, profile) for profile in PROFILE_PROMPT_COUNTS}
    assert {profile: len(items) for profile, items in selected.items()} == {
        "smoke": 4,
        "pilot": 32,
        "full": 256,
    }
    assert {item.geometry_type for item in selected["smoke"]} == set(GEOMETRY_TYPES)
    assert {item.design_row for item in selected["smoke"]} == {1, 3, 5, 7}
    assert sorted(item.mask_rotation for item in selected["smoke"]) == [0, 2, 4, 6]
    # The historical brutalist anchor (ordinal 1, payload d2/0001) is part of
    # every profile, so the anchor DOE configuration reproduces the E044 parent
    # with its exact mask 4 even in smoke.
    for profile in ("smoke", "pilot", "full"):
        anchors = [item for item in selected[profile] if item.is_historical_anchor]
        assert [item.ordinal for item in anchors] == [1], profile
        assert anchors[0].design_row == 5
        assert anchors[0].tags == {
            "spatial_frequency": "medium",
            "symmetry": "symmetric",
            "brightness": "mixed",
            "contrast": "low",
            "colorfulness": "muted",
            "negative_space": "high",
            "texture_density": "fine",
            "composition_type": "centered",
        }
    assert Counter(item.family for item in selected["pilot"]) == {
        family.family: 2 for family in FAMILIES
    }
    assert Counter(item.design_row for item in selected["pilot"]) == {
        row: 2 for row in range(1, 17)
    }
    assert tuple(item.prompt_id for item in prompts_for_profile(prompts, "pilot")) == tuple(
        item.prompt_id for item in selected["pilot"]
    )
    with pytest.raises(ValueError, match="unknown E046 large profile"):
        prompts_for_profile(prompts, "accidental-full")


def test_catalog_rejects_semantic_tampering_even_with_recomputed_hash(tmp_path) -> None:
    tampered = copy.deepcopy(load_catalog())
    tampered["prompts"][1]["payload"] = tampered["prompts"][0]["payload"]
    tampered["catalog_sha256"] = stable_hash(
        {key: value for key, value in tampered.items() if key != "catalog_sha256"}
    )
    path = tmp_path / "tampered-catalog.json"
    path.write_text(_canonical_text(tampered), encoding="utf-8")
    with pytest.raises(AssertionError):
        load_catalog(path)
