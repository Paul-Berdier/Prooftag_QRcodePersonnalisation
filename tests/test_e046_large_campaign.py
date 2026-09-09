from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import prooftag_qr.e046_large_campaign as large_campaign
from prooftag_qr.config import Settings
from prooftag_qr.e046_catalog import ParentRecipe
from prooftag_qr.e046_large_campaign import (
    DATASET_SCHEMA,
    DEFAULT_OUTPUT_ROOT,
    EXPERIMENT,
    PARENT_GENERATION_REQUIRED,
    PLAN_SCHEMA,
    _assert_runtime_provenance,
    _best_by_prompt,
    _bind_e046_iteration_zero_to_parent,
    _dataset_summary,
    _parameter_coverage,
    _promote_e046_attempt,
    _promotion_valid,
    _recipe_features,
    _record_failure,
    _runtime_scientific_contract,
    _technical_failure_audit,
    _terminal_srl_failure,
    create_plan,
    list_parent_ids,
    list_refinement_ids,
    load_plan,
    plan_summary,
    scientific_plan,
    verify,
)
from prooftag_qr.e046_large_catalog import (
    CATALOG_PATH,
    DOE_PATH,
    catalog_file_sha256,
    doe_file_sha256,
)
from prooftag_qr.resilient_experiment import (
    atomic_write_json,
    promote_attempt,
    sha256_file,
)

SOURCE_COMMIT = "a" * 40
RUNTIME_IMAGE = "prooftag-qr:test"
RUNTIME_DIGEST = "sha256:" + "b" * 64


def _frozen_runtime_contract() -> dict:
    settings = Settings().model_copy(
        update={
            "base_model_revision": "1" * 40,
            "base_model_config_revision": "2" * 40,
            "controlnet_model_revision": "3" * 40,
            "diffqrcoder_upstream_enabled": True,
            "lab_clip_scoring_enabled": True,
            "lab_hps_scoring_enabled": True,
            "lab_quality_scoring_fail_closed": True,
        }
    )
    return _runtime_scientific_contract(settings)


def _plan(profile: str) -> dict:
    return scientific_plan(
        profile=profile,
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )


def test_runtime_provenance_checks_environment_and_embedded_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attestation = tmp_path / "prooftag-build-commit.txt"
    attestation.write_text(SOURCE_COMMIT, encoding="ascii")
    monkeypatch.setattr(large_campaign, "BUILD_COMMIT_ATTESTATION", attestation)
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE", RUNTIME_IMAGE)
    monkeypatch.setenv("PROOFTAG_RUNTIME_IMAGE_DIGEST", RUNTIME_DIGEST)

    assert _assert_runtime_provenance(_plan("smoke")) == (
        RUNTIME_IMAGE,
        RUNTIME_DIGEST,
    )

    attestation.write_text("c" * 40, encoding="ascii")
    with pytest.raises(RuntimeError, match="embedded build commit"):
        _assert_runtime_provenance(_plan("smoke"))


@pytest.mark.parametrize(
    ("profile", "prompts", "configs", "parents"),
    (("smoke", 4, 2, 8), ("pilot", 32, 6, 192), ("full", 256, 12, 3072)),
)
def test_scientific_plan_profile_counts(
    profile: str, prompts: int, configs: int, parents: int
) -> None:
    plan = _plan(profile)
    assert plan["expected_prompt_count"] == prompts
    assert plan["profile_spec"]["config_count"] == configs
    assert plan["expected_parent_count"] == parents
    assert len(plan["candidates"]) == parents
    assert len({row["seed"] for row in plan["candidates"]}) == parents


def test_plan_is_deterministic_and_freezes_catalog_and_doe() -> None:
    left = _plan("smoke")
    right = _plan("smoke")
    assert left == right
    assert left["scientific_plan_hash"] == right["scientific_plan_hash"]
    assert left["catalog_file_sha256"] == catalog_file_sha256()
    assert left["doe_file_sha256"] == doe_file_sha256()
    contract = left["runtime_scientific_contract"]
    assert contract["contract_sha256"]
    assert contract["settings"]["base_model_revision"]
    assert contract["qr_verify"]["implementation_sha256"]
    assert left["parameter_contract"]["random_python_unseeded_forbidden"] is True
    assert left["parameter_contract"]["row_random_split_forbidden"] is True
    assert min(row["doe_isolation_score"] for row in left["candidates"]) > 0
    summary = plan_summary(left)
    assert summary["scientific_plan_sha256"] == left["scientific_plan_hash"]
    assert summary["estimated_gpu_generation_hours"] > 0
    assert summary["estimated_cpu_scoring_hours"] > 0
    assert (
        summary["estimated_gpu_generation_hours"]
        + summary["estimated_cpu_scoring_hours"]
        == pytest.approx(summary["estimated_hours"], abs=0.2)
    )


def test_plan_loading_rejects_semantic_tampering(tmp_path: Path) -> None:
    plan = create_plan(
        output_root=tmp_path,
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )
    plan_path = tmp_path / plan["plan_id"] / "plan.json"
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["candidates"][0]["prompt"] = "tampered"
    atomic_write_json(plan_path, tampered)
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_plan(tmp_path, plan["plan_id"])


def test_plan_rejects_a_non_hex_runtime_digest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        scientific_plan(
            profile="smoke",
            source_commit=SOURCE_COMMIT,
            runtime_image=RUNTIME_IMAGE,
            runtime_image_digest="sha256:" + "z" * 64,
            runtime_scientific_contract=_frozen_runtime_contract(),
        )


def test_plan_rejects_unpinned_models_and_disabled_scorers() -> None:
    with pytest.raises(RuntimeError, match="pinned 40-hex revision"):
        scientific_plan(
            profile="smoke",
            source_commit=SOURCE_COMMIT,
            runtime_image=RUNTIME_IMAGE,
            runtime_image_digest=RUNTIME_DIGEST,
            runtime_scientific_contract=_runtime_scientific_contract(Settings()),
        )


def test_default_plan_uses_pinned_generation_models_and_required_scorers() -> None:
    plan = scientific_plan(
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
    )
    settings = plan["runtime_scientific_contract"]["settings"]
    assert all(settings[field] for field in (
        "base_model_revision",
        "base_model_config_revision",
        "controlnet_model_revision",
        "diffqrcoder_revision",
    ))
    assert settings["lab_clip_scoring_enabled"] is True
    assert settings["lab_hps_scoring_enabled"] is True
    assert settings["lab_quality_scoring_fail_closed"] is True


def test_effective_stage2_parameter_semantics_are_explicit() -> None:
    common = {
        "id": "test",
        "error_correction": "M",
        "qr_mask_pattern": 0,
        "stage1_steps": 40,
        "stage1_guidance_scale": 7.5,
        "stage1_controlnet_scale": 1.2,
        "control_guidance_start": 0.02,
        "control_guidance_end": 0.96,
        "stage2_strength": 0.8,
        "stage2_steps": 40,
        "stage2_controlnet_scale": 1.05,
        "stage2_qr_weight": 50.0,
        "stage2_perceptual_weight": 20.0,
        "rationale": "test",
    }
    public = _recipe_features(
        ParentRecipe(stage2_initialization="public_random", **common)
    )
    paper = _recipe_features(
        ParentRecipe(stage2_initialization="paper_stage1_noise", **common)
    )
    assert public["stage2_strength_applicable"] is False
    assert public["stage2_strength_effective"] is None
    assert paper["stage2_strength_applicable"] is True
    assert paper["stage2_strength_effective"] == 0.8
    assert paper["stage2_guidance_scale_effective"] == 7.5
    assert paper["guidance_scale_scope"] == "shared_stage1_stage2_upstream_request"


def test_parameter_coverage_reports_all_thirteen_effective_factors() -> None:
    plan = _plan("smoke")
    recipes = {item["id"]: ParentRecipe(**item) for item in plan["parent_recipes"]}
    rows = []
    for candidate in plan["candidates"]:
        recipe = recipes[candidate["parent_recipe_id"]]
        rows.append(
            {
                **candidate,
                **_recipe_features(recipe),
                "error_correction": recipe.error_correction,
                "qr_mask_pattern": recipe.qr_mask_pattern,
            }
        )
    coverage = _parameter_coverage(rows)
    assert coverage["covered_factor_count"] == 13
    assert coverage["all_factors_present"] is True
    assert set(coverage["factors"]) == {
        "stage1_steps",
        "stage1_guidance_scale",
        "stage1_controlnet_scale",
        "control_guidance_start",
        "control_guidance_end",
        "stage2_strength",
        "stage2_steps",
        "stage2_controlnet_scale",
        "stage2_qr_weight",
        "stage2_perceptual_weight",
        "stage2_initialization",
        "error_correction",
        "mask_slot",
    }


def test_e046_iteration_zero_is_exact_parent_without_changing_e040(tmp_path: Path) -> None:
    root = tmp_path / "trajectory"
    recipe_id = "recipe"
    trajectory = root / recipe_id
    (trajectory / "images").mkdir(parents=True)
    decoded = Image.new("RGB", (8, 8), "red")
    parent = Image.new("RGB", (8, 8), "blue")
    decoded.save(trajectory / "images/iteration-000.png")
    atomic_write_json(
        trajectory / "trace.json",
        [{"iteration": 0, "image_sha256": "old", "cuda": {}}],
    )
    (trajectory / "trace.csv").write_text("old\n", encoding="utf-8")
    checkpoint = SimpleNamespace(iteration=0, trace_step={})
    audit = _bind_e046_iteration_zero_to_parent(
        trajectory_root=root,
        recipe_id=recipe_id,
        parent_image=parent,
        checkpoints=[checkpoint],
    )
    persisted = Image.open(trajectory / "images/iteration-000.png").convert("RGB")
    diagnostic = Image.open(
        trajectory / "diagnostics/iteration-000-vae-decoded.png"
    ).convert("RGB")
    assert list(persisted.getdata()) == list(parent.getdata())
    assert list(diagnostic.getdata()) == list(decoded.getdata())
    assert audit["exact_parent_raster"] is True
    assert checkpoint.trace_step["iteration_zero_exact_parent_raster"] is True


def test_same_size_artifact_mutation_invalidates_e046_promotion(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "payload.bin").write_bytes(b"AAAA")
    final = tmp_path / "final"
    _promote_e046_attempt(
        attempt_dir=attempt,
        final_dir=final,
        required_files=("payload.bin",),
        metadata={"experiment": EXPERIMENT},
    )
    assert _promotion_valid(final, ("payload.bin",))
    (final / "payload.bin").write_bytes(b"BBBB")
    assert not _promotion_valid(final, ("payload.bin",))


def test_scientific_consumption_rejects_mutation_with_restored_mtime(
    tmp_path: Path,
) -> None:
    plan = create_plan(
        output_root=tmp_path,
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )
    candidate_id = str(plan["candidates"][0]["id"])
    attempt = tmp_path / "parent-attempt"
    for name in PARENT_GENERATION_REQUIRED:
        path = attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"AAAA")
    parent_dir = tmp_path / plan["plan_id"] / "parents" / candidate_id
    _promote_e046_attempt(
        attempt_dir=attempt,
        final_dir=parent_dir,
        required_files=PARENT_GENERATION_REQUIRED,
        metadata={"experiment": EXPERIMENT},
    )

    latent = parent_dir / "stage2-latent.safetensors"
    original = latent.stat()
    latent.write_bytes(b"BBBB")
    os.utime(
        latent,
        ns=(original.st_atime_ns, original.st_mtime_ns),
    )

    # The fast status/list path may trust unchanged metadata, but every
    # scientific consumer must recompute the content digest.
    assert _promotion_valid(parent_dir, PARENT_GENERATION_REQUIRED)
    assert not _promotion_valid(
        parent_dir, PARENT_GENERATION_REQUIRED, deep=True
    )
    with pytest.raises(FileNotFoundError, match="parent generation incomplete"):
        large_campaign.score_parent(
            output_root=tmp_path,
            plan_id=plan["plan_id"],
            candidate_id=candidate_id,
        )


def test_invalid_qr_cannot_win_and_bad_examples_are_retained() -> None:
    invalid_beautiful = {
        "prompt_id": "p1",
        "candidate_id": "invalid",
        "iteration": 0,
        "delivery_eligible": False,
        "wechat_exact_presets": 0,
        "multiobjective_prompt_score": 1.0,
        "clip_score": 1.0,
    }
    valid_plain = {
        "prompt_id": "p1",
        "candidate_id": "valid",
        "iteration": 0,
        "delivery_eligible": True,
        "wechat_exact_presets": 34,
        "multiobjective_prompt_score": 0.0,
        "clip_score": 0.0,
    }
    assert _best_by_prompt([invalid_beautiful, valid_plain])[0]["candidate_id"] == "valid"
    summary = _dataset_summary(
        plan={"plan_id": "plan", "profile": "smoke"},
        parents=[
            {
                **invalid_beautiful,
                "prompt_family": "family",
                "observation_independence_class": "independent_parent",
            },
            {
                **valid_plain,
                "prompt_family": "family",
                "observation_independence_class": "independent_parent",
            },
        ],
        srmpgd=[],
    )
    assert summary["parent_observation_count"] == 2
    assert summary["hard_negative_count"] == 1
    assert summary["training_ready"] is False


def test_plan_records_stage_and_trajectory_contracts() -> None:
    plan = _plan("smoke")
    policy = plan["observation_policy"]
    assert policy["primary_training_observation"] == "stage2_raw"
    assert policy["stage1_final_eligible"] is False
    assert policy["scene_qz_final_eligible"] is False
    assert policy["srmpgd_checkpoints_independent"] is False
    assert plan["phase_b_policy"]["terminal_srl_fidelity_failure"] == {
        "classification": "scientific_fidelity_mismatch",
        "retryable": False,
        "usable": False,
        "abort_unrelated_tasks": False,
    }
    assert {"trajectory_id", "generation_group_id"}.issubset(
        plan["parameter_contract"]["split_groups"]
    )
    assert _terminal_srl_failure(
        RuntimeError("local upstream SRL port diverged from the pinned official class")
    )


def test_resume_lists_only_incomplete_parents(tmp_path: Path) -> None:
    plan = _plan("smoke")
    plan_dir = tmp_path / plan["plan_id"]
    plan_dir.mkdir(parents=True)
    atomic_write_json(plan_dir / "plan.json", plan)
    (plan_dir / "prompt-catalog.json").write_bytes(CATALOG_PATH.read_bytes())
    (plan_dir / "doe.json").write_bytes(DOE_PATH.read_bytes())
    first = str(plan["candidates"][0]["id"])
    attempt = plan_dir / "attempt"
    (attempt / "images").mkdir(parents=True)
    (attempt / "images/stage1-raw.png").write_bytes(b"stage1")
    (attempt / "images/stage2-raw.png").write_bytes(b"stage2")
    (attempt / "stage2-latent.safetensors").write_bytes(b"latent")
    atomic_write_json(attempt / "parent-metadata.json", {"ok": True})
    atomic_write_json(attempt / "GENERATION_COMPLETE.json", {"ok": True})
    promote_attempt(
        attempt_dir=attempt,
        final_dir=plan_dir / "parents" / first,
        required_files=PARENT_GENERATION_REQUIRED,
        metadata={"candidate_id": first},
    )
    pending = list_parent_ids(
        output_root=tmp_path,
        plan_id=plan["plan_id"],
        pending_only=True,
    )
    assert first not in pending
    assert len(pending) == 7


def test_successful_retry_resolves_but_preserves_failure_history(
    tmp_path: Path,
) -> None:
    plan = _plan("smoke")
    plan_dir = tmp_path / plan["plan_id"]
    plan_dir.mkdir(parents=True)
    candidate_id = str(plan["candidates"][0]["id"])
    _record_failure(
        plan_dir=plan_dir,
        task_kind="parent",
        task_id=candidate_id,
        error=RuntimeError("transient interrupted attempt"),
    )
    audit = _technical_failure_audit(plan_dir, plan)
    assert len(audit["history"]) == 1
    assert len(audit["unresolved"]) == 1

    attempt = plan_dir / "retry-attempt"
    (attempt / "images").mkdir(parents=True)
    (attempt / "images/stage1-raw.png").write_bytes(b"stage1")
    (attempt / "images/stage2-raw.png").write_bytes(b"stage2")
    (attempt / "stage2-latent.safetensors").write_bytes(b"latent")
    atomic_write_json(attempt / "parent-metadata.json", {"ok": True})
    atomic_write_json(attempt / "GENERATION_COMPLETE.json", {"ok": True})
    promote_attempt(
        attempt_dir=attempt,
        final_dir=plan_dir / "parents" / candidate_id,
        required_files=PARENT_GENERATION_REQUIRED,
        metadata={"candidate_id": candidate_id},
    )
    audit = _technical_failure_audit(plan_dir, plan)
    assert len(audit["history"]) == 1
    assert audit["history"][0]["resolved_by_valid_promotion"] is True
    assert audit["unresolved"] == []


def test_unpromoted_terminal_srl_failure_remains_blocking(tmp_path: Path) -> None:
    plan = _plan("smoke")
    plan_dir = tmp_path / plan["plan_id"]
    plan_dir.mkdir(parents=True)
    candidate_id = str(plan["candidates"][0]["id"])
    recipe_id = str(plan["srmpgd_recipes"][0]["id"])
    _record_failure(
        plan_dir=plan_dir,
        task_kind="refinement",
        task_id=f"{candidate_id}__{recipe_id}",
        error=RuntimeError(
            "local upstream SRL port diverged from the pinned official class"
        ),
        classification="scientific_fidelity_mismatch",
        retryable=False,
    )
    audit = _technical_failure_audit(plan_dir, plan)
    assert len(audit["history"]) == 1
    assert len(audit["unresolved"]) == 1


def test_terminal_srl_retry_requires_explicit_override(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = create_plan(
        output_root=tmp_path,
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )
    plan_dir = tmp_path / plan["plan_id"]
    candidate_id = str(plan["candidates"][0]["id"])
    recipe_id = str(plan["srmpgd_recipes"][0]["id"])
    atomic_write_json(
        plan_dir / "selected-refinements.json",
        {"tasks": [{"candidate_id": candidate_id, "srmpgd_recipe_id": recipe_id}]},
    )
    _record_failure(
        plan_dir=plan_dir,
        task_kind="refinement",
        task_id=f"{candidate_id}__{recipe_id}",
        error=RuntimeError(
            "local upstream SRL port diverged from the pinned official class"
        ),
        classification="scientific_fidelity_mismatch",
        retryable=False,
    )
    monkeypatch.delenv("PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL", raising=False)
    assert list_refinement_ids(
        output_root=tmp_path,
        plan_id=plan["plan_id"],
        pending_only=True,
    ) == []
    monkeypatch.setenv("PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL", "1")
    assert list_refinement_ids(
        output_root=tmp_path,
        plan_id=plan["plan_id"],
        pending_only=True,
    ) == [(candidate_id, recipe_id)]


def test_promoted_terminal_srl_branch_does_not_block_other_refinements(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = create_plan(
        output_root=tmp_path,
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )
    plan_dir = tmp_path / plan["plan_id"]
    first_candidate = str(plan["candidates"][0]["id"])
    second_candidate = str(plan["candidates"][1]["id"])
    recipe_id = str(plan["srmpgd_recipes"][0]["id"])
    tasks = [
        {"candidate_id": first_candidate, "srmpgd_recipe_id": recipe_id},
        {"candidate_id": second_candidate, "srmpgd_recipe_id": recipe_id},
    ]
    atomic_write_json(plan_dir / "selected-refinements.json", {"tasks": tasks})
    error = RuntimeError(
        "local upstream SRL port diverged from the pinned official class"
    )
    failure = _record_failure(
        plan_dir=plan_dir,
        task_kind="refinement",
        task_id=f"{first_candidate}__{recipe_id}",
        error=error,
        classification="scientific_fidelity_mismatch",
        retryable=False,
    )
    attempt = plan_dir / "terminal-attempt"
    attempt.mkdir()
    atomic_write_json(attempt / "task.json", tasks[0])
    atomic_write_json(
        attempt / "TERMINAL_FAILURE.json", {**failure, "terminal": True}
    )
    _promote_e046_attempt(
        attempt_dir=attempt,
        final_dir=plan_dir / "refinements" / first_candidate / recipe_id,
        required_files=("task.json", "TERMINAL_FAILURE.json"),
        metadata={"candidate_id": first_candidate, "recipe_id": recipe_id},
    )
    monkeypatch.delenv("PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL", raising=False)
    assert list_refinement_ids(
        output_root=tmp_path,
        plan_id=plan["plan_id"],
        pending_only=True,
    ) == [(second_candidate, recipe_id)]
    audit = _technical_failure_audit(plan_dir, plan)
    assert audit["history"][0]["resolved_by_valid_promotion"] is True
    assert audit["unresolved"] == []


def test_verification_rejects_an_empty_refinement_set_instead_of_vacuous_success(
    tmp_path: Path,
) -> None:
    semantic = {
        "schema": PLAN_SCHEMA,
        "experiment": EXPERIMENT,
        "profile": "smoke",
        "source_commit": SOURCE_COMMIT,
        "runtime_image_digest": RUNTIME_DIGEST,
        "runtime_scientific_contract": _frozen_runtime_contract(),
        "expected_parent_count": 0,
        "candidates": [],
        "srmpgd_recipes": [],
        "catalog_file_sha256": catalog_file_sha256(),
        "doe_file_sha256": doe_file_sha256(),
    }
    from prooftag_qr.resilient_experiment import stable_hash

    plan_hash = stable_hash(semantic)
    plan_id = plan_hash[:16]
    plan_dir = tmp_path / plan_id
    plan_dir.mkdir(parents=True)
    plan = {
        **semantic,
        "plan_id": plan_id,
        "scientific_plan_hash": plan_hash,
    }
    atomic_write_json(plan_dir / "plan.json", plan)
    (plan_dir / "prompt-catalog.json").write_bytes(CATALOG_PATH.read_bytes())
    (plan_dir / "doe.json").write_bytes(DOE_PATH.read_bytes())
    atomic_write_json(plan_dir / "selected-refinements.json", {"tasks": []})
    atomic_write_json(plan_dir / "REFINEMENT_SCORING_COMPLETE.json", {"ok": True})
    (plan_dir / "failures").mkdir()
    manifest = [
        {
            "path": "plan.json",
            "sha256": sha256_file(plan_dir / "plan.json"),
            "size": (plan_dir / "plan.json").stat().st_size,
        }
    ]
    atomic_write_json(plan_dir / "artifact-manifest.json", manifest)
    atomic_write_json(
        plan_dir / "COMPLETE.json",
        {"artifact_manifest_sha256": sha256_file(plan_dir / "artifact-manifest.json")},
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        verify(output_root=tmp_path, plan_id=plan_id)


def test_verification_accepts_a_complete_manifest_then_detects_corruption(
    tmp_path: Path, monkeypatch,
) -> None:
    plan = create_plan(
        output_root=tmp_path,
        profile="smoke",
        source_commit=SOURCE_COMMIT,
        runtime_image=RUNTIME_IMAGE,
        runtime_image_digest=RUNTIME_DIGEST,
        runtime_scientific_contract=_frozen_runtime_contract(),
    )
    plan_dir = tmp_path / plan["plan_id"]
    candidate_id = str(plan["candidates"][0]["id"])
    recipe_id = str(plan["srmpgd_recipes"][0]["id"])
    contract_sha = plan["runtime_scientific_contract"]["contract_sha256"]
    parent = {
        "source_commit": SOURCE_COMMIT,
        "runtime_image_digest": RUNTIME_DIGEST,
        "scientific_contract_sha256": contract_sha,
        "source_kind": "parent",
        "clip_score": 0.5,
        "clip_aesthetic": 5.0,
        "hpsv2_1": 0.2,
    }
    parent_rows = [
        {
            **parent,
            "candidate_id": str(candidate["id"]),
            "parent_id": str(candidate["id"]),
        }
        for candidate in plan["candidates"]
    ]
    maximum_iterations = int(plan["srmpgd_recipes"][0]["max_iterations"])
    complete_refinement_rows = [
        {
            **parent,
            "source_kind": "srmpgd",
            "candidate_id": candidate_id,
            "parent_id": candidate_id,
            "srmpgd_recipe_id": recipe_id,
            "iteration": iteration,
            "iteration_zero_matches_parent_raster": (
                True if iteration == 0 else None
            ),
        }
        for iteration in range(maximum_iterations + 1)
    ]
    monkeypatch.setattr(
        large_campaign,
        "_parent_rows",
        lambda _plan_dir, _plan: parent_rows,
    )
    monkeypatch.setattr(
        large_campaign,
        "_refinement_rows",
        lambda _plan_dir, _plan: complete_refinement_rows,
    )
    atomic_write_json(
        plan_dir / "selected-refinements.json",
        {"tasks": [{"candidate_id": candidate_id, "srmpgd_recipe_id": recipe_id}]},
    )
    atomic_write_json(plan_dir / "REFINEMENT_SCORING_COMPLETE.json", {"ok": True})
    artifact = plan_dir / "dataset.bin"
    artifact.write_bytes(b"sealed")
    manifest = [
        {
            "path": artifact.name,
            "sha256": sha256_file(artifact),
            "size": artifact.stat().st_size,
        }
    ]
    atomic_write_json(plan_dir / "artifact-manifest.json", manifest)
    atomic_write_json(
        plan_dir / "COMPLETE.json",
        {"artifact_manifest_sha256": sha256_file(plan_dir / "artifact-manifest.json")},
    )
    result = verify(output_root=tmp_path, plan_id=plan["plan_id"])
    assert result["valid"] is True
    assert result["parent_observation_coverage_exact"] is True
    assert result["srmpgd_checkpoint_sequences_valid"] is True

    missing_nonzero = [
        row for row in complete_refinement_rows if row["iteration"] != 2
    ]
    monkeypatch.setattr(
        large_campaign,
        "_refinement_rows",
        lambda _plan_dir, _plan: missing_nonzero,
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        verify(output_root=tmp_path, plan_id=plan["plan_id"])

    duplicated_nonzero = [
        *complete_refinement_rows,
        dict(complete_refinement_rows[2]),
    ]
    monkeypatch.setattr(
        large_campaign,
        "_refinement_rows",
        lambda _plan_dir, _plan: duplicated_nonzero,
    )
    with pytest.raises(RuntimeError, match="verification failed"):
        verify(output_root=tmp_path, plan_id=plan["plan_id"])

    monkeypatch.setattr(
        large_campaign,
        "_refinement_rows",
        lambda _plan_dir, _plan: complete_refinement_rows,
    )

    artifact.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="verification failed"):
        verify(output_root=tmp_path, plan_id=plan["plan_id"])


def test_new_experiment_never_targets_historical_output_root() -> None:
    assert str(DEFAULT_OUTPUT_ROOT).replace("\\", "/") == (
        "/data/e046-large-advisor-dataset-v1"
    )
    assert "controlled-best-generator" not in str(DEFAULT_OUTPUT_ROOT)
    assert DATASET_SCHEMA.startswith("e046-large-")


def test_create_plan_refuses_historical_root_and_children(
    tmp_path: Path, monkeypatch,
) -> None:
    historical = tmp_path / "e046-controlled-best-generator-v1"
    monkeypatch.setattr(
        large_campaign, "HISTORICAL_E046_OUTPUT_ROOT", historical
    )
    for output_root in (historical, historical / "accidental-child"):
        with pytest.raises(RuntimeError, match="historical E046 output root"):
            create_plan(
                output_root=output_root,
                profile="smoke",
                source_commit=SOURCE_COMMIT,
                runtime_image=RUNTIME_IMAGE,
                runtime_image_digest=RUNTIME_DIGEST,
                runtime_scientific_contract=_frozen_runtime_contract(),
            )
        assert not output_root.exists()


def test_create_plan_refuses_historical_e045_root_and_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical = tmp_path / "e045-foundation-v1"
    monkeypatch.setattr(
        large_campaign, "HISTORICAL_E045_OUTPUT_ROOT", historical
    )
    for output_root in (historical, historical / "accidental-child"):
        with pytest.raises(RuntimeError, match="historical E045 output root"):
            create_plan(
                output_root=output_root,
                profile="smoke",
                source_commit="1" * 40,
                runtime_image="example.invalid/e046@sha256:" + "a" * 64,
                runtime_image_digest="sha256:" + "a" * 64,
            )
        assert not output_root.exists()
