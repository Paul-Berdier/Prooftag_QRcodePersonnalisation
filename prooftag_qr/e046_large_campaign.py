"""E046 large — reproducible advisor-ready dataset campaign.

This experiment is deliberately separate from ``e046-controlled-best-generator-v1``.
It reuses the proven DiffQRCoder/QR-Verify/SR-MPGD primitives while defining a new,
immutable scientific plan:

* Phase A generates one independent Stage1/Stage2 parent per prompt/config pair;
* only the raw Stage2 raster receives the expensive automatic scoring stack;
* Phase B selects informative parents after Phase A has completed;
* SR-MPGD checkpoints are exported as correlated observations, never independent rows;
* all bad examples remain in the advisor dataset.

The CLI is intentionally task-oriented so a Kubernetes runner can stop between jobs,
resume without recomputing completed work, and keep the single RTX sequential.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .e046_campaign import (
    _annotate_prompt_objectives,
    _blueprint,
    _finite_number,
    _image_sha256,
    _load_parent_artifact,
    _module_fields,
    _parent_visual_guard,
    _refinement_visual_guard,
    _safe_settings_provenance,
    _seed_everything,
    _settings_for,
    _stage_core_change,
    _validate_commit,
)
from .e046_catalog import (
    CANVAS_PX,
    NEGATIVE_PROMPT,
    QR_MODULE_SIZE,
    QR_PADDING_PX,
    QR_SOFTWARE_ENGINE,
    QR_VERIFY_PRESET_COUNT,
    QR_VERSION,
    ParentRecipe,
    SRMPGDRecipe,
)
from .e046_large_catalog import (
    CATALOG_PATH,
    DOE_PATH,
    EXPERIMENT,
    PromptSpec,
    catalog_file_sha256,
    doe_file_sha256,
    load_catalog,
    load_doe,
    load_prompts,
    prompts_for_profile,
    tag_coverage,
)
from .e046_large_doe import (
    DIMENSIONS,
    actual_mask,
    config_to_parent_recipe_kwargs,
    configs_for_profile,
    parent_seed,
)
from .resilient_experiment import (
    atomic_write_json,
    atomic_write_text,
    build_artifact_manifest,
    promote_attempt,
    sha256_file,
    stable_hash,
    utc_now,
)

DEFAULT_OUTPUT_ROOT = Path("/data/e046-large-advisor-dataset-v1")
HISTORICAL_E045_OUTPUT_ROOT = Path("/data/e045-foundation-v1")
HISTORICAL_E046_OUTPUT_ROOT = Path("/data/e046-controlled-best-generator-v1")
PLAN_SCHEMA = "e046-large-scientific-plan-v1"
DATASET_SCHEMA = "e046-large-advisor-observation-v1"
LATEST_SCHEMA = "e046-large-latest-v1"
TERMINAL_SRL_FRAGMENT = (
    "local upstream SRL port diverged from the pinned official class"
)
BUILD_COMMIT_ATTESTATION = Path("/app/prooftag-build-commit.txt")

# The large campaign measures 13 independently designed generation/QR factors.
DOE_PARAMETER_COUNT = 13
PROMPT_STRUCTURAL_FEATURE_COUNT = 9

SCIENTIFIC_SETTINGS_FIELDS: tuple[str, ...] = (
    "base_model_id",
    "base_model_revision",
    "base_model_config_id",
    "base_model_config_revision",
    "controlnet_model_id",
    "controlnet_model_subfolder",
    "controlnet_model_revision",
    "controlnet_conditioning_profile",
    "controlnet_pipeline_mode",
    "diffqrcoder_upstream_enabled",
    "diffqrcoder_revision",
    "quality_clip_model_id",
    "quality_clip_model_revision",
    "quality_aesthetic_weights_url",
    "quality_aesthetic_weights_sha256",
    "quality_hps_model_version",
    "quality_hps_package_name",
    "quality_hps_package_version",
    "quality_hps_source_revision",
    "quality_hps_checkpoint_repo",
    "quality_hps_checkpoint_revision",
    "quality_hps_checkpoint_filename",
    "quality_hps_checkpoint_sha256",
    "lab_clip_scoring_enabled",
    "lab_hps_scoring_enabled",
    "lab_quality_scoring_fail_closed",
)
PINNED_REVISION_FIELDS: tuple[str, ...] = (
    "base_model_revision",
    "base_model_config_revision",
    "controlnet_model_revision",
    "diffqrcoder_revision",
    "quality_clip_model_revision",
    "quality_hps_source_revision",
    "quality_hps_checkpoint_revision",
)

SRMPGD_RECIPES: tuple[SRMPGDRecipe, ...] = (
    SRMPGDRecipe(
        "g250_r100_i04",
        250.0,
        0.100,
        4,
        0.01,
        0.040,
        0.040,
        0.10,
        10,
        "Conservative low-gamma trust region.",
    ),
    SRMPGDRecipe(
        "g500_r200_i08",
        500.0,
        0.200,
        8,
        0.01,
        0.050,
        0.050,
        0.10,
        12,
        "E044 software frontier with projection and backtracking.",
    ),
    SRMPGDRecipe(
        "g1000_r150_i08",
        1000.0,
        0.150,
        8,
        0.01,
        0.050,
        0.050,
        0.10,
        12,
        "High raw gamma constrained by the trust region.",
    ),
    SRMPGDRecipe(
        "g500_r150_i08",
        500.0,
        0.150,
        8,
        0.02,
        0.040,
        0.040,
        0.12,
        12,
        "Tighter visual-preservation branch with stronger LPIPS.",
    ),
)

# Prefixes are deliberately small. Full can never be reached implicitly from smoke.
PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "smoke": {
        "prompt_count": 4,
        "config_count": 2,
        "secondary_parent_count": 0,
        "srmpgd_recipes_per_parent": 1,
    },
    "pilot": {
        "prompt_count": 32,
        "config_count": 6,
        "secondary_parent_count": 8,
        "srmpgd_recipes_per_parent": 2,
    },
    "full": {
        "prompt_count": 256,
        "config_count": 12,
        "secondary_parent_count": 64,
        "srmpgd_recipes_per_parent": 2,
    },
}

# Initial planning assumptions only. E013 measured 160-216 s per Stage1+Stage2
# pair at 768 px on the same class of GPU; no per-trajectory SR-MPGD or per-image
# scoring timing exists in the repository, so those values are deliberately
# conservative. ``status`` replaces them with the observed average as soon as
# tasks complete.
ESTIMATE_SECONDS = {
    "parent_generation": 180.0,
    "parent_scoring": 30.0,
    "srmpgd_generation": 360.0,
    "srmpgd_checkpoint_scoring": 25.0,
}
ESTIMATE_BYTES = {
    "parent": 4_500_000,
    "srmpgd_checkpoint": 1_850_000,
    "metadata_overhead_fraction": 0.20,
}

PARENT_GENERATION_REQUIRED = (
    "images/stage1-raw.png",
    "images/stage2-raw.png",
    "stage2-latent.safetensors",
    "parent-metadata.json",
    "GENERATION_COMPLETE.json",
)
SCORING_REQUIRED = (
    "comparison.json",
    "comparison.csv",
    "qr-verify-evidence.json",
    "quality-scores.json",
    "quality-provenance.json",
    "SCORING_COMPLETE.json",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_isolated_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    historical_roots = (
        ("E045", HISTORICAL_E045_OUTPUT_ROOT.resolve()),
        ("E046", HISTORICAL_E046_OUTPUT_ROOT.resolve()),
    )
    for experiment, historical in historical_roots:
        if resolved == historical or historical in resolved.parents:
            raise RuntimeError(
                f"E046 large refuses the historical {experiment} output root "
                f"and every child: {resolved}"
            )
    return resolved


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) or ["empty"]
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _flatten_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for key, value in list(row.items()):
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        output.append(row)
    return output


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=False, compress_level=9)


def _promote_e046_attempt(
    *,
    attempt_dir: Path,
    final_dir: Path,
    required_files: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote, then seal E046 files with a cheap resume integrity fingerprint.

    The generic promotion already records SHA-256 for every byte.  E046 adds
    ``mtime_ns`` after the atomic move: unchanged files can be accepted quickly
    during thousands of resume checks, while any ordinary same-size mutation
    forces a full SHA-256 comparison in :func:`_promotion_valid`.
    """
    promote_attempt(
        attempt_dir=attempt_dir,
        final_dir=final_dir,
        required_files=required_files,
        metadata=metadata,
    )
    manifest_path = final_dir / "PROMOTION_MANIFEST.json"
    stored = _load_json(manifest_path)
    files: list[dict[str, Any]] = []
    for source in stored["files"]:
        item = dict(source)
        path = final_dir / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"promoted E046 artifact disappeared: {path}")
        item["mtime_ns"] = path.stat().st_mtime_ns
        files.append(item)
    material = {"metadata": dict(stored["metadata"]), "files": files}
    sealed = {**material, "manifest_hash": stable_hash(material)}
    atomic_write_json(manifest_path, sealed)
    return sealed


def _promotion_valid(
    final_dir: Path,
    required_files: Sequence[str],
    *,
    deep: bool = False,
) -> bool:
    """Validate a promotion fail-closed without quadratic full-run hashing.

    E046 promotions carry the mtime observed immediately after their atomic
    move.  Size *and* mtime unchanged means the content hash computed at
    promotion is still authoritative.  A changed/missing mtime, legacy
    manifest, or explicit ``deep`` audit triggers a byte-level SHA-256 check.
    """
    manifest_path = final_dir / "PROMOTION_MANIFEST.json"
    if not manifest_path.is_file():
        return False
    try:
        stored = _load_json(manifest_path)
        material = {
            "metadata": dict(stored["metadata"]),
            "files": list(stored["files"]),
        }
        if stored.get("manifest_hash") != stable_hash(material):
            return False
        entries = {str(item["path"]): item for item in stored["files"]}
        for name in required_files:
            item = entries.get(name)
            path = final_dir / name
            if item is None or not path.is_file():
                return False
            stat = path.stat()
            if int(item["size_bytes"]) != stat.st_size:
                return False
            recorded_mtime = item.get("mtime_ns")
            must_hash = (
                deep
                or recorded_mtime is None
                or int(recorded_mtime) != stat.st_mtime_ns
            )
            if must_hash and str(item["sha256"]) != sha256_file(path):
                return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _recipe_map(plan: Mapping[str, Any]) -> dict[str, ParentRecipe]:
    return {
        str(item["id"]): ParentRecipe(**item)
        for item in plan["parent_recipes"]
    }


def _srmpgd_recipe_map(plan: Mapping[str, Any]) -> dict[str, SRMPGDRecipe]:
    return {
        str(item["id"]): SRMPGDRecipe(**item)
        for item in plan["srmpgd_recipes"]
    }


def _candidate(plan: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    item = next(
        (row for row in plan["candidates"] if row["id"] == candidate_id),
        None,
    )
    if item is None:
        raise KeyError(f"unknown E046 large candidate: {candidate_id}")
    return dict(item)


def _parent_dir(plan_dir: Path, candidate_id: str) -> Path:
    return plan_dir / "parents" / candidate_id


def _refinement_dir(plan_dir: Path, candidate_id: str, recipe_id: str) -> Path:
    return plan_dir / "refinements" / candidate_id / recipe_id


def _refinement_generation_required(recipe: SRMPGDRecipe) -> tuple[str, ...]:
    expected = recipe.max_iterations + 1
    checkpoint_files = [
        f"trajectory/{recipe.id}/{kind}/iteration-{iteration:03d}.{suffix}"
        for iteration in range(expected)
        for kind, suffix in (("images", "png"), ("latents", "safetensors"))
    ]
    return tuple(
        [
            f"trajectory/{recipe.id}/trace.json",
            f"trajectory/{recipe.id}/trace.csv",
            f"trajectory/{recipe.id}/iteration-zero-audit.json",
            f"trajectory/{recipe.id}/diagnostics/iteration-000-vae-decoded.png",
            "runtime.json",
            "refinement-metadata.json",
            "GENERATION_COMPLETE.json",
        ]
        + checkpoint_files
    )


def _profile_prompts(
    prompts: Sequence[PromptSpec], profile: str
) -> tuple[PromptSpec, ...]:
    """Compatibility wrapper around the canonical catalog selection."""
    return prompts_for_profile(prompts, profile)


def _candidate_seed(prompt_ordinal: int, config_index: int) -> int:
    """Compatibility wrapper around the seed policy frozen in the DOE."""
    return parent_seed(prompt_ordinal, config_index)


def _unit_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    names = sorted(set(left) & set(right))
    return math.sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _doe_isolation(config: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]) -> float:
    point = dict(config["unit_point"])
    config_id = str(config["config_id"])
    others = [
        dict(item["unit_point"])
        for item in configs
        if str(item["config_id"]) != config_id
    ]
    if len(others) != len(configs) - 1:
        raise RuntimeError(f"DOE config id is not unique: {config_id}")
    return min((_unit_distance(point, other) for other in others), default=0.0)


def _runtime_scientific_contract(settings: Any | None = None) -> dict[str, Any]:
    from .config import Settings
    from .validation import (
        CONSERVATIVE_QR_VERIFY_SCORING_VERSION,
        QRVerifyDecoder,
    )

    settings = settings or Settings()
    settings_contract = {
        field: getattr(settings, field) for field in SCIENTIFIC_SETTINGS_FIELDS
    }
    bridge = Path(__file__).resolve().parent.parent / "qr_verify_bridge/bridge.mjs"
    lock = bridge.parent / "package-lock.json"
    digest = hashlib.sha256()
    digest.update(QRVerifyDecoder.engine_version.encode("utf-8"))
    for path in (bridge, lock):
        if not path.is_file():
            raise FileNotFoundError(f"frozen QR-Verify component missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    semantic = {
        "schema": "e046-large-runtime-scientific-contract-v1",
        "settings": settings_contract,
        "qr_verify": {
            "engine_version": QRVerifyDecoder.engine_version,
            "preset_count": QRVerifyDecoder.preset_count,
            "repetitions": 3,
            "scoring_version": CONSERVATIVE_QR_VERIFY_SCORING_VERSION,
            "implementation_sha256": digest.hexdigest(),
        },
    }
    return {**semantic, "contract_sha256": stable_hash(semantic)}


def _validate_runtime_scientific_contract(contract: Mapping[str, Any]) -> None:
    semantic = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if str(contract.get("contract_sha256") or "") != stable_hash(semantic):
        raise RuntimeError("E046 runtime scientific contract hash mismatch")
    settings = dict(contract.get("settings") or {})
    for field in PINNED_REVISION_FIELDS:
        revision = str(settings.get(field) or "")
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise RuntimeError(f"E046 requires a pinned 40-hex revision for {field}")
    for field in (
        "diffqrcoder_upstream_enabled",
        "lab_clip_scoring_enabled",
        "lab_hps_scoring_enabled",
        "lab_quality_scoring_fail_closed",
    ):
        if settings.get(field) is not True:
            raise RuntimeError(f"E046 requires {field}=true")
    qr_verify = dict(contract.get("qr_verify") or {})
    if int(qr_verify.get("preset_count") or 0) != QR_VERIFY_PRESET_COUNT:
        raise RuntimeError("E046 QR-Verify preset count is not frozen correctly")


def build_candidates(
    *, profile: str, prompts: Sequence[PromptSpec], doe: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build independent parents and their distinct effective recipes."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown E046 large profile: {profile}")
    spec = PROFILE_SPECS[profile]
    selected_prompts = _profile_prompts(prompts, profile)
    configs = list(configs_for_profile(doe, profile))
    all_configs = list(doe["configs"])
    candidates: list[dict[str, Any]] = []
    recipes: dict[str, dict[str, Any]] = {}

    for prompt in selected_prompts:
        for config in configs:
            config_index = int(config["config_index"])
            mask = actual_mask(int(config["mask_slot"]), prompt.mask_rotation)
            recipe_id = f"{config['config_id']}_m{mask}"
            recipes.setdefault(
                recipe_id,
                config_to_parent_recipe_kwargs(
                    config,
                    qr_mask_pattern=mask,
                    recipe_id=recipe_id,
                ),
            )
            candidate_id = f"p{prompt.ordinal:04d}_k{config_index:02d}"
            candidate = {
                "id": candidate_id,
                "prompt_id": prompt.prompt_id,
                "prompt_ordinal": prompt.ordinal,
                "prompt_family": prompt.family,
                "prompt": prompt.prompt,
                "payload": prompt.payload,
                "payload_length": prompt.payload_length,
                "prompt_variant_index": 0,
                "replicate_index": 0,
                "config_id": config["config_id"],
                "config_index": config_index,
                "mask_slot": int(config["mask_slot"]),
                "mask_rotation": int(prompt.mask_rotation),
                "parent_recipe_id": recipe_id,
                "seed": _candidate_seed(prompt.ordinal, config_index),
                "generation_group_id": candidate_id,
                "prompt_group_id": prompt.prompt_id,
                "observation_independence_class": "independent_parent",
                "doe_isolation_score": _doe_isolation(config, all_configs),
                "diagnostic_sample": int(
                    hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16
                )
                % 20
                == 0,
                **{
                    key: getattr(prompt, key)
                    for key in (
                        "spatial_frequency",
                        "symmetry",
                        "brightness",
                        "contrast",
                        "colorfulness",
                        "negative_space",
                        "texture_density",
                        "geometry_type",
                        "composition_type",
                    )
                },
            }
            candidates.append(candidate)
    expected = int(spec["prompt_count"]) * int(spec["config_count"])
    if len(candidates) != expected:
        raise AssertionError(f"candidate count {len(candidates)} != {expected}")
    return candidates, list(recipes.values())


def _estimated_budget(profile: str) -> dict[str, Any]:
    spec = PROFILE_SPECS[profile]
    parent_count = int(spec["prompt_count"]) * int(spec["config_count"])
    selected_parent_count = int(spec["prompt_count"]) + int(
        spec["secondary_parent_count"]
    )
    refinement_count = selected_parent_count * int(spec["srmpgd_recipes_per_parent"])
    average_checkpoints = sum(
        recipe.max_iterations + 1 for recipe in SRMPGD_RECIPES
    ) / len(SRMPGD_RECIPES)
    checkpoint_count = math.ceil(refinement_count * average_checkpoints)
    generation_seconds = (
        parent_count * ESTIMATE_SECONDS["parent_generation"]
        + refinement_count * ESTIMATE_SECONDS["srmpgd_generation"]
    )
    scoring_seconds = (
        parent_count * ESTIMATE_SECONDS["parent_scoring"]
        + checkpoint_count * ESTIMATE_SECONDS["srmpgd_checkpoint_scoring"]
    )
    seconds = generation_seconds + scoring_seconds
    bytes_estimate = int(
        (
            parent_count * ESTIMATE_BYTES["parent"]
            + checkpoint_count * ESTIMATE_BYTES["srmpgd_checkpoint"]
        )
        * (1.0 + ESTIMATE_BYTES["metadata_overhead_fraction"])
    )
    return {
        "parent_count": parent_count,
        "selected_srmpgd_parent_maximum": selected_parent_count,
        "srmpgd_trajectory_maximum": refinement_count,
        "srmpgd_checkpoint_estimate": checkpoint_count,
        "wall_time_hours_sequential_estimate": round(seconds / 3600.0, 1),
        "gpu_generation_hours_estimate": round(generation_seconds / 3600.0, 1),
        "cpu_scoring_hours_estimate": round(scoring_seconds / 3600.0, 1),
        "disk_gib_estimate": round(bytes_estimate / 1024**3, 1),
        "assumptions_seconds": dict(ESTIMATE_SECONDS),
        "assumptions_bytes": dict(ESTIMATE_BYTES),
    }


def scientific_plan(
    *,
    profile: str,
    source_commit: str,
    runtime_image: str,
    runtime_image_digest: str,
    prompts: Sequence[PromptSpec] | None = None,
    catalog: Mapping[str, Any] | None = None,
    doe: Mapping[str, Any] | None = None,
    runtime_scientific_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_commit = _validate_commit(source_commit)
    if not runtime_image.strip():
        raise ValueError("runtime_image must be non-empty")
    if not (
        runtime_image_digest.startswith("sha256:")
        and len(runtime_image_digest) == 71
        and all(character in "0123456789abcdef" for character in runtime_image_digest[7:])
    ):
        raise ValueError("runtime_image_digest must be sha256:<64 hex>")
    if profile not in PROFILE_SPECS:
        raise ValueError(f"unknown E046 large profile: {profile}")
    catalog = dict(catalog or load_catalog())
    doe = dict(doe or load_doe())
    prompts = tuple(prompts or load_prompts())
    candidates, parent_recipes = build_candidates(
        profile=profile,
        prompts=prompts,
        doe=doe,
    )
    if runtime_scientific_contract is None:
        effective = _settings_for(candidates[0], ParentRecipe(**parent_recipes[0]))
        effective = type(effective).model_validate(
            {
                **effective.model_dump(),
                "lab_clip_scoring_enabled": True,
                "lab_hps_scoring_enabled": True,
                "lab_quality_scoring_fail_closed": True,
            }
        )
        runtime_scientific_contract = _runtime_scientific_contract(effective)
    scientific_contract = dict(runtime_scientific_contract)
    _validate_runtime_scientific_contract(scientific_contract)
    budget = _estimated_budget(profile)
    spec = dict(PROFILE_SPECS[profile])
    semantic = {
        "schema": PLAN_SCHEMA,
        "experiment": EXPERIMENT,
        "profile": profile,
        "source_commit": source_commit,
        "runtime_image": runtime_image,
        "runtime_image_digest": runtime_image_digest,
        "runtime_scientific_contract": scientific_contract,
        "catalog_schema": catalog["schema"],
        "catalog_sha256": catalog["catalog_sha256"],
        "catalog_file_sha256": catalog_file_sha256(),
        "doe_id": doe["doe_id"],
        "doe_sha256": doe["doe_sha256"],
        "doe_file_sha256": doe_file_sha256(),
        "profile_spec": spec,
        "candidates": candidates,
        "parent_recipes": parent_recipes,
        "srmpgd_recipes": [asdict(recipe) for recipe in SRMPGD_RECIPES],
        "expected_prompt_count": spec["prompt_count"],
        "expected_parent_count": budget["parent_count"],
        "expected_selected_parent_maximum": budget[
            "selected_srmpgd_parent_maximum"
        ],
        "expected_refinement_count_maximum": budget[
            "srmpgd_trajectory_maximum"
        ],
        "expected_checkpoint_count_estimate": budget[
            "srmpgd_checkpoint_estimate"
        ],
        "parameter_contract": {
            "doe_parameter_count": DOE_PARAMETER_COUNT,
            "prompt_structural_feature_count": PROMPT_STRUCTURAL_FEATURE_COUNT,
            "random_python_unseeded_forbidden": True,
            "seed_policy": str(doe["parent_seed_policy"]),
            "row_random_split_forbidden": True,
            "split_groups": [
                "prompt_id",
                "payload",
                "generation_group_id",
                "trajectory_id",
            ],
            "supported_evaluation": ["GroupKFold by prompt", "leave-prompt-family-out"],
        },
        "observation_policy": {
            "primary_training_observation": "stage2_raw",
            "stage1_saved_for_provenance": True,
            "stage1_final_eligible": False,
            "scene_qz_generated_by_default": False,
            "scene_qz_final_eligible": False,
            "bad_examples_retained": True,
            "srmpgd_checkpoints_independent": False,
        },
        "validity_policy": {
            "final_original_exact_required": True,
            "final_minimum_exact_presets": 34,
            "ideal_minimum_exact_presets": 37,
            "refinement_minimum_exact_presets": 16,
            "visual_guard_required": True,
            "raw_required": True,
        },
        "multiobjective_policy": {
            "weights": {
                "wechat_robustness": 0.40,
                "clip_prompt_alignment": 0.25,
                "hps_human_preference": 0.20,
                "clip_aesthetic": 0.15,
            },
            "individual_targets_preserved": True,
            "invalid_qr_can_win": False,
        },
        "phase_b_policy": {
            "one_primary_parent_per_prompt": True,
            "secondary_parent_count": spec["secondary_parent_count"],
            "recipes_per_selected_parent": spec["srmpgd_recipes_per_parent"],
            "strata": [
                "best_valid",
                "borderline",
                "high_aesthetic_low_qr",
                "high_qr_low_aesthetic",
                "unstable",
                "sparse_doe_region",
            ],
            "terminal_srl_fidelity_failure": {
                "classification": "scientific_fidelity_mismatch",
                "retryable": False,
                "usable": False,
                "abort_unrelated_tasks": False,
            },
        },
        "qr_scoring": {
            "engine": QR_SOFTWARE_ENGINE,
            "preset_count": QR_VERIFY_PRESET_COUNT,
            "repetitions": 3,
            "exact_payload_only": True,
            "content_addressed_cache": True,
        },
        "geometry": {
            "qr_version": QR_VERSION,
            "module_size": QR_MODULE_SIZE,
            "padding_px": QR_PADDING_PX,
            "canvas_px": CANVAS_PX,
        },
        "budget_estimate": budget,
        "automatic_full_launch_authorized": False,
        "manual_validation_required": False,
        "production_ready": False,
    }
    plan_hash = stable_hash(semantic)
    return {
        **semantic,
        "scientific_plan_hash": plan_hash,
        "plan_id": plan_hash[:16],
    }


def _plan_semantic_material(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in plan.items()
        if key not in {"scientific_plan_hash", "plan_id", "created_at_utc"}
    }


def _validate_plan_integrity(plan_dir: Path, plan: Mapping[str, Any]) -> None:
    expected = stable_hash(_plan_semantic_material(plan))
    recorded = str(plan.get("scientific_plan_hash") or "")
    plan_id = str(plan.get("plan_id") or "")
    if recorded != expected:
        raise RuntimeError("E046 large scientific plan content hash mismatch")
    if plan_id != expected[:16] or plan_dir.name != plan_id:
        raise RuntimeError("E046 large plan id does not match its scientific hash")
    _validate_runtime_scientific_contract(
        dict(plan.get("runtime_scientific_contract") or {})
    )

    catalog_snapshot = plan_dir / "prompt-catalog.json"
    doe_snapshot = plan_dir / "doe.json"
    if not catalog_snapshot.is_file() or not doe_snapshot.is_file():
        raise FileNotFoundError("E046 large frozen catalog/DOE snapshot is incomplete")
    if sha256_file(catalog_snapshot) != str(plan.get("catalog_file_sha256") or ""):
        raise RuntimeError("E046 large frozen prompt catalog file hash mismatch")
    if sha256_file(doe_snapshot) != str(plan.get("doe_file_sha256") or ""):
        raise RuntimeError("E046 large frozen DOE file hash mismatch")
    load_catalog(catalog_snapshot)
    load_doe(doe_snapshot)


def create_plan(
    *,
    output_root: Path,
    profile: str,
    source_commit: str,
    runtime_image: str,
    runtime_image_digest: str,
    runtime_scientific_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = _assert_isolated_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    plan = scientific_plan(
        profile=profile,
        source_commit=source_commit,
        runtime_image=runtime_image,
        runtime_image_digest=runtime_image_digest,
        runtime_scientific_contract=runtime_scientific_contract,
    )
    plan_dir = output_root / str(plan["plan_id"])
    plan_path = plan_dir / "plan.json"
    if plan_dir.exists():
        if not plan_path.is_file():
            raise RuntimeError(f"incomplete E046 large plan directory: {plan_dir}")
        existing = _load_json(plan_path)
        _validate_plan_integrity(plan_dir, existing)
        if existing.get("scientific_plan_hash") != plan["scientific_plan_hash"]:
            raise RuntimeError(f"E046 large plan-id collision: {plan_dir}")
        plan = existing
    else:
        staging = output_root / f".{plan['plan_id']}.planning-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            plan = {**plan, "created_at_utc": utc_now()}
            atomic_write_json(staging / "plan.json", plan)
            shutil.copy2(CATALOG_PATH, staging / "prompt-catalog.json")
            shutil.copy2(DOE_PATH, staging / "doe.json")
            for name in (
                "parents",
                "refinements",
                "attempts",
                "failures",
                "dataset",
                "analysis",
                "control",
                "cache/qr-verify",
            ):
                (staging / name).mkdir(parents=True, exist_ok=True)
            os.replace(staging, plan_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        _validate_plan_integrity(plan_dir, plan)
    atomic_write_json(
        output_root / "LATEST.json",
        {
            "schema": LATEST_SCHEMA,
            "plan_id": plan["plan_id"],
            "plan_dir": str(plan_dir),
            "profile": profile,
            "source_commit": source_commit,
            "status": "planned",
            "updated_at_utc": utc_now(),
        },
    )
    return plan


def resolve_plan_dir(output_root: Path, plan_id: str | None = None) -> Path:
    output_root = _assert_isolated_output_root(output_root)
    if plan_id:
        plan_dir = output_root / plan_id
    else:
        latest = _load_json(output_root / "LATEST.json")
        plan_dir = output_root / str(latest["plan_id"])
    plan_dir = plan_dir.resolve()
    if output_root not in plan_dir.parents:
        raise RuntimeError("E046 large plan path escapes its dedicated output root")
    if not (plan_dir / "plan.json").is_file():
        raise FileNotFoundError(f"E046 large plan missing: {plan_dir}")
    return plan_dir


def load_plan(
    output_root: Path, plan_id: str | None = None
) -> tuple[Path, dict[str, Any]]:
    plan_dir = resolve_plan_dir(output_root, plan_id)
    plan = _load_json(plan_dir / "plan.json")
    if plan.get("experiment") != EXPERIMENT or plan.get("schema") != PLAN_SCHEMA:
        raise RuntimeError(f"unexpected E046 large plan: {plan.get('experiment')}")
    _validate_plan_integrity(plan_dir, plan)
    return plan_dir, plan


def plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    budget = dict(plan["budget_estimate"])
    return {
        "experiment": plan["experiment"],
        "profile": plan["profile"],
        "plan_id": plan["plan_id"],
        "scientific_plan_sha256": plan["scientific_plan_hash"],
        "source_commit": plan["source_commit"],
        "runtime_image": plan["runtime_image"],
        "runtime_image_digest": plan["runtime_image_digest"],
        "scientific_contract_sha256": plan["runtime_scientific_contract"][
            "contract_sha256"
        ],
        "prompt_catalog_sha256": plan["catalog_sha256"],
        "doe_sha256": plan["doe_sha256"],
        "prompts": plan["expected_prompt_count"],
        "configs_per_prompt": plan["profile_spec"]["config_count"],
        "parents": plan["expected_parent_count"],
        "selected_srmpgd_parent_maximum": plan[
            "expected_selected_parent_maximum"
        ],
        "srmpgd_trajectory_maximum": plan[
            "expected_refinement_count_maximum"
        ],
        "srmpgd_checkpoint_estimate": plan[
            "expected_checkpoint_count_estimate"
        ],
        "doe_parameters": DOE_PARAMETER_COUNT,
        "prompt_structural_features": PROMPT_STRUCTURAL_FEATURE_COUNT,
        "estimated_hours": budget["wall_time_hours_sequential_estimate"],
        "estimated_gpu_generation_hours": budget["gpu_generation_hours_estimate"],
        "estimated_cpu_scoring_hours": budget["cpu_scoring_hours_estimate"],
        "estimated_disk_gib": budget["disk_gib_estimate"],
        "full_requires_explicit_profile": True,
    }


def _record_failure(
    *,
    plan_dir: Path,
    task_kind: str,
    task_id: str,
    error: BaseException,
    classification: str = "technical_failure",
    retryable: bool = False,
    usable: bool = False,
) -> dict[str, Any]:
    record = {
        "experiment": EXPERIMENT,
        "plan_id": plan_dir.name,
        "task_kind": task_kind,
        "task_id": task_id,
        "classification": classification,
        "retryable": bool(retryable),
        "usable": bool(usable),
        "error_class": type(error).__name__,
        "error": str(error)[:8000],
        "timestamp_utc": utc_now(),
    }
    path = plan_dir / "failures" / (
        f"{task_kind}-{task_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}.json"
    )
    atomic_write_json(path, record)
    return record


def _technical_failure_audit(
    plan_dir: Path,
    plan: Mapping[str, Any],
    *,
    deep: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Separate historical failures from failures still blocking completion.

    A failed attempt is never deleted: it remains part of the scientific audit.
    It stops blocking the campaign only when the exact task has subsequently
    produced a valid atomic promotion (or a valid terminal SRL marker).
    """

    recipes = {
        str(item["id"]): SRMPGDRecipe(**item)
        for item in plan.get("srmpgd_recipes", [])
    }
    history: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for path in sorted((plan_dir / "failures").glob("*.json")):
        record = _load_json(path)
        record = {**record, "failure_record_path": str(path.relative_to(plan_dir))}
        history.append(record)
        task_kind = str(record.get("task_kind") or "")
        task_id = str(record.get("task_id") or "")
        resolved = False
        if task_kind == "parent":
            resolved = _promotion_valid(
                _parent_dir(plan_dir, task_id),
                PARENT_GENERATION_REQUIRED,
                deep=deep,
            )
        elif task_kind == "score-parent":
            resolved = _promotion_valid(
                _parent_dir(plan_dir, task_id) / "scoring",
                SCORING_REQUIRED,
                deep=deep,
            )
        elif task_kind in {"refinement", "score-refinement"}:
            candidate_id, separator, recipe_id = task_id.rpartition("__")
            recipe = recipes.get(recipe_id) if separator else None
            if recipe is not None:
                root = _refinement_dir(plan_dir, candidate_id, recipe_id)
                terminal = _promotion_valid(
                    root,
                    ("task.json", "TERMINAL_FAILURE.json"),
                    deep=deep,
                )
                if task_kind == "refinement":
                    resolved = terminal or _promotion_valid(
                        root,
                        _refinement_generation_required(recipe),
                        deep=deep,
                    )
                else:
                    resolved = terminal or _promotion_valid(
                        root / "scoring",
                        SCORING_REQUIRED,
                        deep=deep,
                    )
        record["resolved_by_valid_promotion"] = bool(resolved)
        if not resolved:
            unresolved.append(record)
    return {"history": history, "unresolved": unresolved}


def _has_unpromoted_terminal_srl_failure(plan_dir: Path, task_id: str) -> bool:
    """Detect a recorded non-retryable SRL mismatch not sealed as terminal."""

    for path in sorted((plan_dir / "failures").glob("*.json")):
        record = _load_json(path)
        if (
            record.get("task_kind") == "refinement"
            and record.get("task_id") == task_id
            and record.get("classification") == "scientific_fidelity_mismatch"
            and record.get("retryable") is False
        ):
            return True
    return False


def _terminal_srl_retry_overridden() -> bool:
    value = str(
        os.environ.get("PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL", "0")
    ).strip()
    if value not in {"0", "1"}:
        raise RuntimeError(
            "PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL must be 0 or 1"
        )
    return value == "1"


def _assert_runtime_provenance(plan: Mapping[str, Any]) -> tuple[str, str]:
    """Fail closed when a GPU task is not running in the frozen OCI image."""
    runtime_image = str(os.environ.get("PROOFTAG_RUNTIME_IMAGE") or "")
    runtime_digest = str(os.environ.get("PROOFTAG_RUNTIME_IMAGE_DIGEST") or "")
    if runtime_image != str(plan["runtime_image"]):
        raise RuntimeError(
            f"runtime image differs from plan: {runtime_image!r} != "
            f"{plan['runtime_image']!r}"
        )
    if runtime_digest != str(plan["runtime_image_digest"]):
        raise RuntimeError("runtime image digest differs from scientific plan")
    try:
        embedded_commit = BUILD_COMMIT_ATTESTATION.read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise RuntimeError("embedded build commit attestation is unavailable") from exc
    if embedded_commit != str(plan["source_commit"]):
        raise RuntimeError("embedded build commit differs from scientific plan")
    return runtime_image, runtime_digest


def _settings_for_plan(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recipe: ParentRecipe,
) -> Any:
    settings = _settings_for(candidate, recipe)
    contract = dict(plan["runtime_scientific_contract"])
    frozen = dict(contract["settings"])
    settings = type(settings).model_validate({**settings.model_dump(), **frozen})
    effective = _runtime_scientific_contract(settings)
    if effective != contract:
        raise RuntimeError("effective E046 models/scorers differ from frozen plan")
    return settings


def _assert_qr_scorer_contract(plan: Mapping[str, Any], scorer: Any) -> None:
    expected = dict(plan["runtime_scientific_contract"]["qr_verify"])
    effective = {
        "engine_version": str(scorer.engine_version),
        "preset_count": int(getattr(scorer.decoder, "preset_count", 0)),
        "repetitions": int(scorer.repetitions),
        "scoring_version": str(scorer.scoring_version),
        "implementation_sha256": str(scorer.implementation_sha256),
    }
    if effective != expected:
        raise RuntimeError("effective QR-Verify scorer differs from frozen E046 plan")


def _seed_e046_strict(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    _seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def _determinism_runtime() -> dict[str, Any]:
    import torch

    return {
        "strict": True,
        "seeded": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _parent_recipe(plan: Mapping[str, Any], candidate: Mapping[str, Any]) -> ParentRecipe:
    try:
        return _recipe_map(plan)[str(candidate["parent_recipe_id"])]
    except KeyError as exc:
        raise KeyError(
            f"parent recipe missing from plan: {candidate['parent_recipe_id']}"
        ) from exc


def generate_parent(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    source_commit: str,
) -> dict[str, Any]:
    """Generate exactly one independent Stage1/Stage2 pair and preserve z0."""
    import torch
    from safetensors.torch import save_file

    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .schemas import GenerationRequest

    source_commit = _validate_commit(source_commit)
    plan_dir, plan = load_plan(output_root, plan_id)
    if source_commit != plan["source_commit"]:
        raise RuntimeError("E046 large parent source commit differs from plan")
    candidate = _candidate(plan, candidate_id)
    recipe = _parent_recipe(plan, candidate)
    final_dir = _parent_dir(plan_dir, candidate_id)
    marker = final_dir / "GENERATION_COMPLETE.json"
    if marker.is_file():
        if _promotion_valid(final_dir, PARENT_GENERATION_REQUIRED):
            return _load_json(marker)
        raise RuntimeError(f"corrupt promoted parent: {candidate_id}")
    if final_dir.exists():
        raise FileExistsError(f"incomplete parent final directory: {final_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("E046 large parent generation requires CUDA")

    attempt = (
        plan_dir
        / "attempts/parents"
        / f"{candidate_id}-{source_commit[:12]}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        attempt / "task.json",
        {
            "experiment": EXPERIMENT,
            "kind": "independent_parent",
            "candidate": candidate,
            "parent_recipe": asdict(recipe),
            "source_commit": source_commit,
            "started_at_utc": utc_now(),
        },
    )
    started = time.perf_counter()
    backend = None
    try:
        runtime_image, runtime_digest = _assert_runtime_provenance(plan)
        settings = _settings_for_plan(plan, candidate, recipe)
        _seed_e046_strict(int(candidate["seed"]))
        backend = UpstreamDiffQRCoderBackend(settings)
        blueprint = _blueprint(candidate, recipe)
        request = GenerationRequest(
            payload=str(candidate["payload"]),
            prompt=str(candidate["prompt"]),
            negative_prompt=NEGATIVE_PROMPT,
            backend="controlnet",
            error_correction=recipe.error_correction,
            seed=int(candidate["seed"]),
            steps=recipe.stage1_steps,
            guidance_scale=recipe.stage1_guidance_scale,
            controlnet_scale=recipe.stage1_controlnet_scale,
            strength=1.0,
            max_attempts=1,
        )
        torch.cuda.reset_peak_memory_stats()
        stage1 = backend.generate(request, blueprint, int(candidate["seed"])).convert("RGB")
        if stage1.size != (CANVAS_PX, CANVAS_PX):
            raise RuntimeError(f"Stage1 expected 736x736, got {stage1.size}")
        _save_png(attempt / "images/stage1-raw.png", stage1)

        stage2 = backend._run_stage2(
            stage1,
            blueprint,
            request,
            int(candidate["seed"]),
        ).convert("RGB")
        state = backend.export_stage2_state()
        if state is None:
            raise RuntimeError("Stage2 produced no exportable latent")
        if stage2.size != (CANVAS_PX, CANVAS_PX):
            raise RuntimeError(f"Stage2 expected 736x736, got {stage2.size}")
        latent = state["latent"].detach().cpu().contiguous()
        _save_png(attempt / "images/stage2-raw.png", stage2)
        save_file(
            {"latent": latent},
            str(attempt / "stage2-latent.safetensors"),
            metadata={
                "experiment": EXPERIMENT,
                "candidate_id": candidate_id,
                "source_commit": source_commit,
            },
        )
        metadata = {
            "experiment": EXPERIMENT,
            "kind": "independent_parent",
            "candidate": candidate,
            "parent_recipe": asdict(recipe),
            "source_commit": source_commit,
            "runtime_image": runtime_image,
            "runtime_image_digest": runtime_digest,
            "scientific_contract_sha256": plan["runtime_scientific_contract"][
                "contract_sha256"
            ],
            "settings": _safe_settings_provenance(settings),
            "determinism": _determinism_runtime(),
            "stage1_image_sha256": _image_sha256(stage1),
            "stage1_image_file_sha256": sha256_file(
                attempt / "images/stage1-raw.png"
            ),
            "stage2_image_sha256": _image_sha256(stage2),
            "stage2_image_file_sha256": sha256_file(
                attempt / "images/stage2-raw.png"
            ),
            "stage2_latent_file_sha256": sha256_file(
                attempt / "stage2-latent.safetensors"
            ),
            "stage2_latent_tensor_sha256": str(state["latent_sha256"]),
            "stage2_diagnostics": dict(state.get("diagnostics") or {}),
            "primary_training_observation": "stage2_raw",
            "scene_qz_generated": False,
            "elapsed_s": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "parent-metadata.json", metadata)
        completion = {
            "experiment": EXPERIMENT,
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "generation_group_id": candidate["generation_group_id"],
            "generation_complete": True,
            "stage1_image_sha256": metadata["stage1_image_sha256"],
            "stage2_image_sha256": metadata["stage2_image_sha256"],
            "stage2_latent_file_sha256": metadata["stage2_latent_file_sha256"],
            "stage2_latent_tensor_sha256": metadata["stage2_latent_tensor_sha256"],
            "elapsed_s": metadata["elapsed_s"],
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "GENERATION_COMPLETE.json", completion)
        promoted = _promote_e046_attempt(
            attempt_dir=attempt,
            final_dir=final_dir,
            required_files=PARENT_GENERATION_REQUIRED,
            metadata={
                "experiment": EXPERIMENT,
                "kind": "independent_parent",
                "candidate_id": candidate_id,
                "source_commit": source_commit,
            },
        )
        return {**completion, "promotion_manifest_hash": promoted["manifest_hash"]}
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="parent",
            task_id=candidate_id,
            error=exc,
        )
        raise
    finally:
        if backend is not None:
            backend._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _new_qr_scorer(plan_dir: Path, plan: Mapping[str, Any]) -> Any:
    from .validation import (
        ConservativeQRVerifyScorer,
    )

    scorer = ConservativeQRVerifyScorer(
        repetitions=3,
        cache_dir=plan_dir / "cache/qr-verify",
    )
    _assert_qr_scorer_contract(plan, scorer)
    return scorer


def _score_qr_cached(
    *,
    plan_dir: Path,
    image: Image.Image,
    payload: str,
    scorer: Any | None = None,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .validation import canonical_conservative_qr_verify_evidence

    owned_scorer = scorer is None
    if scorer is None and plan is None:
        _, plan = load_plan(plan_dir.parent, plan_dir.name)
    assert plan is not None
    scorer = scorer or _new_qr_scorer(plan_dir, plan)
    try:
        return canonical_conservative_qr_verify_evidence(
            scorer.score(image, payload)
        )
    finally:
        if owned_scorer:
            scorer.close()


def _quality_scores(
    *,
    images: Mapping[str, Image.Image],
    prompt: str,
    settings: Any,
    scorer: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .quality_scoring import quality_scorer_from_settings

    scorer = scorer or quality_scorer_from_settings(settings, device="cpu")
    if bool(getattr(scorer, "hps_enabled", False)):
        scorer.hps_enabled = False
        try:
            results = {
                name: asdict(scorer.score(image, prompt))
                for name, image in images.items()
            }
        finally:
            scorer.hps_enabled = True
        hps = _e046_hps_scores(scorer, images, prompt)
        for name, value in hps.items():
            results[name]["hpsv2_1"] = value
    else:
        results = {
            name: asdict(scorer.score(image, prompt))
            for name, image in images.items()
        }
    return results, scorer.provenance()


def _e046_hps_scores(
    scorer: Any,
    images: Mapping[str, Image.Image],
    prompt: str,
) -> dict[str, float | None]:
    """Score HPS v2.1 with one pinned checkpoint load per E046 scoring run."""
    import importlib

    import numpy as np
    import torch

    from .quality_scoring import QualityScoringError

    try:
        checkpoint = scorer._ensure_hps_checkpoint()
        cache_key = (
            str(checkpoint),
            str(scorer.hps_model_version),
            str(scorer.hps_checkpoint_sha256),
        )
        runtime = getattr(scorer, "_e046_hps_runtime", None)
        if runtime is None or runtime["cache_key"] != cache_key:
            module = importlib.import_module("hpsv2.img_score")
            module.device = "cpu"
            module.initialize_model()
            model = module.model_dict["model"]
            state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state["state_dict"])
            model.requires_grad_(False).eval().to("cpu")
            runtime = {
                "cache_key": cache_key,
                "model": model,
                "preprocess": module.model_dict["preprocess_val"],
                "tokenizer": module.get_tokenizer("ViT-H-14"),
            }
            scorer._e046_hps_runtime = runtime

        text = runtime["tokenizer"]([prompt]).to("cpu", non_blocking=False)
        output: dict[str, float | None] = {}
        for name, image in images.items():
            tensor = runtime["preprocess"](image.convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                prediction = runtime["model"](tensor.to("cpu"), text)
                value = (
                    prediction["image_features"]
                    @ prediction["text_features"].T
                ).diagonal()[0]
            number = float(value.detach().cpu())
            if not np.isfinite(number):
                raise QualityScoringError("HPS v2.1 returned a non-finite score")
            output[name] = number
        return output
    except Exception as exc:
        if bool(getattr(scorer, "hps_fail_closed", False)):
            if isinstance(exc, QualityScoringError):
                raise
            raise QualityScoringError(
                f"E046 HPS v2.1 scoring failed: {type(exc).__name__}: {exc}"
            ) from exc
        return {name: None for name in images}


def _qr_fields(evidence: Mapping[str, Any]) -> dict[str, Any]:
    exact = int(evidence.get("conservative_exact_presets", 0))
    preset_count = int(evidence.get("preset_count", QR_VERIFY_PRESET_COUNT))
    return {
        "wechat_engine": QR_SOFTWARE_ENGINE,
        "wechat_preset_count": preset_count,
        "wechat_repetitions": int(evidence.get("repetitions", 3)),
        "wechat_exact_presets": exact,
        "wechat_exact_rate": exact / preset_count if preset_count else 0.0,
        "wechat_original_exact": bool(
            evidence.get("direct_exact_all_repetitions")
        ),
        "wechat_instability": int(evidence.get("unstable_preset_count", 0)),
        "direct_exact_all_repetitions": bool(
            evidence.get("direct_exact_all_repetitions")
        ),
        "each_repetition_any_exact": bool(
            evidence.get("each_repetition_any_exact")
        ),
        "consistent_any_exact": bool(evidence.get("consistent_any_exact")),
        "unstable_preset_count": int(evidence.get("unstable_preset_count", 0)),
        "stable_preset_count": int(evidence.get("stable_preset_count", 0)),
        "qr_verify_minimum_tolerance_score": _finite_number(
            evidence.get("minimum_tolerance_score"), 0.0
        ),
        "qr_verify_mean_tolerance_score": _finite_number(
            evidence.get("mean_tolerance_score"), 0.0
        ),
        "qr_verify_maximum_tolerance_score": _finite_number(
            evidence.get("maximum_tolerance_score"), 0.0
        ),
        "qr_verify_cache_key": evidence.get("cache_key"),
        "qr_verify_implementation_sha256": evidence.get("implementation_sha256"),
        "qr_verify_engine_version": evidence.get("engine_version"),
        "qr_verify_scoring_version": evidence.get("scoring_version"),
    }


def _recipe_features(recipe: ParentRecipe) -> dict[str, Any]:
    strength_applicable = recipe.stage2_initialization == "paper_stage1_noise"
    return {
        "stage1_steps": recipe.stage1_steps,
        "stage1_guidance_scale": recipe.stage1_guidance_scale,
        "stage1_controlnet_scale": recipe.stage1_controlnet_scale,
        "control_guidance_start": recipe.control_guidance_start,
        "control_guidance_end": recipe.control_guidance_end,
        "stage2_initialization": recipe.stage2_initialization,
        "stage2_strength": recipe.stage2_strength,
        "stage2_strength_applicable": strength_applicable,
        "stage2_strength_effective": (
            recipe.stage2_strength if strength_applicable else None
        ),
        "stage2_steps": recipe.stage2_steps,
        "stage2_steps_planned": recipe.stage2_steps,
        "stage2_guidance_scale_effective": recipe.stage1_guidance_scale,
        "stage2_control_guidance_start_effective": recipe.control_guidance_start,
        "stage2_control_guidance_end_effective": recipe.control_guidance_end,
        "guidance_scale_scope": "shared_stage1_stage2_upstream_request",
        "control_guidance_scope": "shared_stage1_stage2_upstream_settings",
        "stage2_controlnet_scale": recipe.stage2_controlnet_scale,
        "stage2_qr_weight": recipe.stage2_qr_weight,
        "stage2_perceptual_weight": recipe.stage2_perceptual_weight,
    }


def _candidate_features(candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "prompt_id",
        "prompt_ordinal",
        "prompt_family",
        "prompt",
        "payload",
        "payload_length",
        "spatial_frequency",
        "symmetry",
        "brightness",
        "contrast",
        "colorfulness",
        "negative_space",
        "texture_density",
        "geometry_type",
        "composition_type",
        "config_id",
        "config_index",
        "mask_slot",
        "parent_recipe_id",
        "seed",
        "generation_group_id",
        "prompt_group_id",
        "doe_isolation_score",
    )
    return {key: candidate.get(key) for key in keys}


def _stage2_parent_row(
    *,
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recipe: ParentRecipe,
    parent_dir: Path,
    parent_metadata: Mapping[str, Any],
    stage1: Image.Image,
    stage2: Image.Image,
    qr_evidence: Mapping[str, Any],
    quality: Mapping[str, Any],
    stage1_quality: Mapping[str, Any],
) -> dict[str, Any]:
    from .quality import image_change_metrics, image_quality_metrics

    blueprint = _blueprint(candidate, recipe)
    row = {
        "schema": DATASET_SCHEMA,
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "source_commit": parent_metadata.get("source_commit"),
        "runtime_image": parent_metadata.get("runtime_image"),
        "runtime_image_digest": parent_metadata.get("runtime_image_digest"),
        "scientific_contract_sha256": parent_metadata.get(
            "scientific_contract_sha256"
        ),
        "source_kind": "parent",
        "candidate_id": candidate["id"],
        "parent_id": candidate["id"],
        "trajectory_id": None,
        "observation_independence_class": "independent_parent",
        "stage": "stage2",
        "variant": "stage2_raw",
        "raw": True,
        "quiet_zone_variant": "raw",
        "primary_training_observation": True,
        "diagnostic_variant": False,
        "stage1_final_eligible": False,
        "scene_qz_final_eligible": False,
        "image_path": str(parent_dir / "images/stage2-raw.png"),
        "latent_path": str(parent_dir / "stage2-latent.safetensors"),
        "image_sha256": _image_sha256(stage2),
        "image_file_sha256": sha256_file(parent_dir / "images/stage2-raw.png"),
        "latent_sha256": parent_metadata["stage2_latent_tensor_sha256"],
        "latent_file_sha256": sha256_file(parent_dir / "stage2-latent.safetensors"),
        "payload_sha256": hashlib.sha256(
            str(candidate["payload"]).encode("utf-8")
        ).hexdigest(),
        "qr_version": QR_VERSION,
        "error_correction": recipe.error_correction,
        "qr_mask_pattern": recipe.qr_mask_pattern,
        "qr_module_size": QR_MODULE_SIZE,
        "qr_padding_px": QR_PADDING_PX,
        "iteration": 0,
        "srmpgd_recipe_id": None,
        "gamma": None,
        "latent_radius_rms": None,
        "srmpgd_max_iterations": None,
        "srmpgd_lpips_weight": None,
        "stage1_quality_scored": False,
        "stage2_effective_steps": int(
            round(
                _finite_number(
                    dict(parent_metadata.get("stage2_diagnostics") or {}).get(
                        "diffqrcoder_stage2_effective_steps"
                    ),
                    recipe.stage2_steps,
                )
            )
        ),
        **_candidate_features(candidate),
        **_recipe_features(recipe),
        **_qr_fields(qr_evidence),
        **_module_fields(stage2, blueprint),
        **image_change_metrics(stage2, stage1),
        **image_quality_metrics(stage2),
        **dict(quality),
    }
    guard = _parent_visual_guard(
        row=row,
        stage1_quality=stage1_quality,
        scene_qz_guard=None,
    )
    row["visual_guard_pass"] = bool(guard["passed"])
    row["visual_guard_checks"] = guard["checks"]
    row["visual_guard_policy"] = guard["policy"]
    row["eligible_for_refinement"] = bool(row["visual_guard_pass"])
    row["eligible_final"] = bool(row["visual_guard_pass"])
    minimum = int(plan["validity_policy"]["final_minimum_exact_presets"])
    row["delivery_eligible"] = bool(
        row["visual_guard_pass"]
        and row["raw"]
        and row["wechat_original_exact"]
        and row["wechat_exact_presets"] >= minimum
    )
    return row


def score_parent(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    qr_scorer: Any | None = None,
    quality_scorer: Any | None = None,
    _plan_context: tuple[Path, Mapping[str, Any]] | None = None,
    _candidate_context: Mapping[str, Any] | None = None,
    _recipe_context: ParentRecipe | None = None,
) -> dict[str, Any]:
    plan_dir, plan = _plan_context or load_plan(output_root, plan_id)
    candidate = _candidate_context or _candidate(plan, candidate_id)
    recipe = _recipe_context or _parent_recipe(plan, candidate)
    parent_dir = _parent_dir(plan_dir, candidate_id)
    if not _promotion_valid(
        parent_dir, PARENT_GENERATION_REQUIRED, deep=True
    ):
        raise FileNotFoundError(f"parent generation incomplete: {candidate_id}")
    root_marker = parent_dir / "SCORING_COMPLETE.json"
    scoring_dir = parent_dir / "scoring"
    nested_marker = scoring_dir / "SCORING_COMPLETE.json"
    scoring_valid = _promotion_valid(
        scoring_dir, SCORING_REQUIRED, deep=True
    )
    if root_marker.is_file() and scoring_valid:
        return _load_json(root_marker)
    if nested_marker.is_file() and scoring_valid:
        completion = _load_json(nested_marker)
        atomic_write_json(root_marker, completion)
        return completion
    if scoring_dir.exists():
        raise FileExistsError(f"incomplete scoring directory: {scoring_dir}")

    attempt = (
        plan_dir
        / "attempts/scoring-parents"
        / f"{candidate_id}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        stage1 = Image.open(parent_dir / "images/stage1-raw.png").convert("RGB")
        stage2 = Image.open(parent_dir / "images/stage2-raw.png").convert("RGB")
        metadata = _load_json(parent_dir / "parent-metadata.json")
        settings = _settings_for_plan(plan, candidate, recipe)
        quality, provenance = _quality_scores(
            images={"stage2_raw": stage2},
            prompt=str(candidate["prompt"]),
            settings=settings,
            scorer=quality_scorer,
        )
        provenance = {
            **provenance,
            "scientific_contract_sha256": plan["runtime_scientific_contract"][
                "contract_sha256"
            ],
        }
        qr_evidence = _score_qr_cached(
            plan_dir=plan_dir,
            image=stage2,
            payload=str(candidate["payload"]),
            scorer=qr_scorer,
            plan=plan,
        )
        row = _stage2_parent_row(
            plan=plan,
            candidate=candidate,
            recipe=recipe,
            parent_dir=parent_dir,
            parent_metadata=metadata,
            stage1=stage1,
            stage2=stage2,
            qr_evidence=qr_evidence,
            quality=quality["stage2_raw"],
            stage1_quality={},
        )
        atomic_write_json(attempt / "comparison.json", [row])
        _write_csv(attempt / "comparison.csv", _flatten_rows([row]))
        atomic_write_json(attempt / "qr-verify-evidence.json", qr_evidence)
        atomic_write_json(attempt / "quality-scores.json", quality)
        atomic_write_json(attempt / "quality-provenance.json", provenance)
        completion = {
            "experiment": EXPERIMENT,
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "row_count": 1,
            "primary_training_observation_count": 1,
            "wechat_exact_presets": row["wechat_exact_presets"],
            "wechat_original_exact": row["wechat_original_exact"],
            "scoring_complete": True,
            "elapsed_s": time.perf_counter() - started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "SCORING_COMPLETE.json", completion)
        _promote_e046_attempt(
            attempt_dir=attempt,
            final_dir=scoring_dir,
            required_files=SCORING_REQUIRED,
            metadata={
                "experiment": EXPERIMENT,
                "kind": "score-parent",
                "candidate_id": candidate_id,
            },
        )
        atomic_write_json(root_marker, completion)
        return completion
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="score-parent",
            task_id=candidate_id,
            error=exc,
        )
        raise


def score_all_parents(*, output_root: Path, plan_id: str) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    from .quality_scoring import quality_scorer_from_settings

    candidates = list(plan["candidates"])
    pending = [
        candidate
        for candidate in candidates
        if not _promotion_valid(
            _parent_dir(plan_dir, str(candidate["id"])) / "scoring",
            SCORING_REQUIRED,
        )
    ]
    qr_scorer = _new_qr_scorer(plan_dir, plan) if pending else None
    quality_scorer = None
    if pending:
        first = pending[0]
        quality_scorer = quality_scorer_from_settings(
            _settings_for_plan(plan, first, _parent_recipe(plan, first)),
            device="cpu",
        )
    parent_recipe_map = _recipe_map(plan)
    try:
        results = [
            score_parent(
                output_root=output_root,
                plan_id=plan_id,
                candidate_id=str(candidate["id"]),
                qr_scorer=qr_scorer,
                quality_scorer=quality_scorer,
                _plan_context=(plan_dir, plan),
                _candidate_context=candidate,
                _recipe_context=parent_recipe_map[
                    str(candidate["parent_recipe_id"])
                ],
            )
            for candidate in candidates
        ]
    finally:
        if qr_scorer is not None:
            qr_scorer.close()
        del quality_scorer
        gc.collect()
    completion = {
        "experiment": EXPERIMENT,
        "plan_id": plan_id,
        "candidate_count": len(plan["candidates"]),
        "scored_count": sum(bool(row.get("scoring_complete")) for row in results),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "PARENT_SCORING_COMPLETE.json", completion)
    return completion


def _parent_rows(plan_dir: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        parent_root = _parent_dir(plan_dir, str(candidate["id"]))
        scoring_root = parent_root / "scoring"
        path = scoring_root / "comparison.json"
        if _promotion_valid(
            parent_root, PARENT_GENERATION_REQUIRED, deep=True
        ) and _promotion_valid(scoring_root, SCORING_REQUIRED, deep=True):
            rows.extend(_load_json(path))
    return rows


def _visual_score(row: Mapping[str, Any]) -> float:
    values = (
        _finite_number(row.get("clip_score"), -1e9),
        _finite_number(row.get("hpsv2_1"), -1e9),
        _finite_number(row.get("clip_aesthetic"), -1e9),
    )
    if min(values) <= -1e8:
        return -1e9
    # Only a selector heuristic. Final normalized targets are computed later.
    return values[0] + values[1] + values[2] / 10.0


def _selection_key(row: Mapping[str, Any], reason: str) -> tuple[Any, ...]:
    exact = int(row.get("wechat_exact_presets") or 0)
    original = int(bool(row.get("wechat_original_exact")))
    visual = _visual_score(row)
    unstable = int(row.get("unstable_preset_count") or 0)
    isolation = _finite_number(row.get("doe_isolation_score"), 0.0)
    candidate_id = str(row.get("candidate_id"))
    if reason == "best_valid":
        return (-int(original and exact >= 34), -exact, -visual, candidate_id)
    if reason == "borderline":
        return (abs(exact - 34), -original, -unstable, -visual, candidate_id)
    if reason == "high_aesthetic_low_qr":
        return (int(exact >= 34), -visual, exact, candidate_id)
    if reason == "high_qr_low_aesthetic":
        return (-exact, visual, -original, candidate_id)
    if reason == "unstable":
        return (-unstable, abs(exact - 34), -visual, candidate_id)
    if reason == "sparse_doe_region":
        return (-isolation, abs(exact - 25), -visual, candidate_id)
    raise ValueError(f"unknown selection stratum: {reason}")


def _selection_reason(prompt_id: str) -> str:
    reasons = tuple(
        (
            "best_valid",
            "borderline",
            "high_aesthetic_low_qr",
            "high_qr_low_aesthetic",
            "unstable",
            "sparse_doe_region",
        )
    )
    index = int(hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:8], 16)
    return reasons[index % len(reasons)]


def _assign_srmpgd_recipes(
    candidate_id: str, recipes_per_parent: int
) -> list[str]:
    offset = int(
        hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16
    ) % len(SRMPGD_RECIPES)
    return [
        SRMPGD_RECIPES[(offset + index) % len(SRMPGD_RECIPES)].id
        for index in range(recipes_per_parent)
    ]


def select_refinements(*, output_root: Path, plan_id: str) -> dict[str, Any]:
    """Select an informative, family-diverse Phase-B subset automatically."""
    plan_dir, plan = load_plan(output_root, plan_id)
    path = plan_dir / "selected-refinements.json"
    if path.is_file():
        return _load_json(path)
    if not (plan_dir / "PARENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("parent scoring is not complete")
    rows = _parent_rows(plan_dir, plan)
    if len(rows) != int(plan["expected_parent_count"]):
        raise RuntimeError(
            f"parent observation count {len(rows)} != {plan['expected_parent_count']}"
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)

    primary: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    for prompt_id in sorted(grouped):
        group = grouped[prompt_id]
        reason = _selection_reason(prompt_id)
        chosen = dict(sorted(group, key=lambda row: _selection_key(row, reason))[0])
        chosen["selection_role"] = "primary"
        chosen["selection_reason"] = reason
        primary.append(chosen)
        for alternate_reason in (
            "borderline",
            "high_aesthetic_low_qr",
            "high_qr_low_aesthetic",
            "unstable",
            "sparse_doe_region",
            "best_valid",
        ):
            options = [
                row for row in group if row["candidate_id"] != chosen["candidate_id"]
            ]
            if not options:
                continue
            alternate = dict(
                sorted(options, key=lambda row: _selection_key(row, alternate_reason))[0]
            )
            alternate["selection_role"] = "secondary"
            alternate["selection_reason"] = alternate_reason
            alternate["secondary_priority"] = (
                -int(alternate.get("unstable_preset_count") or 0),
                abs(int(alternate.get("wechat_exact_presets") or 0) - 34),
                -_finite_number(alternate.get("doe_isolation_score"), 0.0),
                str(alternate["candidate_id"]),
            )
            alternatives.append(alternate)
            break

    secondary_target = int(plan["profile_spec"]["secondary_parent_count"])
    secondary: list[dict[str, Any]] = []
    if secondary_target:
        families = sorted({str(row["prompt_family"]) for row in alternatives})
        by_family = {
            family: sorted(
                [row for row in alternatives if row["prompt_family"] == family],
                key=lambda row: row["secondary_priority"],
            )
            for family in families
        }
        while len(secondary) < secondary_target:
            progressed = False
            for family in families:
                if by_family[family] and len(secondary) < secondary_target:
                    secondary.append(by_family[family].pop(0))
                    progressed = True
            if not progressed:
                break
    if len(secondary) != secondary_target:
        raise RuntimeError(
            f"selected {len(secondary)} secondary parents; expected {secondary_target}"
        )

    recipes_per_parent = int(plan["profile_spec"]["srmpgd_recipes_per_parent"])
    selected = primary + secondary
    tasks: list[dict[str, Any]] = []
    for row in selected:
        for recipe_id in _assign_srmpgd_recipes(
            str(row["candidate_id"]), recipes_per_parent
        ):
            tasks.append(
                {
                    "candidate_id": row["candidate_id"],
                    "parent_id": row["candidate_id"],
                    "prompt_id": row["prompt_id"],
                    "prompt_family": row["prompt_family"],
                    "selection_role": row["selection_role"],
                    "selection_reason": row["selection_reason"],
                    "srmpgd_recipe_id": recipe_id,
                    "trajectory_id": f"{row['candidate_id']}__{recipe_id}",
                }
            )
    expected = int(plan["expected_refinement_count_maximum"])
    if len(tasks) != expected:
        raise AssertionError(f"refinement task count {len(tasks)} != {expected}")
    payload = {
        "experiment": EXPERIMENT,
        "plan_id": plan_id,
        "primary_parent_count": len(primary),
        "secondary_parent_count": len(secondary),
        "selected_parent_count": len(selected),
        "task_count": len(tasks),
        "selected": [
            {
                key: value
                for key, value in row.items()
                if key not in {"secondary_priority"}
            }
            for row in selected
        ],
        "tasks": tasks,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(path, payload)
    atomic_write_json(plan_dir / "selected-parents.json", payload)
    _write_csv(plan_dir / "selected-refinements.csv", _flatten_rows(tasks))
    return payload


def refinement_tasks(
    plan_dir: Path, plan: Mapping[str, Any]
) -> list[tuple[str, str]]:
    del plan
    path = plan_dir / "selected-refinements.json"
    if not path.is_file():
        return []
    return [
        (str(item["candidate_id"]), str(item["srmpgd_recipe_id"]))
        for item in _load_json(path)["tasks"]
    ]


def _terminal_srl_failure(error: BaseException) -> bool:
    return TERMINAL_SRL_FRAGMENT.lower() in str(error).lower()


def _bind_e046_iteration_zero_to_parent(
    *,
    trajectory_root: Path,
    recipe_id: str,
    parent_image: Image.Image,
    checkpoints: Sequence[Any],
) -> dict[str, Any]:
    """Expose the exact Stage-2 parent as E046 checkpoint ``i0``.

    The inherited E040 trajectory evaluates its differentiable objective on
    ``D(z0)`` and writes that VAE-decoded raster.  E046 needs a stricter dataset
    contract: the externally scored no-op checkpoint must be pixel-identical to
    the persisted Stage-2 parent.  Keep the decoded diagnostic separately, then
    bind only this new experiment's i0 raster and trace to the exact parent.
    """
    root = trajectory_root / recipe_id
    trace_path = root / "trace.json"
    trace_csv_path = root / "trace.csv"
    image_path = root / "images/iteration-000.png"
    if not trace_path.is_file() or not image_path.is_file():
        raise FileNotFoundError("E046 SR-MPGD iteration-zero artifacts are missing")

    decoded_image = Image.open(image_path).convert("RGB")
    decoded_hash = _image_sha256(decoded_image)
    diagnostic_path = root / "diagnostics/iteration-000-vae-decoded.png"
    _save_png(diagnostic_path, decoded_image)

    exact_parent = parent_image.convert("RGB")
    parent_hash = _image_sha256(exact_parent)
    _save_png(image_path, exact_parent)
    with Image.open(image_path) as persisted:
        persisted_hash = _image_sha256(persisted.convert("RGB"))
    if persisted_hash != parent_hash:
        raise RuntimeError("E046 failed to persist an exact Stage-2 i0 raster")

    rows = _load_json(trace_path)
    if not isinstance(rows, list):
        raise TypeError("E046 SR-MPGD trace must be a JSON list")
    zero_rows = [row for row in rows if int(row.get("iteration", -1)) == 0]
    if len(zero_rows) != 1:
        raise RuntimeError(
            f"E046 SR-MPGD trace has {len(zero_rows)} iteration-zero rows"
        )
    zero = zero_rows[0]
    zero.update(
        {
            "image_sha256": parent_hash,
            "iteration_zero_exact_parent_raster": True,
            "external_scoring_raster_semantics": "exact_persisted_stage2_parent",
            "objective_raster_semantics": "vae_decoded_D_z0",
            "objective_raster_sha256": decoded_hash,
            "objective_raster_path": "diagnostics/iteration-000-vae-decoded.png",
        }
    )
    atomic_write_json(trace_path, rows)
    _write_csv(
        trace_csv_path,
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"candidate_checks", "cuda"}
                },
                **{
                    f"check_{key}": value
                    for key, value in (row.get("candidate_checks") or {}).items()
                },
                **{
                    f"cuda_{key}": value
                    for key, value in (row.get("cuda") or {}).items()
                },
            }
            for row in rows
        ],
    )

    zero_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if int(getattr(checkpoint, "iteration", -1)) == 0
    ]
    if len(zero_checkpoints) != 1:
        raise RuntimeError(
            f"E046 SR-MPGD returned {len(zero_checkpoints)} iteration-zero checkpoints"
        )
    zero_checkpoints[0].trace_step.update(zero)

    audit = {
        "experiment": EXPERIMENT,
        "recipe_id": recipe_id,
        "external_scoring_iteration": 0,
        "exact_parent_raster": True,
        "parent_image_sha256": parent_hash,
        "persisted_iteration_zero_image_sha256": persisted_hash,
        "vae_decoded_objective_image_sha256": decoded_hash,
        "vae_decoded_objective_image_path": (
            "diagnostics/iteration-000-vae-decoded.png"
        ),
        "objective_values_come_from": "vae_decoded_D_z0",
        "created_at_utc": utc_now(),
    }
    atomic_write_json(root / "iteration-zero-audit.json", audit)
    return audit


def _promote_terminal_refinement_failure(
    *,
    attempt: Path,
    final_dir: Path,
    plan_dir: Path,
    candidate_id: str,
    recipe_id: str,
    error: BaseException,
) -> dict[str, Any]:
    task_id = f"{candidate_id}__{recipe_id}"
    failure = _record_failure(
        plan_dir=plan_dir,
        task_kind="refinement",
        task_id=task_id,
        error=error,
        classification="scientific_fidelity_mismatch",
        retryable=False,
        usable=False,
    )
    marker = {
        **failure,
        "terminal": True,
        "abort_unrelated_tasks": False,
        "candidate_id": candidate_id,
        "srmpgd_recipe_id": recipe_id,
    }
    atomic_write_json(attempt / "TERMINAL_FAILURE.json", marker)
    _promote_e046_attempt(
        attempt_dir=attempt,
        final_dir=final_dir,
        required_files=("task.json", "TERMINAL_FAILURE.json"),
        metadata={
            "experiment": EXPERIMENT,
            "kind": "scientific-terminal-refinement",
            "candidate_id": candidate_id,
            "srmpgd_recipe_id": recipe_id,
        },
    )
    return marker


def generate_refinement(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    srmpgd_recipe_id: str,
    source_commit: str,
) -> dict[str, Any]:
    import torch

    from .diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from .e035_loss_fidelity import _offload_diffusion_modules
    from .e039_limiter_scanaware import E039Config
    from .e040_checkpoint_frontier import Recipe as TrajectoryRecipe
    from .e040_checkpoint_frontier import _run_trajectory

    source_commit = _validate_commit(source_commit)
    plan_dir, plan = load_plan(output_root, plan_id)
    if source_commit != plan["source_commit"]:
        raise RuntimeError("E046 large refinement source commit differs from plan")
    selected = select_refinements(output_root=output_root, plan_id=plan_id)
    enabled = {
        (str(item["candidate_id"]), str(item["srmpgd_recipe_id"]))
        for item in selected["tasks"]
    }
    if (candidate_id, srmpgd_recipe_id) not in enabled:
        raise ValueError("refinement task was not selected by Phase B")
    candidate = _candidate(plan, candidate_id)
    parent_recipe = _parent_recipe(plan, candidate)
    try:
        recipe = _srmpgd_recipe_map(plan)[srmpgd_recipe_id]
    except KeyError as exc:
        raise KeyError(f"unknown SR-MPGD recipe: {srmpgd_recipe_id}") from exc
    final_dir = _refinement_dir(plan_dir, candidate_id, srmpgd_recipe_id)
    complete = final_dir / "GENERATION_COMPLETE.json"
    terminal = final_dir / "TERMINAL_FAILURE.json"
    if complete.is_file():
        if _promotion_valid(final_dir, _refinement_generation_required(recipe)):
            return _load_json(complete)
        raise RuntimeError(
            f"corrupt promoted refinement: {candidate_id}/{srmpgd_recipe_id}"
        )
    if terminal.is_file():
        if _promotion_valid(final_dir, ("task.json", "TERMINAL_FAILURE.json")):
            return _load_json(terminal)
        raise RuntimeError(
            f"corrupt terminal refinement: {candidate_id}/{srmpgd_recipe_id}"
        )
    if final_dir.exists():
        raise FileExistsError(f"incomplete refinement final directory: {final_dir}")
    task_id = f"{candidate_id}__{srmpgd_recipe_id}"
    if (
        _has_unpromoted_terminal_srl_failure(plan_dir, task_id)
        and not _terminal_srl_retry_overridden()
    ):
        raise RuntimeError(
            "non-retryable SRL fidelity failure is recorded but its terminal "
            "promotion is absent; inspect the failure and set "
            "PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL=1 only for an explicit retry"
        )
    parent_root = _parent_dir(plan_dir, candidate_id)
    if not _promotion_valid(
        parent_root, PARENT_GENERATION_REQUIRED, deep=True
    ):
        raise FileNotFoundError(
            f"parent generation incomplete or corrupt: {candidate_id}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("E046 large SR-MPGD generation requires CUDA")

    attempt = (
        plan_dir
        / "attempts/refinements"
        / f"{candidate_id}-{srmpgd_recipe_id}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        attempt / "task.json",
        {
            "experiment": EXPERIMENT,
            "kind": "srmpgd",
            "candidate": candidate,
            "parent_recipe": asdict(parent_recipe),
            "srmpgd_recipe": asdict(recipe),
            "source_commit": source_commit,
            "started_at_utc": utc_now(),
        },
    )
    started = time.perf_counter()
    backend = None
    pipeline = None
    try:
        runtime_image, runtime_digest = _assert_runtime_provenance(plan)
        settings = _settings_for_plan(plan, candidate, parent_recipe)
        _seed_e046_strict(int(candidate["seed"]))
        backend = UpstreamDiffQRCoderBackend(settings)
        pipeline = backend._load()
        blueprint = _blueprint(candidate, parent_recipe)
        parent = _load_parent_artifact(plan_dir=plan_dir, candidate=candidate)
        original_vae_dtype = next(pipeline.vae.parameters()).dtype
        checkpointing_was_enabled = bool(
            getattr(pipeline.vae, "is_gradient_checkpointing", False)
        )
        enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
        disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)
        config = E039Config(
            gamma=recipe.gamma,
            lpips_weight=recipe.lpips_weight,
            crop_padding_px=QR_PADDING_PX,
            qr_version=QR_VERSION,
            qr_mask_pattern=parent_recipe.qr_mask_pattern,
            qr_module_size=QR_MODULE_SIZE,
            quiet_zone_mode="none",
            quiet_zone_minimum_luminance=0.78,
            functional_pattern_tone_factor=0.0,
            max_backtracks=recipe.max_backtracks,
            minimum_alpha=2**-12,
        )
        trajectory_recipe = TrajectoryRecipe(
            name=recipe.id,
            latent_radius_rms=recipe.latent_radius_rms,
            max_iterations=recipe.max_iterations,
            lpips_budget=recipe.lpips_budget,
            core_mae_budget=recipe.core_mae_budget,
            full_module_weight=recipe.full_module_weight,
        )
        checkpoints = []
        try:
            with _offload_diffusion_modules(pipeline) as offloaded:
                if not checkpointing_was_enabled and callable(enable_checkpointing):
                    enable_checkpointing()
                pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)
                atomic_write_json(
                    attempt / "runtime.json",
                    {
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0),
                        "offloaded_modules": list(offloaded),
                        "vae_original_dtype": str(original_vae_dtype),
                        "vae_effective_dtype": str(next(pipeline.vae.parameters()).dtype),
                        "scientific_contract_sha256": plan[
                            "runtime_scientific_contract"
                        ]["contract_sha256"],
                        "determinism": _determinism_runtime(),
                    },
                )
                checkpoints = _run_trajectory(
                    pipeline=pipeline,
                    parent=parent,
                    blueprint=blueprint,
                    recipe=trajectory_recipe,
                    config=config,
                    output_root=attempt / "trajectory",
                )
                iteration_zero_audit = _bind_e046_iteration_zero_to_parent(
                    trajectory_root=attempt / "trajectory",
                    recipe_id=recipe.id,
                    parent_image=parent.image,
                    checkpoints=checkpoints,
                )
        finally:
            pipeline.vae.to(dtype=original_vae_dtype)
            if not checkpointing_was_enabled and callable(disable_checkpointing):
                disable_checkpointing()

        expected = recipe.max_iterations + 1
        if len(checkpoints) != expected:
            raise RuntimeError(
                f"SR-MPGD produced {len(checkpoints)} checkpoints; expected {expected}"
            )
        parent_root = _parent_dir(plan_dir, candidate_id)
        metadata = {
            "experiment": EXPERIMENT,
            "kind": "srmpgd",
            "candidate_id": candidate_id,
            "parent_id": candidate_id,
            "trajectory_id": f"{candidate_id}__{srmpgd_recipe_id}",
            "srmpgd_recipe": asdict(recipe),
            "source_commit": source_commit,
            "runtime_image": runtime_image,
            "runtime_image_digest": runtime_digest,
            "scientific_contract_sha256": plan["runtime_scientific_contract"][
                "contract_sha256"
            ],
            "parent_stage2_image_sha256": _image_sha256(parent.image),
            "parent_stage2_latent_file_sha256": sha256_file(
                parent_root / "stage2-latent.safetensors"
            ),
            "checkpoint_count": len(checkpoints),
            "iteration_zero_audit": iteration_zero_audit,
            "observation_independence_class": "correlated_srmpgd_checkpoint",
            "elapsed_s": time.perf_counter() - started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "refinement-metadata.json", metadata)
        completion = {
            "experiment": EXPERIMENT,
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "srmpgd_recipe_id": srmpgd_recipe_id,
            "trajectory_id": metadata["trajectory_id"],
            "checkpoint_count": len(checkpoints),
            "generation_complete": True,
            "elapsed_s": metadata["elapsed_s"],
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "GENERATION_COMPLETE.json", completion)
        _promote_e046_attempt(
            attempt_dir=attempt,
            final_dir=final_dir,
            required_files=_refinement_generation_required(recipe),
            metadata={
                "experiment": EXPERIMENT,
                "kind": "srmpgd",
                "candidate_id": candidate_id,
                "srmpgd_recipe_id": srmpgd_recipe_id,
                "source_commit": source_commit,
            },
        )
        return completion
    except BaseException as exc:
        if _terminal_srl_failure(exc):
            return _promote_terminal_refinement_failure(
                attempt=attempt,
                final_dir=final_dir,
                plan_dir=plan_dir,
                candidate_id=candidate_id,
                recipe_id=srmpgd_recipe_id,
                error=exc,
            )
        _record_failure(
            plan_dir=plan_dir,
            task_kind="refinement",
            task_id=f"{candidate_id}__{srmpgd_recipe_id}",
            error=exc,
        )
        raise
    finally:
        if backend is not None:
            backend._pipeline = None
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _score_refinement_rows(
    *,
    plan_dir: Path,
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    parent_recipe: ParentRecipe,
    recipe: SRMPGDRecipe,
    root: Path,
    images: Mapping[str, Image.Image],
    traces: Mapping[str, Mapping[str, Any]],
    quality: Mapping[str, Mapping[str, Any]],
    qr: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from .quality import image_change_metrics, image_quality_metrics

    parent_root = _parent_dir(plan_dir, str(candidate["id"]))
    parent_image = Image.open(parent_root / "images/stage2-raw.png").convert("RGB")
    parent_quality = _load_json(parent_root / "scoring/quality-scores.json")[
        "stage2_raw"
    ]
    parent_metadata = _load_json(parent_root / "parent-metadata.json")
    refinement_metadata = _load_json(root / "refinement-metadata.json")
    blueprint = _blueprint(candidate, parent_recipe)
    rows: list[dict[str, Any]] = []
    for key, image in images.items():
        iteration = int(key.removeprefix("i"))
        step = traces[key]
        latent_path = (
            root
            / "trajectory"
            / recipe.id
            / "latents"
            / f"iteration-{iteration:03d}.safetensors"
        )
        row = {
            "schema": DATASET_SCHEMA,
            "experiment": EXPERIMENT,
            "plan_id": plan["plan_id"],
            "source_commit": refinement_metadata.get("source_commit"),
            "runtime_image": refinement_metadata.get("runtime_image"),
            "runtime_image_digest": refinement_metadata.get("runtime_image_digest"),
            "scientific_contract_sha256": refinement_metadata.get(
                "scientific_contract_sha256"
            ),
            "source_kind": "srmpgd",
            "candidate_id": candidate["id"],
            "parent_id": candidate["id"],
            "trajectory_id": f"{candidate['id']}__{recipe.id}",
            "observation_independence_class": "correlated_srmpgd_checkpoint",
            "stage": "srmpgd",
            "variant": f"i{iteration:03d}_raw",
            "raw": True,
            "quiet_zone_variant": "raw",
            "primary_training_observation": False,
            "diagnostic_variant": False,
            "stage1_final_eligible": False,
            "scene_qz_final_eligible": False,
            "image_path": str(
                root
                / "trajectory"
                / recipe.id
                / "images"
                / f"iteration-{iteration:03d}.png"
            ),
            "latent_path": str(latent_path),
            "image_sha256": _image_sha256(image),
            "image_file_sha256": sha256_file(
                root
                / "trajectory"
                / recipe.id
                / "images"
                / f"iteration-{iteration:03d}.png"
            ),
            "latent_sha256": step.get("latent_sha256"),
            "latent_file_sha256": sha256_file(latent_path),
            "parent_stage2_image_sha256": _image_sha256(parent_image),
            "iteration_zero_matches_parent_raster": (
                _image_sha256(image) == _image_sha256(parent_image)
                if iteration == 0
                else None
            ),
            "iteration_zero_scores_reused_from_parent": iteration == 0,
            "payload_sha256": hashlib.sha256(
                str(candidate["payload"]).encode("utf-8")
            ).hexdigest(),
            "qr_version": QR_VERSION,
            "error_correction": parent_recipe.error_correction,
            "qr_mask_pattern": parent_recipe.qr_mask_pattern,
            "qr_module_size": QR_MODULE_SIZE,
            "qr_padding_px": QR_PADDING_PX,
            "iteration": iteration,
            "srmpgd_recipe_id": recipe.id,
            "gamma": recipe.gamma,
            "latent_radius_rms": recipe.latent_radius_rms,
            "srmpgd_max_iterations": recipe.max_iterations,
            "srmpgd_lpips_weight": recipe.lpips_weight,
            "srmpgd_lpips_budget": recipe.lpips_budget,
            "srmpgd_core_mae_budget": recipe.core_mae_budget,
            "srmpgd_full_module_weight": recipe.full_module_weight,
            "srmpgd_max_backtracks": recipe.max_backtracks,
            "stage1_quality_scored": False,
            "stage2_effective_steps": int(
                round(
                    _finite_number(
                        dict(parent_metadata.get("stage2_diagnostics") or {}).get(
                            "diffqrcoder_stage2_effective_steps"
                        ),
                        parent_recipe.stage2_steps,
                    )
                )
            ),
            "lpips": float(step.get("lpips_loss") or 0.0),
            "latent_delta_rms": step.get("latent_delta_rms"),
            "accepted_alpha": step.get("accepted_alpha"),
            "accepted_step_rms": step.get("accepted_step_rms"),
            "raw_step_rms": step.get("raw_step_rms"),
            "projected_step_rms": step.get("projected_step_rms"),
            "rejected_trial_count": step.get("rejected_trial_count"),
            "acceptance_reason": step.get("acceptance_reason"),
            **_candidate_features(candidate),
            **_recipe_features(parent_recipe),
            **_qr_fields(qr[key]),
            **_module_fields(image, blueprint),
            **image_change_metrics(image, parent_image),
            **_stage_core_change(image, parent_image),
            **image_quality_metrics(image),
            **dict(quality[key]),
        }
        guard = _refinement_visual_guard(
            row=row,
            parent_quality=parent_quality,
            recipe=recipe,
            qz_guard=None,
        )
        row["visual_guard_pass"] = bool(guard["passed"])
        row["visual_guard_checks"] = guard["checks"]
        row["eligible_final"] = bool(row["visual_guard_pass"])
        minimum = int(plan["validity_policy"]["final_minimum_exact_presets"])
        row["delivery_eligible"] = bool(
            row["visual_guard_pass"]
            and row["raw"]
            and row["wechat_original_exact"]
            and row["wechat_exact_presets"] >= minimum
        )
        rows.append(row)
    return rows


def score_refinement(
    *,
    output_root: Path,
    plan_id: str,
    candidate_id: str,
    srmpgd_recipe_id: str,
    qr_scorer: Any | None = None,
    quality_scorer: Any | None = None,
    _plan_context: tuple[Path, Mapping[str, Any]] | None = None,
    _candidate_context: Mapping[str, Any] | None = None,
    _parent_recipe_context: ParentRecipe | None = None,
    _srmpgd_recipe_context: SRMPGDRecipe | None = None,
) -> dict[str, Any]:
    plan_dir, plan = _plan_context or load_plan(output_root, plan_id)
    candidate = _candidate_context or _candidate(plan, candidate_id)
    parent_recipe = _parent_recipe_context or _parent_recipe(plan, candidate)
    recipe = _srmpgd_recipe_context or _srmpgd_recipe_map(plan)[srmpgd_recipe_id]
    root = _refinement_dir(plan_dir, candidate_id, srmpgd_recipe_id)
    if _promotion_valid(
        root, ("task.json", "TERMINAL_FAILURE.json"), deep=True
    ):
        return {
            "candidate_id": candidate_id,
            "srmpgd_recipe_id": srmpgd_recipe_id,
            "scoring_skipped": True,
            "terminal_failure": True,
        }
    if not _promotion_valid(
        root, _refinement_generation_required(recipe), deep=True
    ):
        raise FileNotFoundError("refinement generation is incomplete")
    parent_root = _parent_dir(plan_dir, candidate_id)
    if not _promotion_valid(
        parent_root, PARENT_GENERATION_REQUIRED, deep=True
    ):
        raise FileNotFoundError("parent generation is incomplete or corrupt")
    if not _promotion_valid(
        parent_root / "scoring", SCORING_REQUIRED, deep=True
    ):
        raise FileNotFoundError("parent scoring is incomplete or corrupt")
    root_marker = root / "SCORING_COMPLETE.json"
    scoring_dir = root / "scoring"
    nested_marker = scoring_dir / "SCORING_COMPLETE.json"
    scoring_valid = _promotion_valid(
        scoring_dir, SCORING_REQUIRED, deep=True
    )
    if root_marker.is_file() and scoring_valid:
        return _load_json(root_marker)
    if nested_marker.is_file() and scoring_valid:
        completion = _load_json(nested_marker)
        atomic_write_json(root_marker, completion)
        return completion
    if scoring_dir.exists():
        raise FileExistsError(f"incomplete refinement scoring: {scoring_dir}")

    attempt = (
        plan_dir
        / "attempts/scoring-refinements"
        / f"{candidate_id}-{srmpgd_recipe_id}-{uuid.uuid4().hex[:8]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        trajectory = root / "trajectory" / recipe.id
        trace_rows = _load_json(trajectory / "trace.json")
        images: dict[str, Image.Image] = {}
        traces: dict[str, dict[str, Any]] = {}
        for step in trace_rows:
            iteration = int(step["iteration"])
            key = f"i{iteration:03d}"
            images[key] = Image.open(
                trajectory / "images" / f"iteration-{iteration:03d}.png"
            ).convert("RGB")
            traces[key] = dict(step)
        settings = _settings_for_plan(plan, candidate, parent_recipe)
        refinement_images = {
            key: image for key, image in images.items() if key != "i000"
        }
        refinement_quality, provenance = _quality_scores(
            images=refinement_images,
            prompt=str(candidate["prompt"]),
            settings=settings,
            scorer=quality_scorer,
        )
        provenance = {
            **provenance,
            "scientific_contract_sha256": plan["runtime_scientific_contract"][
                "contract_sha256"
            ],
        }
        parent_scoring = _parent_dir(plan_dir, candidate_id) / "scoring"
        quality = {
            "i000": _load_json(parent_scoring / "quality-scores.json")[
                "stage2_raw"
            ],
            **refinement_quality,
        }
        qr = {
            "i000": _load_json(parent_scoring / "qr-verify-evidence.json"),
            **{
            key: _score_qr_cached(
                plan_dir=plan_dir,
                image=image,
                payload=str(candidate["payload"]),
                scorer=qr_scorer,
                plan=plan,
            )
            for key, image in refinement_images.items()
            },
        }
        rows = _score_refinement_rows(
            plan_dir=plan_dir,
            plan=plan,
            candidate=candidate,
            parent_recipe=parent_recipe,
            recipe=recipe,
            root=root,
            images=images,
            traces=traces,
            quality=quality,
            qr=qr,
        )
        atomic_write_json(attempt / "comparison.json", rows)
        _write_csv(attempt / "comparison.csv", _flatten_rows(rows))
        atomic_write_json(attempt / "qr-verify-evidence.json", qr)
        atomic_write_json(attempt / "quality-scores.json", quality)
        atomic_write_json(attempt / "quality-provenance.json", provenance)
        completion = {
            "experiment": EXPERIMENT,
            "plan_id": plan_id,
            "candidate_id": candidate_id,
            "srmpgd_recipe_id": srmpgd_recipe_id,
            "trajectory_id": f"{candidate_id}__{srmpgd_recipe_id}",
            "row_count": len(rows),
            "iteration_zero_scores_reused_from_parent": True,
            "scoring_complete": True,
            "elapsed_s": time.perf_counter() - started,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(attempt / "SCORING_COMPLETE.json", completion)
        _promote_e046_attempt(
            attempt_dir=attempt,
            final_dir=scoring_dir,
            required_files=SCORING_REQUIRED,
            metadata={
                "experiment": EXPERIMENT,
                "kind": "score-refinement",
                "candidate_id": candidate_id,
                "srmpgd_recipe_id": srmpgd_recipe_id,
            },
        )
        atomic_write_json(root_marker, completion)
        return completion
    except BaseException as exc:
        _record_failure(
            plan_dir=plan_dir,
            task_kind="score-refinement",
            task_id=f"{candidate_id}__{srmpgd_recipe_id}",
            error=exc,
        )
        raise


def score_all_refinements(*, output_root: Path, plan_id: str) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    tasks = refinement_tasks(plan_dir, plan)
    from .quality_scoring import quality_scorer_from_settings

    pending = [
        (candidate_id, recipe_id)
        for candidate_id, recipe_id in tasks
        if not _promotion_valid(
            _refinement_dir(plan_dir, candidate_id, recipe_id) / "scoring",
            SCORING_REQUIRED,
        )
        and not (
            _refinement_dir(plan_dir, candidate_id, recipe_id)
            / "TERMINAL_FAILURE.json"
        ).is_file()
    ]
    qr_scorer = _new_qr_scorer(plan_dir, plan) if pending else None
    quality_scorer = None
    if pending:
        first_candidate = _candidate(plan, pending[0][0])
        quality_scorer = quality_scorer_from_settings(
            _settings_for_plan(
                plan,
                first_candidate,
                _parent_recipe(plan, first_candidate),
            ),
            device="cpu",
        )
    try:
        candidate_map = {
            str(item["id"]): item for item in plan["candidates"]
        }
        parent_recipe_map = _recipe_map(plan)
        srmpgd_recipe_map = _srmpgd_recipe_map(plan)
        results = [
            score_refinement(
                output_root=output_root,
                plan_id=plan_id,
                candidate_id=candidate_id,
                srmpgd_recipe_id=recipe_id,
                qr_scorer=qr_scorer,
                quality_scorer=quality_scorer,
                _plan_context=(plan_dir, plan),
                _candidate_context=candidate_map[candidate_id],
                _parent_recipe_context=parent_recipe_map[
                    str(candidate_map[candidate_id]["parent_recipe_id"])
                ],
                _srmpgd_recipe_context=srmpgd_recipe_map[recipe_id],
            )
            for candidate_id, recipe_id in tasks
        ]
    finally:
        if qr_scorer is not None:
            qr_scorer.close()
        del quality_scorer
        gc.collect()
    completion = {
        "experiment": EXPERIMENT,
        "plan_id": plan_id,
        "task_count": len(tasks),
        "scored_count": sum(bool(row.get("scoring_complete")) for row in results),
        "terminal_failure_count": sum(bool(row.get("terminal_failure")) for row in results),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "REFINEMENT_SCORING_COMPLETE.json", completion)
    return completion


def _refinement_rows(plan_dir: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recipes = _srmpgd_recipe_map(plan)
    for candidate_id, recipe_id in refinement_tasks(plan_dir, plan):
        refinement_root = _refinement_dir(plan_dir, candidate_id, recipe_id)
        scoring_root = refinement_root / "scoring"
        path = scoring_root / "comparison.json"
        if _promotion_valid(
            refinement_root,
            _refinement_generation_required(recipes[recipe_id]),
            deep=True,
        ) and _promotion_valid(scoring_root, SCORING_REQUIRED, deep=True):
            rows.extend(_load_json(path))
    return rows


def _histogram(values: Iterable[int]) -> dict[str, int]:
    values = list(values)
    return {
        f"{lower:02d}-{upper:02d}": sum(lower <= value <= upper for value in values)
        for lower, upper in ((0, 5), (6, 15), (16, 25), (26, 33), (34, 37))
    }


def _numeric_distribution(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [
        _finite_number(row.get(field), float("nan")) for row in rows
    ]
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "minimum": values[0],
        "q25": quantile(0.25),
        "median": quantile(0.50),
        "q75": quantile(0.75),
        "maximum": values[-1],
        "mean": sum(values) / len(values),
        "distinct_count": len(set(values)),
    }


def _parameter_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe all 13 designed factors and their actually effective values."""
    factors: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        designed_field = dimension.name
        effective_field = {
            "stage2_strength": "stage2_strength_effective",
            "mask_slot": "qr_mask_pattern",
        }.get(designed_field, designed_field)
        designed_values = [
            row.get(designed_field)
            for row in rows
            if row.get(designed_field) is not None
        ]
        effective_values = [
            row.get(effective_field)
            for row in rows
            if row.get(effective_field) is not None
        ]
        factor: dict[str, Any] = {
            "stage": dimension.stage,
            "kind": dimension.kind,
            "designed_field": designed_field,
            "effective_field": effective_field,
            "row_count": len(rows),
            "designed_applicable_count": len(designed_values),
            "effective_applicable_count": len(effective_values),
            "effective_inactive_count": len(rows) - len(effective_values),
            "designed_distinct_count": len(
                {json.dumps(value, sort_keys=True) for value in designed_values}
            ),
            "effective_distinct_count": len(
                {json.dumps(value, sort_keys=True) for value in effective_values}
            ),
        }
        if dimension.kind == "categorical":
            factor["designed_counts"] = {
                str(value): sum(item == value for item in designed_values)
                for value in sorted(set(designed_values), key=str)
            }
            factor["effective_counts"] = {
                str(value): sum(item == value for item in effective_values)
                for value in sorted(set(effective_values), key=str)
            }
        else:
            factor["designed_distribution"] = _numeric_distribution(
                rows, designed_field
            )
            factor["effective_distribution"] = _numeric_distribution(
                rows, effective_field
            )
        factors[designed_field] = factor
    if len(factors) != DOE_PARAMETER_COUNT:
        raise RuntimeError(
            f"E046 parameter coverage has {len(factors)} factors, "
            f"expected {DOE_PARAMETER_COUNT}"
        )
    return {
        "doe_parameter_count": DOE_PARAMETER_COUNT,
        "covered_factor_count": len(factors),
        "all_factors_present": all(
            factor["designed_applicable_count"] == len(rows)
            for factor in factors.values()
        ),
        "factors": factors,
    }


def _best_by_prompt(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(dict(row))
    for prompt_id in sorted(grouped):
        eligible = [row for row in grouped[prompt_id] if row.get("delivery_eligible")]
        if not eligible:
            continue
        winner = sorted(
            eligible,
            key=lambda row: (
                -int(row.get("wechat_exact_presets") or 0),
                -_finite_number(row.get("multiobjective_prompt_score"), -1e9),
                -_finite_number(row.get("clip_score"), -1e9),
                str(row.get("candidate_id")),
                int(row.get("iteration") or 0),
            ),
        )[0]
        winners.append(winner)
    return winners


def _dataset_summary(
    *,
    plan: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    srmpgd: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(parents) + list(srmpgd)
    return {
        "schema": "e046-large-data-summary-v1",
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "profile": plan["profile"],
        "parent_observation_count": len(parents),
        "independent_parent_count": sum(
            row.get("observation_independence_class") == "independent_parent"
            for row in parents
        ),
        "srmpgd_observation_count": len(srmpgd),
        "srmpgd_trajectory_count": len(
            {row.get("trajectory_id") for row in srmpgd if row.get("trajectory_id")}
        ),
        "advisor_observation_count": len(rows),
        "prompt_count": len({row["prompt_id"] for row in parents}),
        "family_count": len({row["prompt_family"] for row in parents}),
        "hard_negative_count": sum(
            int(row.get("wechat_exact_presets") or 0) < 34 for row in parents
        ),
        "delivery_eligible_count": sum(bool(row.get("delivery_eligible")) for row in rows),
        "wechat_bucket_counts": _histogram(
            int(row.get("wechat_exact_presets") or 0) for row in parents
        ),
        "targets_preserved_separately": [
            "wechat_exact_presets",
            "wechat_exact_rate",
            "wechat_original_exact",
            "wechat_instability",
            "clip_score",
            "hpsv2_1",
            "clip_aesthetic",
            "module_error_rate",
            "visual_guard_pass",
            "multiobjective_prompt_score",
        ],
        "training_ready": False,
        "final_advisor_trained": False,
    }


GROUP_KFOLD_COUNT = 5
SPLIT_GROUP_KEYS: tuple[str, ...] = (
    "prompt_id",
    "payload_sha256",
    "generation_group_id",
    "trajectory_id",
)
ADVISOR_FEATURE_COLUMNS: tuple[str, ...] = (
    "prompt",
    "prompt_family",
    "spatial_frequency",
    "symmetry",
    "brightness",
    "contrast",
    "colorfulness",
    "negative_space",
    "texture_density",
    "geometry_type",
    "composition_type",
    "payload",
    "payload_length",
    "qr_version",
    "error_correction",
    "qr_mask_pattern",
    "seed",
    "stage1_steps",
    "stage1_guidance_scale",
    "stage1_controlnet_scale",
    "control_guidance_start",
    "control_guidance_end",
    "stage2_initialization",
    "stage2_strength",
    "stage2_strength_effective",
    "stage2_steps",
    "stage2_controlnet_scale",
    "stage2_qr_weight",
    "stage2_perceptual_weight",
    "srmpgd_recipe_id",
    "gamma",
    "latent_radius_rms",
    "srmpgd_max_iterations",
    "srmpgd_lpips_weight",
    "srmpgd_lpips_budget",
    "srmpgd_core_mae_budget",
    "srmpgd_full_module_weight",
    "srmpgd_max_backtracks",
    "iteration",
)
ADVISOR_TARGET_COLUMNS: tuple[str, ...] = (
    "wechat_exact_presets",
    "wechat_exact_rate",
    "wechat_original_exact",
    "wechat_instability",
    "delivery_eligible",
    "clip_score",
    "hpsv2_1",
    "clip_aesthetic",
    "module_error_rate",
    "visual_guard_pass",
    "multiobjective_prompt_score",
)


def prompt_group_fold(design_row: int, fold_count: int = GROUP_KFOLD_COUNT) -> int:
    """Deterministic family-stratified fold: prompts of one family spread over folds."""
    return (int(design_row) - 1) % int(fold_count)


def build_group_splits(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[PromptSpec],
    *,
    fold_count: int = GROUP_KFOLD_COUNT,
) -> dict[str, Any]:
    """Assign every observation to GroupKFold-by-prompt and leave-family-out folds.

    Folds are defined at the prompt level only. Every row inherits the fold of
    its ``prompt_id``; SR-MPGD checkpoints therefore always share the fold of
    their parent and of the whole trajectory. A random row split is never
    produced here and :func:`validate_group_split` rejects one.
    """
    by_prompt = {prompt.prompt_id: prompt for prompt in prompts}
    used_prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    missing = [prompt_id for prompt_id in used_prompt_ids if prompt_id not in by_prompt]
    if missing:
        raise KeyError(f"observations reference unknown prompts: {missing[:5]}")
    kfold_prompt_folds = {
        prompt_id: prompt_group_fold(by_prompt[prompt_id].design_row, fold_count)
        for prompt_id in used_prompt_ids
    }
    family_prompt_folds = {
        prompt_id: by_prompt[prompt_id].family for prompt_id in used_prompt_ids
    }
    row_assignments = []
    for row in rows:
        prompt_id = str(row["prompt_id"])
        row_assignments.append(
            {
                "candidate_id": row.get("candidate_id"),
                "image_sha256": row.get("image_sha256"),
                "prompt_id": prompt_id,
                "payload_sha256": row.get("payload_sha256"),
                "generation_group_id": row.get("generation_group_id"),
                "trajectory_id": row.get("trajectory_id"),
                "observation_independence_class": row.get(
                    "observation_independence_class"
                ),
                "group_kfold_by_prompt": kfold_prompt_folds[prompt_id],
                "leave_family_out": family_prompt_folds[prompt_id],
            }
        )

    def fold_sizes(field: str) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for item in row_assignments:
            key = str(item[field])
            sizes[key] = sizes.get(key, 0) + 1
        return dict(sorted(sizes.items()))

    document = {
        "schema": "e046-large-group-splits-v1",
        "random_row_split_allowed": False,
        "group_keys": list(SPLIT_GROUP_KEYS),
        "fold_unit": "prompt_id",
        "group_kfold_by_prompt": {
            "fold_count": fold_count,
            "assignment_rule": "fold = (design_row - 1) mod fold_count (family-stratified)",
            "prompt_folds": kfold_prompt_folds,
            "row_counts_per_fold": fold_sizes("group_kfold_by_prompt"),
            "prompt_counts_per_fold": {
                str(fold): sum(value == fold for value in kfold_prompt_folds.values())
                for fold in range(fold_count)
            },
        },
        "leave_family_out": {
            "fold_count": len(set(family_prompt_folds.values())),
            "assignment_rule": "fold = prompt_family",
            "prompt_folds": family_prompt_folds,
            "row_counts_per_fold": fold_sizes("leave_family_out"),
        },
        "row_assignments": row_assignments,
    }
    validate_group_split(row_assignments, "group_kfold_by_prompt")
    validate_group_split(row_assignments, "leave_family_out")
    return document


def validate_group_split(
    assignments: Sequence[Mapping[str, Any]],
    field: str,
) -> None:
    """Fail when any group key (prompt, payload, generation group, trajectory)
    is split across folds, i.e. when a row-level split leaked into the dataset."""
    for key in SPLIT_GROUP_KEYS:
        folds_by_group: dict[str, set[Any]] = {}
        for item in assignments:
            group = item.get(key)
            if group is None:
                continue
            folds_by_group.setdefault(str(group), set()).add(item[field])
        leaking = {
            group: sorted(map(str, folds))
            for group, folds in folds_by_group.items()
            if len(folds) > 1
        }
        if leaking:
            sample = dict(list(leaking.items())[:3])
            raise ValueError(
                f"{field}: {len(leaking)} {key} groups span several folds, e.g. {sample}"
            )


def advisor_training_contract(
    plan: Mapping[str, Any],
    *,
    splits: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Interfaces for the future E047 advisor. Nothing here claims a trained model."""
    return {
        "schema": "e047-advisor-training-contract-from-e046-large-v1",
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "profile": plan["profile"],
        "source_commit": plan["source_commit"],
        "runtime_image_digest": plan["runtime_image_digest"],
        "dataset_hashes": dict(dataset_hashes),
        "primary_training_rows": {
            "file": "parent-observations.jsonl",
            "filter": "observation_independence_class == 'independent_parent'",
        },
        "correlated_rows": {
            "file": "srmpgd-observations.jsonl",
            "filter": "observation_independence_class == 'correlated_srmpgd_checkpoint'",
            "rule": "never counted as independent experiments; weight or model per trajectory",
        },
        "feature_columns": list(ADVISOR_FEATURE_COLUMNS),
        "target_columns": list(ADVISOR_TARGET_COLUMNS),
        "separate_heads_required": [
            "P(delivery_eligible)",
            "E[wechat_exact_presets]",
            "E[clip_score]",
            "E[hpsv2_1]",
            "E[clip_aesthetic]",
            "E[module_error_rate]",
        ],
        "multiobjective_score_is_derived_not_primary": True,
        "group_keys": list(SPLIT_GROUP_KEYS),
        "split_file": "splits.json",
        "supported_evaluations": {
            "group_kfold_by_prompt": splits["group_kfold_by_prompt"]["fold_count"],
            "leave_family_out": splits["leave_family_out"]["fold_count"],
        },
        "row_random_split_forbidden": True,
        "same_trajectory_cross_split_forbidden": True,
        "same_prompt_cross_split_forbidden": True,
        "generalization_claim_requires_held_out_prompts": True,
        "active_learning_hooks": [
            "predict on unexplored (prompt, config) pairs",
            "rank by predictive uncertainty and doe_isolation_score",
            "emit a new E046-large plan with the requested candidates",
        ],
        "sufficient_for_final_advisor": plan["profile"] == "full",
        "final_advisor_trained": False,
        "automatic_training_authorized": False,
    }


def aggregate(*, output_root: Path, plan_id: str) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    if not (plan_dir / "PARENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("parent scoring is incomplete")
    if not (plan_dir / "selected-refinements.json").is_file():
        raise FileNotFoundError("Phase-B selection is incomplete")
    if not (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").is_file():
        raise FileNotFoundError("refinement scoring is incomplete")
    parents = _parent_rows(plan_dir, plan)
    srmpgd = _refinement_rows(plan_dir, plan)
    annotated = _annotate_prompt_objectives([*parents, *srmpgd], plan)
    parent_ids = {str(row["candidate_id"]) for row in parents}
    parent_rows = [
        row
        for row in annotated
        if str(row["candidate_id"]) in parent_ids
        and row["source_kind"] == "parent"
    ]
    srmpgd_rows = [row for row in annotated if row["source_kind"] == "srmpgd"]
    dataset_dir = plan_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    exports = {
        "parent-observations": parent_rows,
        "srmpgd-observations": srmpgd_rows,
        "advisor-observations": annotated,
    }
    for name, rows in exports.items():
        atomic_write_text(
            dataset_dir / f"{name}.jsonl",
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
        )
        _write_csv(dataset_dir / f"{name}.csv", _flatten_rows(rows))

    summary = _dataset_summary(plan=plan, parents=parent_rows, srmpgd=srmpgd_rows)
    atomic_write_json(dataset_dir / "data-summary.json", summary)
    selected_prompt_ids = {str(item["prompt_id"]) for item in plan["candidates"]}
    prompts = tuple(
        prompt for prompt in load_prompts() if prompt.prompt_id in selected_prompt_ids
    )
    atomic_write_json(
        dataset_dir / "prompt-coverage.json",
        {
            "profile": plan["profile"],
            "prompt_count": summary["prompt_count"],
            "family_count": summary["family_count"],
            "catalog_coverage": tag_coverage(prompts),
        },
    )
    parameter_coverage = _parameter_coverage(parent_rows)
    parameter_coverage.update(
        {
            "prompt_structural_feature_count": PROMPT_STRUCTURAL_FEATURE_COUNT,
            "actual_mask_counts": {
                str(mask): sum(
                    int(row["qr_mask_pattern"]) == mask for row in parent_rows
                )
                for mask in range(8)
            },
            "ecc_counts": {
                ecc: sum(row["error_correction"] == ecc for row in parent_rows)
                for ecc in ("M", "Q")
            },
            "config_counts": {
                config_id: sum(row["config_id"] == config_id for row in parent_rows)
                for config_id in sorted(
                    {str(row["config_id"]) for row in parent_rows}
                )
            },
        }
    )
    atomic_write_json(
        dataset_dir / "parameter-coverage.json",
        parameter_coverage,
    )
    atomic_write_json(
        dataset_dir / "score-distributions.json",
        {
            field: _numeric_distribution(annotated, field)
            for field in (
                "wechat_exact_presets",
                "clip_score",
                "hpsv2_1",
                "clip_aesthetic",
                "module_error_rate",
                "multiobjective_prompt_score",
            )
        },
    )
    atomic_write_json(
        dataset_dir / "split-contract.json",
        {
            "random_row_split_allowed": False,
            "required_group_keys": [
                "prompt_id",
                "payload",
                "generation_group_id",
                "trajectory_id",
            ],
            "group_kfold_primary": "prompt_id",
            "held_out_family_evaluation": True,
            "same_trajectory_cross_split_allowed": False,
        },
    )
    splits = build_group_splits(annotated, prompts)
    atomic_write_json(dataset_dir / "splits.json", splits)
    dataset_hashes = {
        name: sha256_file(dataset_dir / name)
        for name in sorted(
            f"{stem}.{suffix}" for stem in exports for suffix in ("jsonl", "csv")
        )
    }
    atomic_write_json(
        dataset_dir / "advisor-training-contract.json",
        advisor_training_contract(
            plan,
            splits=splits,
            dataset_hashes=dataset_hashes,
        ),
    )
    best = _best_by_prompt(annotated)
    atomic_write_json(dataset_dir / "best-valid-by-prompt.json", best)
    _write_csv(dataset_dir / "best-valid-by-prompt.csv", _flatten_rows(best))

    terminal_failures = [
        _load_json(path)
        for path in plan_dir.glob("refinements/*/*/TERMINAL_FAILURE.json")
    ]
    failure_audit = _technical_failure_audit(plan_dir, plan, deep=True)
    verdict = {
        **summary,
        "complete": True,
        "best_valid_prompt_count": len(best),
        "terminal_scientific_failure_count": len(terminal_failures),
        "technical_failure_history_count": len(failure_audit["history"]),
        "technical_failure_count": len(failure_audit["unresolved"]),
        "manifest_valid": None,
        "production_ready": False,
        "automatic_full_launch_authorized": False,
        "next_action": (
            "VERIFY_SMOKE_BEFORE_PILOT"
            if plan["profile"] == "smoke"
            else "REVIEW_DATASET_COVERAGE"
        ),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "verdict.json", verdict)

    manifest = [
        item
        for item in build_artifact_manifest(plan_dir)
        if item["path"] not in {"artifact-manifest.json", "COMPLETE.json"}
        and not str(item["path"]).startswith("cache/qr-verify/")
    ]
    atomic_write_json(plan_dir / "artifact-manifest.json", manifest)
    complete = {
        **verdict,
        "artifact_manifest_sha256": sha256_file(plan_dir / "artifact-manifest.json"),
    }
    atomic_write_json(plan_dir / "COMPLETE.json", complete)
    atomic_write_json(
        output_root / "LATEST.json",
        {
            "schema": LATEST_SCHEMA,
            "plan_id": plan_id,
            "plan_dir": str(plan_dir),
            "profile": plan["profile"],
            "source_commit": plan["source_commit"],
            "status": "complete",
            "complete_path": str(plan_dir / "COMPLETE.json"),
            "updated_at_utc": utc_now(),
        },
    )
    return complete


def _directory_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _elapsed_values(paths: Iterable[Path]) -> list[float]:
    values: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        value = _finite_number(_load_json(path).get("elapsed_s"), float("nan"))
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return values


def status(*, output_root: Path, plan_id: str | None = None) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    parent_total = len(plan["candidates"])
    parent_generated = sum(
        _promotion_valid(
            _parent_dir(plan_dir, str(item["id"])),
            PARENT_GENERATION_REQUIRED,
        )
        for item in plan["candidates"]
    )
    parent_scored = sum(
        _promotion_valid(
            _parent_dir(plan_dir, str(item["id"])) / "scoring",
            SCORING_REQUIRED,
        )
        for item in plan["candidates"]
    )
    tasks = refinement_tasks(plan_dir, plan)
    expected_refinements = int(plan["expected_refinement_count_maximum"])
    refinement_total = len(tasks) if tasks else expected_refinements
    refinement_generated = 0
    refinement_scored = 0
    terminal = 0
    recipes = _srmpgd_recipe_map(plan)
    for candidate_id, recipe_id in tasks:
        root = _refinement_dir(plan_dir, candidate_id, recipe_id)
        terminal_valid = _promotion_valid(
            root,
            ("task.json", "TERMINAL_FAILURE.json"),
        )
        terminal += terminal_valid
        if not terminal_valid:
            refinement_generated += _promotion_valid(
                root,
                _refinement_generation_required(recipes[recipe_id]),
            )
            refinement_scored += _promotion_valid(
                root / "scoring",
                SCORING_REQUIRED,
            )

    # A terminal scientific mismatch resolves both generation and scoring for
    # that trajectory: it is deliberately unusable and must not keep ETA open.
    completed_units = (
        parent_generated
        + parent_scored
        + refinement_generated
        + refinement_scored
        + terminal * 2
    )
    total_units = parent_total * 2 + refinement_total * 2
    remaining_units = max(0, total_units - completed_units)
    elapsed = _elapsed_values(plan_dir.glob("parents/*/GENERATION_COMPLETE.json"))
    elapsed += _elapsed_values(plan_dir.glob("parents/*/SCORING_COMPLETE.json"))
    elapsed += _elapsed_values(plan_dir.glob("refinements/*/*/GENERATION_COMPLETE.json"))
    elapsed += _elapsed_values(plan_dir.glob("refinements/*/*/SCORING_COMPLETE.json"))
    average = sum(elapsed) / len(elapsed) if elapsed else None
    eta_seconds = average * remaining_units if average is not None else None
    failure_audit = _technical_failure_audit(plan_dir, plan)
    return {
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "profile": plan["profile"],
        "plan_dir": str(plan_dir),
        "parents_planned": parent_total,
        "parents_generated": parent_generated,
        "parents_scored": parent_scored,
        "selection_complete": (plan_dir / "selected-refinements.json").is_file(),
        "srmpgd_planned": refinement_total,
        "srmpgd_generated": refinement_generated,
        "srmpgd_scored": refinement_scored,
        "terminal_scientific_failures": terminal,
        "technical_failure_history_files": len(failure_audit["history"]),
        "technical_failure_files": len(failure_audit["unresolved"]),
        "remaining_tasks": remaining_units,
        "percent_completion": round(100.0 * completed_units / total_units, 2)
        if total_units
        else 100.0,
        "observed_average_seconds_per_completed_task": round(average, 2)
        if average is not None
        else None,
        "eta_hours_from_observed_tasks": round(eta_seconds / 3600.0, 2)
        if eta_seconds is not None
        else None,
        "disk_usage_gib": round(_directory_size(plan_dir) / 1024**3, 3),
        "pause_after_current": (plan_dir / "control/PAUSE_AFTER_CURRENT.json").is_file(),
        "aggregate_complete": (plan_dir / "COMPLETE.json").is_file(),
    }


def request_pause(*, output_root: Path, plan_id: str | None = None) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    payload = {
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "pause_after_current": True,
        "requested_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "control/PAUSE_AFTER_CURRENT.json", payload)
    return payload


def clear_pause(*, output_root: Path, plan_id: str | None = None) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    path = plan_dir / "control/PAUSE_AFTER_CURRENT.json"
    if path.exists():
        path.unlink()
    payload = {
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "pause_after_current": False,
        "cleared_at_utc": utc_now(),
    }
    atomic_write_json(plan_dir / "control/LAST_RESUME.json", payload)
    return payload


def list_parent_ids(
    *, output_root: Path, plan_id: str, pending_only: bool
) -> list[str]:
    plan_dir, plan = load_plan(output_root, plan_id)
    result = []
    for item in plan["candidates"]:
        candidate_id = str(item["id"])
        if pending_only and _promotion_valid(
            _parent_dir(plan_dir, candidate_id),
            PARENT_GENERATION_REQUIRED,
        ):
            continue
        result.append(candidate_id)
    return result


def list_refinement_ids(
    *, output_root: Path, plan_id: str, pending_only: bool
) -> list[tuple[str, str]]:
    plan_dir, plan = load_plan(output_root, plan_id)
    result = []
    recipes = _srmpgd_recipe_map(plan)
    for candidate_id, recipe_id in refinement_tasks(plan_dir, plan):
        root = _refinement_dir(plan_dir, candidate_id, recipe_id)
        if pending_only and (
            _promotion_valid(
                root,
                _refinement_generation_required(recipes[recipe_id]),
            )
            or _promotion_valid(root, ("task.json", "TERMINAL_FAILURE.json"))
        ):
            continue
        task_id = f"{candidate_id}__{recipe_id}"
        if (
            pending_only
            and _has_unpromoted_terminal_srl_failure(plan_dir, task_id)
            and not _terminal_srl_retry_overridden()
        ):
            continue
        result.append((candidate_id, recipe_id))
    return result


def verify(*, output_root: Path, plan_id: str | None = None) -> dict[str, Any]:
    plan_dir, plan = load_plan(output_root, plan_id)
    complete_path = plan_dir / "COMPLETE.json"
    manifest_path = plan_dir / "artifact-manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("E046 large campaign is not complete")
    complete = _load_json(complete_path)
    manifest = _load_json(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    manifest_link_valid = (
        str(complete.get("artifact_manifest_sha256") or "") == manifest_sha256
    )
    missing: list[str] = []
    mismatched: list[str] = []
    for item in manifest:
        path = plan_dir / str(item["path"])
        if not path.is_file():
            missing.append(str(item["path"]))
        elif sha256_file(path) != str(item["sha256"]):
            mismatched.append(str(item["path"]))
    parent_rows = _parent_rows(plan_dir, plan)
    expected_parent_ids = {
        str(candidate["id"]) for candidate in plan["candidates"]
    }
    observed_parent_ids = [
        str(row.get("candidate_id") or "") for row in parent_rows
    ]
    parent_observation_coverage_exact = bool(expected_parent_ids) and (
        len(observed_parent_ids) == len(expected_parent_ids)
        and len(set(observed_parent_ids)) == len(observed_parent_ids)
        and set(observed_parent_ids) == expected_parent_ids
        and all(
            row.get("source_kind") == "parent"
            and str(row.get("parent_id") or "")
            == str(row.get("candidate_id") or "")
            for row in parent_rows
        )
    )
    required_metrics = ("clip_score", "clip_aesthetic", "hpsv2_1")
    finite_metrics = all(
        math.isfinite(_finite_number(row.get(metric), float("nan")))
        for row in parent_rows
        for metric in required_metrics
    )
    failure_audit = _technical_failure_audit(plan_dir, plan, deep=True)
    catalog_hash_valid = (
        sha256_file(plan_dir / "prompt-catalog.json")
        == str(plan["catalog_file_sha256"])
    )
    doe_hash_valid = (
        sha256_file(plan_dir / "doe.json") == str(plan["doe_file_sha256"])
    )
    scientific_contract_sha256 = plan["runtime_scientific_contract"][
        "contract_sha256"
    ]
    refinement_rows = _refinement_rows(plan_dir, plan)
    provenance_valid = all(
        row.get("source_commit") == plan["source_commit"]
        and row.get("runtime_image_digest") == plan["runtime_image_digest"]
        and row.get("scientific_contract_sha256") == scientific_contract_sha256
        for row in [*parent_rows, *refinement_rows]
    )
    selected_document = _load_json(plan_dir / "selected-refinements.json")
    selected_keys = {
        (str(item["candidate_id"]), str(item["srmpgd_recipe_id"]))
        for item in selected_document.get("tasks", [])
    }
    successful_keys = {
        (str(row["candidate_id"]), str(row["srmpgd_recipe_id"]))
        for row in refinement_rows
    }
    rows_by_successful_key: dict[
        tuple[str, str], list[Mapping[str, Any]]
    ] = {}
    for row in refinement_rows:
        key = (str(row["candidate_id"]), str(row["srmpgd_recipe_id"]))
        rows_by_successful_key.setdefault(key, []).append(row)
    maximum_iterations = {
        str(recipe["id"]): int(recipe["max_iterations"])
        for recipe in plan.get("srmpgd_recipes", [])
    }
    invalid_checkpoint_sequences: list[dict[str, Any]] = []
    expected_successful_checkpoint_count = 0
    for candidate_id, recipe_id in sorted(successful_keys):
        maximum = maximum_iterations.get(recipe_id)
        observed: list[int] = []
        malformed = maximum is None
        for row in rows_by_successful_key[(candidate_id, recipe_id)]:
            try:
                observed.append(int(row["iteration"]))
            except (KeyError, TypeError, ValueError):
                malformed = True
        expected = list(range(maximum + 1)) if maximum is not None else []
        expected_successful_checkpoint_count += len(expected)
        if malformed or len(observed) != len(expected) or sorted(observed) != expected:
            invalid_checkpoint_sequences.append(
                {
                    "candidate_id": candidate_id,
                    "srmpgd_recipe_id": recipe_id,
                    "expected_iterations": expected,
                    "observed_iterations": sorted(observed),
                }
            )
    srmpgd_checkpoint_sequences_valid = bool(successful_keys) and not (
        invalid_checkpoint_sequences
    )
    iteration_zero_rows: list[dict[str, Any]] = []
    for row in refinement_rows:
        try:
            if int(row["iteration"]) == 0:
                iteration_zero_rows.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    iteration_zero_keys = {
        (str(row["candidate_id"]), str(row["srmpgd_recipe_id"]))
        for row in iteration_zero_rows
    }
    terminal_keys = {
        (candidate_id, recipe_id)
        for candidate_id, recipe_id in selected_keys
        if _promotion_valid(
            _refinement_dir(plan_dir, candidate_id, recipe_id),
            ("task.json", "TERMINAL_FAILURE.json"),
            deep=True,
        )
    }
    refinement_resolution_complete = bool(selected_keys) and (
        not (successful_keys & terminal_keys)
        and (successful_keys | terminal_keys) == selected_keys
    )
    iteration_zero_exact = bool(successful_keys) and (
        iteration_zero_keys == successful_keys
        and all(
            row.get("iteration_zero_matches_parent_raster") is True
            for row in iteration_zero_rows
        )
    )
    result = {
        "experiment": EXPERIMENT,
        "plan_id": plan["plan_id"],
        "profile": plan["profile"],
        "manifest_entry_count": len(manifest),
        "missing": missing,
        "mismatched": mismatched,
        "manifest_sha256": manifest_sha256,
        "manifest_link_valid": manifest_link_valid,
        "catalog_hash_valid": catalog_hash_valid,
        "doe_hash_valid": doe_hash_valid,
        "runtime_provenance_valid": provenance_valid,
        "scientific_contract_sha256": scientific_contract_sha256,
        "srmpgd_iteration_zero_exact": iteration_zero_exact,
        "selected_refinement_count": len(selected_keys),
        "successful_refinement_count": len(successful_keys),
        "terminal_refinement_count": len(terminal_keys),
        "refinement_resolution_complete": refinement_resolution_complete,
        "parent_observation_count": len(parent_rows),
        "expected_parent_observation_count": plan["expected_parent_count"],
        "parent_observation_coverage_exact": parent_observation_coverage_exact,
        "srmpgd_checkpoint_sequences_valid": srmpgd_checkpoint_sequences_valid,
        "observed_successful_srmpgd_checkpoint_count": len(refinement_rows),
        "expected_successful_srmpgd_checkpoint_count": (
            expected_successful_checkpoint_count
        ),
        "invalid_srmpgd_checkpoint_sequences": invalid_checkpoint_sequences,
        "all_parent_quality_metrics_finite": finite_metrics,
        "bad_examples_retained": parent_observation_coverage_exact,
        "technical_failure_history_count": len(failure_audit["history"]),
        "technical_failure_count": len(failure_audit["unresolved"]),
        "scientific_terminal_failures_allowed": True,
        "manifest_valid": not missing and not mismatched and manifest_link_valid,
    }
    result["smoke_valid"] = bool(
        result["manifest_valid"]
        and catalog_hash_valid
        and doe_hash_valid
        and provenance_valid
        and parent_observation_coverage_exact
        and refinement_resolution_complete
        and iteration_zero_exact
        and srmpgd_checkpoint_sequences_valid
        and finite_metrics
        and not failure_audit["unresolved"]
        and (plan_dir / "selected-refinements.json").is_file()
        and (plan_dir / "REFINEMENT_SCORING_COMPLETE.json").is_file()
    )
    result["valid"] = result["smoke_valid"]
    if not result["valid"]:
        raise RuntimeError(f"E046 large verification failed: {result}")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    plan.add_argument("--profile", choices=tuple(PROFILE_SPECS), default="smoke")
    plan.add_argument("--source-commit", required=True)
    plan.add_argument("--runtime-image", required=True)
    plan.add_argument("--runtime-image-digest", required=True)

    commands = (
        "generate-parent",
        "score-parent",
        "score-parents",
        "select-refinements",
        "select",
        "generate-refinement",
        "score-refinement",
        "score-refinements",
        "aggregate",
        "status",
        "progress",
        "verify",
        "list-parents",
        "list-refinements",
        "pause-after-current",
        "clear-pause",
    )
    for name in commands:
        command = sub.add_parser(name)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        command.add_argument("--plan-id", required=False)
        if name in {"generate-parent", "score-parent", "generate-refinement", "score-refinement"}:
            command.add_argument("--candidate-id", required=True)
        if name in {"generate-refinement", "score-refinement"}:
            command.add_argument("--srmpgd-recipe-id", required=True)
        if name in {"generate-parent", "generate-refinement"}:
            command.add_argument("--source-commit", required=True)
        if name in {"list-parents", "list-refinements"}:
            command.add_argument("--pending-only", action="store_true")
    return parser


def _required_plan_id(args: Any) -> str:
    if args.plan_id:
        return str(args.plan_id)
    return str(_load_json(args.output_root / "LATEST.json")["plan_id"])


def _cli() -> int:
    args = _parser().parse_args()
    action = args.action
    if action == "plan":
        plan = create_plan(
            output_root=args.output_root,
            profile=args.profile,
            source_commit=args.source_commit,
            runtime_image=args.runtime_image,
            runtime_image_digest=args.runtime_image_digest,
        )
        result = plan_summary(plan)
    else:
        plan_id = _required_plan_id(args)
        if action == "generate-parent":
            result = generate_parent(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                source_commit=args.source_commit,
            )
        elif action == "score-parent":
            result = score_parent(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
            )
        elif action == "score-parents":
            result = score_all_parents(output_root=args.output_root, plan_id=plan_id)
        elif action in {"select-refinements", "select"}:
            result = select_refinements(output_root=args.output_root, plan_id=plan_id)
        elif action == "generate-refinement":
            result = generate_refinement(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                srmpgd_recipe_id=args.srmpgd_recipe_id,
                source_commit=args.source_commit,
            )
        elif action == "score-refinement":
            result = score_refinement(
                output_root=args.output_root,
                plan_id=plan_id,
                candidate_id=args.candidate_id,
                srmpgd_recipe_id=args.srmpgd_recipe_id,
            )
        elif action == "score-refinements":
            result = score_all_refinements(output_root=args.output_root, plan_id=plan_id)
        elif action == "aggregate":
            result = aggregate(output_root=args.output_root, plan_id=plan_id)
        elif action in {"status", "progress"}:
            result = status(output_root=args.output_root, plan_id=plan_id)
        elif action == "verify":
            result = verify(output_root=args.output_root, plan_id=plan_id)
        elif action == "pause-after-current":
            result = request_pause(output_root=args.output_root, plan_id=plan_id)
        elif action == "clear-pause":
            result = clear_pause(output_root=args.output_root, plan_id=plan_id)
        elif action == "list-parents":
            for candidate_id in list_parent_ids(
                output_root=args.output_root,
                plan_id=plan_id,
                pending_only=args.pending_only,
            ):
                print(candidate_id)
            return 0
        elif action == "list-refinements":
            for candidate_id, recipe_id in list_refinement_ids(
                output_root=args.output_root,
                plan_id=plan_id,
                pending_only=args.pending_only,
            ):
                print(f"{candidate_id}\t{recipe_id}")
            return 0
        else:  # pragma: no cover
            raise AssertionError(action)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
