"""E048 — optimise les 36 Stage2 E047 existants avec SR-MPGD, sans les régénérer.

Le montage /source est strictement en lecture seule. Tous les dérivés vont dans /night.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from . import BASE_COMMIT
from .common import read, write, sha, git_blob, utc
from . import worker as e047

STOP = False

E048_SCIENTIFIC_BLOBS = {
    "srmpgd.py": "04d1e1fb522597656cd10151076a514f366d25cd",
    "e035_loss_fidelity.py": "d47195ac2bee52c345fda1421ab6a13e8cb7fe17",
    "e035_losses.py": "c8c0670d0b65c65c67cb0c7d4d450a931244391c",
    "e035_parent_artifact.py": "534428cb3a8cfbe1caee837f6ca5aaf111d6d237",
    "e036_trust_region.py": "704ccd5a1a4ad9e3ad50e89eac0278cc72e17cf1",
    "e039_limiter_scanaware.py": "656e308c5132a78f63fd15c67a94d04260f4a9ce",
    "guidance.py": "f1fb8ae5c52a774ee46396250e9e841ce8d6e825",
    "qr.py": "93e63cf3352ed66d172aaad4099a8bc067c4b0fe",
    "quality.py": "76744a7a7bfd980ad5ffae82f0ac36508236a013",
    "e038_recipe_frontier.py": "68034ad4603cbb8316aa18743de5fa9df6cd96ce",
    "e040_checkpoint_frontier.py": "720b277ae611c120f2585af5db5e14a9fc3bf1bd",
    "e040_model_bridge.py": "c0bdc96aa3dfbc6fcd80936b1c082b366937ca8c",
}


def validate_e048_runtime() -> dict[str, Any]:
    """Atteste aussi les fichiers scientifiques utilisés uniquement par E048."""
    base = e047.validate_runtime()
    import prooftag_qr
    root = Path(prooftag_qr.__file__).parent
    for name, expected in E048_SCIENTIFIC_BLOBS.items():
        path = root / name
        if not path.is_file() or git_blob(path) != expected:
            raise RuntimeError(f"fichier scientifique E048 incompatible : {name}")
    return {**base, "e048_scientific_git_blobs": dict(E048_SCIENTIFIC_BLOBS)}


def on_signal(*_args):
    global STOP
    STOP = True


def expired(deadline: float, margin: float = 0.0) -> bool:
    return STOP or time.time() >= deadline - margin


def _source(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("e048_source_mount", "/source"))


def _work(cfg: dict[str, Any], fallback: str | Path = "/night") -> Path:
    return Path(cfg.get("e048_output_mount", fallback))


def _generation_plan(source: Path) -> dict[str, Any]:
    plan = read(source / "generation-plan.json")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("generation-plan.json ne contient aucune tâche")
    return plan


def _source_result(source: Path, task_id: str) -> dict[str, Any]:
    path = source / "generated" / task_id / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"résultat E047 absent : {task_id}")
    return read(path)


def _source_paths(source: Path, task_id: str) -> tuple[Path, Path]:
    root = source / "generated" / task_id
    return root / "images/stage2-raw.png", root / "stage2-latent.safetensors"


def _profiles(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = cfg.get("e048_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError("e048_profiles absent")
    names = set()
    clean = []
    for item in profiles:
        item = dict(item)
        name = str(item.get("id", ""))
        if not name or name in names:
            raise RuntimeError("profil SR-MPGD invalide ou dupliqué")
        names.add(name)
        if int(item.get("max_iterations", 0)) < 1 or int(item["max_iterations"]) > 40:
            raise RuntimeError(f"max_iterations hors bornes : {name}")
        if float(item.get("gamma", 0)) <= 0:
            raise RuntimeError(f"gamma invalide : {name}")
        if float(item.get("latent_radius_rms", 0)) <= 0:
            raise RuntimeError(f"latent_radius_rms invalide : {name}")
        if float(item.get("lpips_budget", -1)) < 0 or float(item.get("core_mae_budget", -1)) < 0:
            raise RuntimeError(f"budget visuel invalide : {name}")
        clean.append(item)
    return clean


def validate_source(cfg: dict[str, Any]) -> dict[str, Any]:
    source = _source(cfg)
    plan = _generation_plan(source)
    expected_payload = str(cfg["payload"])
    missing: list[str] = []
    mismatches: list[str] = []
    ids: list[str] = []
    for task in plan["tasks"]:
        task_id = str(task["id"])
        ids.append(task_id)
        image_path, latent_path = _source_paths(source, task_id)
        result_path = source / "generated" / task_id / "result.json"
        for path in (image_path, latent_path, result_path):
            if not path.is_file():
                missing.append(str(path.relative_to(source)))
        payload = str((task.get("candidate") or {}).get("payload", ""))
        if payload != expected_payload:
            mismatches.append(task_id)
        if result_path.is_file():
            result = read(result_path)
            row = result.get("row") or {}
            if bool(row.get("diffqrcoder_srmpgd_enabled", False)):
                mismatches.append(task_id + ":already_srmpgd")
    if len(ids) != int(cfg.get("e048_expected_tasks", 36)):
        raise RuntimeError(f"nombre de tâches E047 inattendu : {len(ids)}")
    if len(ids) != len(set(ids)):
        raise RuntimeError("identifiants de tâches E047 dupliqués")
    if missing:
        raise RuntimeError("artefacts E047 manquants : " + ", ".join(missing[:12]))
    if mismatches:
        raise RuntimeError("source E047 incompatible : " + ", ".join(mismatches[:12]))
    return {
        "task_count": len(ids),
        "task_ids": ids,
        "generation_plan_sha256": sha(source / "generation-plan.json"),
        "payload": expected_payload,
        "source_read_only_expected": True,
    }


def preflight(work: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = validate_e048_runtime()
    source_info = validate_source(cfg)
    profiles = _profiles(cfg)
    # Valide les API exactes utilisées par V3, sans toucher au GPU.
    import inspect
    from prooftag_qr.e035_loss_fidelity import _offload_diffusion_modules
    from prooftag_qr.e039_limiter_scanaware import E039Config
    from prooftag_qr.e040_checkpoint_frontier import Recipe as E040Recipe, _run_trajectory
    from prooftag_qr.e046_catalog import PARENT_RECIPES
    from safetensors.torch import load_file

    expected_parameters = {"pipeline", "parent", "blueprint", "recipe", "config", "output_root"}
    observed_parameters = set(inspect.signature(_run_trajectory).parameters)
    if not expected_parameters.issubset(observed_parameters):
        raise RuntimeError(
            "API E040 _run_trajectory incompatible : "
            + ",".join(sorted(expected_parameters - observed_parameters))
        )
    if not callable(_offload_diffusion_modules):
        raise RuntimeError("API d'offload diffusion indisponible")
    if not E039Config or not E040Recipe:
        raise RuntimeError("API E039/E040 indisponible")

    first = source_info["task_ids"][0]
    _, latent_path = _source_paths(_source(cfg), first)
    latent = load_file(str(latent_path), device="cpu")["latent"]
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise RuntimeError("latent Stage2 E047 invalide")

    # Construit toutes les recettes V3 avec les mêmes dataclasses que le runtime GPU.
    reference_parent = PARENT_RECIPES[4]
    for item in profiles:
        recipe = _trajectory_recipe(item)
        config = _trajectory_config(item, reference_parent)
        if recipe.max_iterations < 1 or recipe.max_iterations > 40:
            raise RuntimeError("profil E040 hors bornes")
        if config.gamma <= 0 or config.max_backtracks < 0:
            raise RuntimeError("configuration E039 invalide")

    result = {
        "status": "PASS",
        "at": utc(),
        "runtime": runtime,
        "source": source_info,
        "profile_count": len(profiles),
        "profiles": profiles,
        "latent_shape": list(latent.shape),
        "trajectory_impl": "e040_checkpoint_frontier._run_trajectory",
        "gpu_used": False,
    }
    write(work / "preflight.json", result)
    return result


def _validation_fields(e, evidence: dict[str, Any]) -> dict[str, Any]:
    fields = e._qr_fields(evidence)
    passed = int(fields["wechat_exact_presets"])
    total = int(fields["wechat_preset_count"])
    strict = bool(fields["wechat_original_exact"] and total > 0 and passed == total)
    rate = passed / total if total else 0.0
    return {
        **fields,
        "passed": passed,
        "total": total,
        "pass_rate": rate,
        "strict_all": strict,
        "worst_decoder_pass_rate": rate,
        "worst_scenario_pass_rate": rate,
    }


def _candidate_rank(item: dict[str, Any]) -> tuple[Any, ...]:
    # La validité QR domine. Parmi les stricts, on minimise la modification visuelle.
    return (
        int(bool(item.get("strict_all"))),
        int(item.get("wechat_exact_presets", 0)),
        int(bool(item.get("wechat_original_exact"))),
        -float(item.get("lpips_loss", 1e9)),
        -float(item.get("mean_absolute_change", 1e9)),
        -float(item.get("latent_delta_rms", 1e9)),
    )


def _profile_sequence(baseline_presets: int, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Les candidats proches de la frontière commencent prudemment ; les faibles scores
    # démarrent au profil équilibré pour ne pas gaspiller la nuit.
    by_id = {p["id"]: p for p in profiles}
    order = [p["id"] for p in profiles]
    if baseline_presets >= 26:
        return [by_id[name] for name in order]
    if baseline_presets >= 12:
        skip = {"catalog_g250_r100_i04"}
        return [by_id[name] for name in order if name not in skip]
    focus = {
        "catalog_g500_r200_i08", "catalog_g1000_r150_i08",
        "extended_g500_i16", "extended_g1000_i24", "robust_i32",
    }
    return [by_id[name] for name in order if name in focus]


def _trajectory_recipe(profile: dict[str, Any]):
    """Map the E048 search profile onto the already validated E040 trajectory API."""
    from prooftag_qr.e040_checkpoint_frontier import Recipe as E040Recipe
    return E040Recipe(
        name=str(profile["id"]),
        latent_radius_rms=float(profile["latent_radius_rms"]),
        max_iterations=int(profile["max_iterations"]),
        lpips_budget=float(profile.get("lpips_budget", 0.05)),
        core_mae_budget=float(profile.get("core_mae_budget", 0.05)),
        full_module_weight=float(profile.get("full_module_weight", 0.10)),
    )


def _trajectory_config(profile: dict[str, Any], parent_recipe: Any):
    """Use E039 scan-aware-v2 exactly as E046 does, with bounded E048 parameters."""
    from prooftag_qr.e039_limiter_scanaware import E039Config
    return E039Config(
        gamma=float(profile["gamma"]),
        gradient_scale=float(profile.get("gradient_scale", 32768.0)),
        lpips_weight=float(profile.get("lpips_weight", 0.01)),
        lpips_net=str(profile.get("lpips_net", "vgg")),
        crop_padding_px=int(profile.get("crop_padding_px", 78)),
        qr_version=3,
        qr_mask_pattern=int(parent_recipe.qr_mask_pattern),
        qr_module_size=20,
        quiet_zone_mode="none",
        quiet_zone_minimum_luminance=0.78,
        functional_pattern_tone_factor=0.0,
        max_backtracks=int(profile.get("max_backtracks", 12)),
        minimum_alpha=float(profile.get("minimum_alpha", 2 ** -12)),
        objective_nonincrease_tolerance=float(profile.get("objective_nonincrease_tolerance", 2e-6)),
    )


def _checkpoint_candidate(e, work: Path, scientific_plan: dict[str, Any], payload: str,
                          initial_image: Image.Image, checkpoint: Any, profile: dict[str, Any],
                          qr_scorer: Any) -> tuple[dict[str, Any], Image.Image]:
    """Score one E040 checkpoint with the real QR-Verify contract used by E047."""
    from prooftag_qr.quality import image_change_metrics

    image = Image.open(checkpoint.image_path).convert("RGB")
    evidence = e._score_qr_cached(
        plan_dir=work / "qr-cache",
        image=image,
        payload=payload,
        scorer=qr_scorer,
        plan=scientific_plan,
    )
    fields = _validation_fields(e, evidence)
    trace = dict(checkpoint.trace_step)
    changes = image_change_metrics(image, initial_image)
    candidate = {
        "profile_id": str(profile["id"]),
        "selected_iteration": int(checkpoint.iteration),
        "wechat_exact_presets": int(fields["wechat_exact_presets"]),
        "wechat_preset_count": int(fields["wechat_preset_count"]),
        "wechat_original_exact": bool(fields["wechat_original_exact"]),
        "strict_all": bool(fields["strict_all"]),
        "lpips_loss": float(trace.get("lpips_loss") or 0.0),
        "mean_absolute_change": float(changes.get("mean_absolute_change") or 0.0),
        "latent_delta_rms": float(trace.get("latent_delta_rms") or 0.0),
        "full_module_error_count": int(trace.get("full_module_error_count") or 0),
        "full_module_error_rate": float(trace.get("full_module_error_rate") or 0.0),
        "acceptance_reason": trace.get("acceptance_reason"),
        "accepted_alpha": trace.get("accepted_alpha"),
        "rejected_trial_count": int(trace.get("rejected_trial_count") or 0),
        "image_path": str(checkpoint.image_path),
        "latent_path": str(checkpoint.latent_path),
        "evidence": evidence,
        "trace_step": trace,
    }
    return candidate, image


def optimize_one(work: Path, cfg: dict[str, Any], task: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Optimize one E047 Stage2 using the E039/E040 path already exercised by E046.

    V2 mixed the generic SR-MPGD helper with an offload policy that also moved
    the scanner-loss module away from CUDA. That combined two incompatible runtime paths.
    V3 deliberately reuses E046's ``_run_trajectory`` implementation instead: it
    offloads diffusion modules, keeps the VAE in float32, loads the pinned official
    upstream SRL on CUDA and persists every checkpoint before QR-Verify scoring.
    """
    import torch
    from safetensors.torch import load_file
    from prooftag_qr.diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from prooftag_qr.e046_catalog import ParentRecipe
    from prooftag_qr.e035_loss_fidelity import _offload_diffusion_modules
    from prooftag_qr.e035_parent_artifact import LoadedParentArtifact
    from prooftag_qr.e040_checkpoint_frontier import _run_trajectory

    e, _dataset_dir, scientific_plan = e047.load(cfg)
    task_id = str(task["id"])
    dest = work / "optimized" / task_id
    final_result = dest / "result.json"
    if final_result.is_file():
        prior = read(final_result)
        if prior.get("source_generation_plan_sha256") == sha(_source(cfg) / "generation-plan.json"):
            return prior
        raise RuntimeError(f"résultat E048 incompatible déjà présent : {task_id}")
    dest.mkdir(parents=True, exist_ok=True)

    source = _source(cfg)
    source_image_path, source_latent_path = _source_paths(source, task_id)
    source_record = _source_result(source, task_id)
    baseline_row = dict(source_record.get("row") or {})
    baseline_presets = int(baseline_row.get("wechat_exact_presets") or 0)
    baseline_original = bool(baseline_row.get("wechat_original_exact"))
    c = dict(task["candidate"])
    parent_recipe = ParentRecipe(**task["recipe"])
    settings = e._settings_for_plan(scientific_plan, c, parent_recipe).model_copy(update={
        "srpg_quiet_zone_mode": "none",
        "srmpgd_enabled": False,
    })
    blueprint = e._blueprint(c, parent_recipe)
    initial_image = Image.open(source_image_path).convert("RGB")
    latent = load_file(str(source_latent_path), device="cpu")["latent"].detach().cpu().contiguous()
    parent = LoadedParentArtifact(
        root=source_image_path.parent.parent,
        image=initial_image,
        latent=latent,
        metadata={"source": baseline_row},
    )
    profiles = _profile_sequence(baseline_presets, _profiles(cfg))
    backend = UpstreamDiffQRCoderBackend(settings)
    pipeline = None
    qr = e._new_qr_scorer(work / "qr-cache", scientific_plan)
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    selected_image = initial_image.copy()
    selected_latent_path: Path | None = None
    selected = {
        "profile_id": "raw_stage2",
        "selected_iteration": 0,
        "wechat_exact_presets": baseline_presets,
        "wechat_preset_count": int(baseline_row.get("wechat_preset_count") or 37),
        "wechat_original_exact": baseline_original,
        "strict_all": bool(baseline_original and baseline_presets == 37),
        "lpips_loss": 0.0,
        "mean_absolute_change": 0.0,
        "latent_delta_rms": 0.0,
        "stop_reason": "raw_baseline",
    }
    offloaded_modules: list[str] = []
    original_vae_dtype = None
    checkpointing_was_enabled = False
    disable_checkpointing = None

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible dans le Job E048")
        pipeline = backend._load()
        original_vae_dtype = next(pipeline.vae.parameters()).dtype
        checkpointing_was_enabled = bool(getattr(pipeline.vae, "is_gradient_checkpointing", False))
        enable_checkpointing = getattr(pipeline.vae, "enable_gradient_checkpointing", None)
        disable_checkpointing = getattr(pipeline.vae, "disable_gradient_checkpointing", None)

        # Exact lifecycle copied from the E046 refinement implementation.
        with _offload_diffusion_modules(pipeline) as moved:
            offloaded_modules = list(moved)
            if not checkpointing_was_enabled and callable(enable_checkpointing):
                enable_checkpointing()
            pipeline.vae.requires_grad_(False).eval().to(dtype=torch.float32)

            for profile in profiles:
                if bool(selected.get("strict_all")):
                    break
                if expired(deadline, float(cfg.get("e048_profile_start_margin_seconds", 300))):
                    break

                profile_id = str(profile["id"])
                profile_dir = dest / "profiles" / profile_id
                profile_dir.mkdir(parents=True, exist_ok=True)
                profile_started = time.perf_counter()
                checkpoints = _run_trajectory(
                    pipeline=pipeline,
                    parent=parent,
                    blueprint=blueprint,
                    recipe=_trajectory_recipe(profile),
                    config=_trajectory_config(profile, parent_recipe),
                    output_root=profile_dir / "trajectory",
                )
                if not checkpoints:
                    raise RuntimeError(f"SR-MPGD n'a produit aucun checkpoint : {profile_id}")

                checkpoint_rows: list[dict[str, Any]] = []
                profile_best: dict[str, Any] | None = None
                profile_best_image: Image.Image | None = None
                profile_best_latent: Path | None = None
                for checkpoint in checkpoints:
                    if expired(deadline, 60):
                        break
                    candidate, image = _checkpoint_candidate(
                        e, work, scientific_plan, str(c["payload"]), initial_image,
                        checkpoint, profile, qr,
                    )
                    # Evidence can be large; persist it separately but keep the ranking row small.
                    evidence = candidate.pop("evidence")
                    evidence_dir = profile_dir / "qr-evidence"
                    evidence_dir.mkdir(exist_ok=True)
                    write(evidence_dir / f"iteration-{int(checkpoint.iteration):03d}.json", evidence)
                    checkpoint_rows.append({k: v for k, v in candidate.items() if k != "trace_step"})
                    if profile_best is None or _candidate_rank(candidate) > _candidate_rank(profile_best):
                        profile_best = candidate
                        profile_best_image = image.copy()
                        profile_best_latent = Path(checkpoint.latent_path)

                if profile_best is None or profile_best_image is None or profile_best_latent is None:
                    raise RuntimeError(f"aucun checkpoint scoré avant deadline : {profile_id}")

                profile_best = dict(profile_best)
                profile_best["profile_wall_s"] = time.perf_counter() - profile_started
                profile_best["checkpoint_count"] = len(checkpoints)
                profile_best["profile"] = profile
                profile_best_image.save(profile_dir / "selected.png", format="PNG", optimize=False, compress_level=9)
                shutil.copy2(profile_best_latent, profile_dir / "selected-latent.safetensors")
                write(profile_dir / "checkpoints.json", checkpoint_rows)
                write(profile_dir / "result.json", {
                    **{k: v for k, v in profile_best.items() if k not in {"trace_step"}},
                    "trajectory_impl": "e040_checkpoint_frontier._run_trajectory",
                    "all_checkpoints_real_qr_verify": True,
                })
                attempts.append(profile_best)

                if _candidate_rank(profile_best) > _candidate_rank(selected):
                    selected = profile_best
                    selected_image = profile_best_image.copy()
                    selected_latent_path = profile_best_latent

                gc.collect()
                torch.cuda.empty_cache()

        selected_image.save(dest / "best.png", format="PNG", optimize=False, compress_level=9)
        if selected_latent_path is None:
            shutil.copy2(source_latent_path, dest / "best-latent.safetensors")
        else:
            shutil.copy2(selected_latent_path, dest / "best-latent.safetensors")
        result_payload = {
            "task_id": task_id,
            "method": task.get("method"),
            "prompt": c.get("prompt"),
            "payload": c.get("payload"),
            "seed": c.get("seed"),
            "source_image_sha256": sha(source_image_path),
            "source_latent_sha256": sha(source_latent_path),
            "source_generation_plan_sha256": sha(source / "generation-plan.json"),
            "baseline": {
                "wechat_exact_presets": baseline_presets,
                "wechat_original_exact": baseline_original,
                "strict_all": bool(baseline_original and baseline_presets == 37),
                "row": baseline_row,
            },
            "selected": {k: v for k, v in selected.items() if k not in {"trace_step", "profile"}},
            "attempts": [
                {k: v for k, v in item.items() if k not in {"trace_step", "profile"}}
                for item in attempts
            ],
            "strict_success": bool(selected.get("strict_all")),
            "elapsed_s": time.perf_counter() - started,
            "completed_at": utc(),
            "source_modified": False,
            "offloaded_diffusion_modules": offloaded_modules,
            "trajectory_impl": "e040_checkpoint_frontier._run_trajectory",
        }
        write(final_result, result_payload)
        return result_payload
    except BaseException as exc:
        write(dest / "FAILED.json", {
            "at": utc(),
            "task_id": task_id,
            "type": type(exc).__name__,
            "error": str(exc)[:6000],
            "traceback": traceback.format_exc()[-16000:],
        })
        raise
    finally:
        try:
            qr.close()
        except Exception:
            pass
        if pipeline is not None and original_vae_dtype is not None:
            try:
                pipeline.vae.to(dtype=original_vae_dtype)
                if not checkpointing_was_enabled and callable(disable_checkpointing):
                    disable_checkpointing()
            except Exception:
                pass
        if backend is not None:
            backend._pipeline = None
        del pipeline
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def gpu_canary(work: Path, cfg: dict[str, Any], deadline: float) -> dict[str, Any]:
    """Execute one real E040-backed SR-MPGD iteration before the long campaign."""
    validate_e048_runtime()
    source_info = validate_source(cfg)
    plan = _generation_plan(_source(cfg))

    def priority(task: dict[str, Any]):
        record = _source_result(_source(cfg), str(task["id"]))
        row = record.get("row") or {}
        return (
            0 if task.get("method") == "advisor" else 1,
            -int(row.get("wechat_exact_presets") or 0),
            str(task["id"]),
        )

    task = sorted(plan["tasks"], key=priority)[0]
    base = dict(_profiles(cfg)[0])
    base.update({
        "id": "gpu_canary_i01",
        "max_iterations": 1,
        "qr_verify_interval": 1,
        "preview_interval": 1,
    })
    canary_cfg = dict(cfg)
    canary_cfg["e048_profiles"] = [base]
    canary_root = work / "gpu-canary"
    started = time.time()
    try:
        result = optimize_one(canary_root, canary_cfg, task, deadline)
        marker = {
            "status": "PASS",
            "at": utc(),
            "task_id": str(task["id"]),
            "elapsed_s": time.time() - started,
            "selected": result.get("selected"),
            "source_generation_plan_sha256": source_info["generation_plan_sha256"],
            "real_gpu_path_tested": True,
            "trajectory_impl": "e040_checkpoint_frontier._run_trajectory",
            "source_modified": False,
        }
        write(work / "GPU_CANARY_PASS.json", marker)
        return marker
    except BaseException as exc:
        write(work / "GPU_CANARY_FAILED.json", {
            "status": "FAIL",
            "at": utc(),
            "task_id": str(task["id"]),
            "elapsed_s": time.time() - started,
            "type": type(exc).__name__,
            "error": str(exc)[:6000],
            "traceback": traceback.format_exc()[-16000:],
            "trajectory_impl": "e040_checkpoint_frontier._run_trajectory",
        })
        raise


def optimize(work: Path, cfg: dict[str, Any], deadline: float) -> dict[str, Any]:
    validate_e048_runtime()
    source_info = validate_source(cfg)
    plan = _generation_plan(_source(cfg))
    # Priorise les sorties adviser, puis les parents déjà proches de 37/37.
    def priority(task: dict[str, Any]):
        record = _source_result(_source(cfg), str(task["id"]))
        row = record.get("row") or {}
        return (
            0 if task.get("method") == "advisor" else 1,
            -int(row.get("wechat_exact_presets") or 0),
            str(task["id"]),
        )
    tasks = sorted(plan["tasks"], key=priority)
    done = 0
    failures = 0
    strict = 0
    started = time.time()
    for task in tasks:
        if expired(deadline, float(cfg.get("e048_candidate_start_margin_seconds", 600))):
            break
        task_id = str(task["id"])
        try:
            result = optimize_one(work, cfg, task, deadline)
            done += 1
            strict += int(bool(result.get("strict_success")))
        except BaseException as exc:
            failures += 1
            write(work / "failures" / f"{task_id}.json", {
                "task_id": task_id,
                "type": type(exc).__name__,
                "error": str(exc)[:6000],
                "at": utc(),
            })
            if failures >= 3 and done == 0:
                raise RuntimeError("E048 échoue sur les premiers candidats; arrêt fail-closed") from exc
        progress = {
            "done": done,
            "failures": failures,
            "strict_successes": strict,
            "total": len(tasks),
            "remaining": len(tasks) - done - failures,
            "elapsed_s": time.time() - started,
            "deadline_epoch": deadline,
            "at": utc(),
        }
        write(work / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    summary = {
        "source": source_info,
        "done": done,
        "failures": failures,
        "strict_successes": strict,
        "total": len(tasks),
        "partial_by_deadline": done + failures < len(tasks),
        "at": utc(),
    }
    write(work / "OPTIMIZATION_COMPLETE.json", summary)
    return summary


def _best_records(work: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((work / "optimized").glob("*/result.json")):
        item = read(path)
        item["result_path"] = str(path)
        out.append(item)
    return out


def _safe_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _make_contact_sheet(work: Path, records: list[dict[str, Any]], limit: int = 12) -> None:
    ranked = sorted(records, key=lambda r: (
        -int(bool(r["selected"].get("strict_all"))),
        -int(r["selected"].get("wechat_exact_presets", 0)),
        float(r["selected"].get("mean_absolute_change", 1e9)),
    ))[:limit]
    if not ranked:
        return
    thumb = 320
    label_h = 74
    cols = 4
    rows = math.ceil(len(ranked) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _safe_font(16)
    for index, rec in enumerate(ranked):
        image = Image.open(Path(rec["result_path"]).parent / "best.png").convert("RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        sheet.paste(image, (x + (thumb - image.width)//2, y + (thumb - image.height)//2))
        sel = rec["selected"]
        text = f"{rec['task_id']}\n{sel['wechat_exact_presets']}/37 | {sel['profile_id']}"
        draw.multiline_text((x + 6, y + thumb + 4), text, fill="black", font=font, spacing=2)
    (work / "report").mkdir(parents=True, exist_ok=True)
    sheet.save(work / "report/contact-sheet.png", format="PNG")


def _make_gif_for_record(work: Path, record: dict[str, Any], rank: int) -> None:
    import imageio.v2 as imageio
    profile_id = str(record["selected"].get("profile_id"))
    if profile_id == "raw_stage2":
        return
    # V3 uses the validated E040 trajectory layout, which persists every real checkpoint.
    root = (Path(record["result_path"]).parent / "profiles" / profile_id
            / "trajectory" / profile_id / "images")
    frames = sorted(root.glob("iteration-*.png"))
    final = Path(record["result_path"]).parent / "best.png"
    if final.is_file() and (not frames or sha(final) != sha(frames[-1])):
        frames.append(final)
    if len(frames) < 2:
        return
    images = [imageio.imread(path) for path in frames]
    out = work / "report" / "gifs"
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / f"top-{rank:02d}-{record['task_id']}.gif", images, duration=0.7, loop=0)


def report(work: Path, cfg: dict[str, Any], deadline: float) -> dict[str, Any]:
    records = _best_records(work)
    if not records:
        raise RuntimeError("aucun résultat E048 à rapporter")
    report_dir = work / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    # Score qualité uniquement sur les images retenues, après la phase GPU.
    e, _d, scientific_plan = e047.load(cfg)
    generation = _generation_plan(_source(cfg))
    tasks = {str(t["id"]): t for t in generation["tasks"]}
    quality_scorer = None
    rows = []
    try:
        from prooftag_qr.quality_scoring import quality_scorer_from_settings
        from prooftag_qr.e046_catalog import ParentRecipe
        for rec in records:
            if expired(deadline, 60):
                break
            task = tasks[rec["task_id"]]
            c = dict(task["candidate"])
            recipe = ParentRecipe(**task["recipe"])
            settings = e._settings_for_plan(scientific_plan, c, recipe)
            if quality_scorer is None:
                quality_scorer = quality_scorer_from_settings(settings, device="cpu")
            image = Image.open(Path(rec["result_path"]).parent / "best.png").convert("RGB")
            quality, _prov = e._quality_scores(
                images={"best": image}, prompt=str(c["prompt"]), settings=settings, scorer=quality_scorer
            )
            q = quality["best"]
            selected = rec["selected"]
            baseline = rec["baseline"]
            rows.append({
                "task_id": rec["task_id"],
                "method": rec.get("method"),
                "prompt": rec.get("prompt"),
                "seed": rec.get("seed"),
                "profile_id": selected.get("profile_id"),
                "selected_iteration": selected.get("selected_iteration"),
                "raw_presets": int(baseline.get("wechat_exact_presets", 0)),
                "optimized_presets": int(selected.get("wechat_exact_presets", 0)),
                "preset_gain": int(selected.get("wechat_exact_presets", 0)) - int(baseline.get("wechat_exact_presets", 0)),
                "raw_original_exact": bool(baseline.get("wechat_original_exact")),
                "optimized_original_exact": bool(selected.get("wechat_original_exact")),
                "strict_37": bool(selected.get("strict_all")),
                "lpips_loss": selected.get("lpips_loss"),
                "mean_absolute_change": selected.get("mean_absolute_change"),
                "clip_score": q.get("clip_score"),
                "clip_aesthetic": q.get("clip_aesthetic"),
                "hpsv2_1": q.get("hpsv2_1"),
            })
    finally:
        del quality_scorer
        gc.collect()

    import csv
    if rows:
        fields = list(rows[0])
        with (report_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    strict_count = sum(int(r["strict_37"]) for r in rows)
    summary = {
        "reported_results": len(rows),
        "strict_37": strict_count,
        "mean_raw_presets": sum(r["raw_presets"] for r in rows) / len(rows) if rows else None,
        "mean_optimized_presets": sum(r["optimized_presets"] for r in rows) / len(rows) if rows else None,
        "mean_gain": sum(r["preset_gain"] for r in rows) / len(rows) if rows else None,
        "best_presets": max((r["optimized_presets"] for r in rows), default=None),
        "source_run": cfg["source_run_id"],
        "at": utc(),
        "physical_phone_validation": False,
    }
    write(report_dir / "summary.json", summary)
    _make_contact_sheet(work, records)
    ranked = sorted(records, key=lambda r: (
        -int(bool(r["selected"].get("strict_all"))),
        -int(r["selected"].get("wechat_exact_presets", 0)),
        float(r["selected"].get("mean_absolute_change", 1e9)),
    ))
    for idx, rec in enumerate(ranked[:6], 1):
        if expired(deadline, 30):
            break
        _make_gif_for_record(work, rec, idx)
    # Rapport Markdown autonome.
    lines = [
        "# E048 — SR-MPGD adaptatif sur les 36 Stage2 E047",
        "",
        f"- Résultats rapportés : **{summary['reported_results']}**",
        f"- QR stricts 37/37 : **{summary['strict_37']}**",
        f"- Presets moyens avant : **{summary['mean_raw_presets']:.2f}/37**" if rows else "- Aucun résultat",
        f"- Presets moyens après : **{summary['mean_optimized_presets']:.2f}/37**" if rows else "",
        f"- Gain moyen : **{summary['mean_gain']:+.2f} presets**" if rows else "",
        "- Validation téléphone : **NON TESTÉE**",
        "- Les sources E047 sont montées en lecture seule ; aucune régénération Stage1/Stage2.",
    ]
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "canary", "optimize", "report"])
    parser.add_argument("--config", default="/config/config.json")
    parser.add_argument("--work", default="/night")
    parser.add_argument("--deadline", type=float, default=time.time() + 3600)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    cfg = read(args.config)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.action == "preflight":
        preflight(work, cfg)
    elif args.action == "canary":
        gpu_canary(work, cfg, args.deadline)
    elif args.action == "optimize":
        optimize(work, cfg, args.deadline)
    else:
        report(work, cfg, args.deadline)


if __name__ == "__main__":
    main()
