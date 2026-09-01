"""Recover E041 Phase B from the already-complete 54 Phase-A checkpoints.

This hotfix never reruns Stage 1, Stage 2 or SR-MPGD. It invalidates only the
pre-hotfix 740px Phase-B raster/scoring directories, regenerates the 18 functional
variants on the exact 736px DiffQRCoder geometry, rescans them, and writes the final
E041 verdict/pipeline artefacts.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .e035_loss_fidelity import _atomic_json, _atomic_text
from .e035_parent_capture import CaptureConfig, _settings_document
from .e038_recipe_frontier import _comparison_sheet
from .e040_model_bridge import advisor_preview
from .e041_gamma_functional_frontier import (
    ERROR_CORRECTION,
    EXPERIMENT,
    FUNCTIONAL_TONE_FACTORS,
    GAMMAS,
    LATENT_RADIUS_RMS,
    PAYLOAD,
    PHASE_B_BASE_COUNT,
    PROMPT,
    QR_MASK_PATTERN,
    QR_MODULE_SIZE,
    QR_PADDING_PX,
    QR_VERSION,
    SEED,
    _functional_tone_exact_diffqrcoder,
    _manifest,
    _rank_key,
    _save_png,
    _score_rows,
)


def _backup_invalid(root: Path, name: str) -> None:
    source = root / name
    backup = root / f"{name}-invalid-740px-prehotfix"
    if source.exists() and not backup.exists():
        source.rename(backup)
    elif source.exists() and backup.exists():
        shutil.rmtree(source)


def _validate_phase_a(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison = root / "phase-a-scoring/comparison.json"
    selected = root / "phase-b-selected-bases.json"
    if not comparison.is_file() or not selected.is_file():
        raise FileNotFoundError("E041 Phase-A scoring/selection is incomplete")
    rows = json.loads(comparison.read_text(encoding="utf-8"))
    bases = json.loads(selected.read_text(encoding="utf-8"))
    if len(rows) != 54:
        raise RuntimeError(f"expected 54 Phase-A checkpoints, got {len(rows)}")
    if len(bases) != PHASE_B_BASE_COUNT:
        raise RuntimeError(f"expected {PHASE_B_BASE_COUNT} Phase-B bases, got {len(bases)}")
    for row in rows:
        image_path = Path(str(row.get("image_path", "")))
        latent_path = Path(str(row.get("latent_path", "")))
        if not image_path.is_file() or not latent_path.is_file():
            raise FileNotFoundError(f"Phase-A checkpoint artefact missing: {row.get('variant')}")
        with Image.open(image_path) as image:
            if image.size != (736, 736):
                raise RuntimeError(f"Phase-A raster is not 736x736: {image_path} -> {image.size}")
    return rows, bases


def recover_e041(*, output_dir: Path) -> dict[str, Any]:
    from .config import Settings
    from .qr import generate_diffqrcoder_qr

    verdict_path = output_dir / "verdict.json"
    if verdict_path.is_file():
        return json.loads(verdict_path.read_text(encoding="utf-8"))

    plan_path = output_dir / "plan.json"
    parent_path = output_dir / "parent/stage2-scan-ready.png"
    if not plan_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("E041 parent/plan missing; recovery refuses to regenerate them")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment") != EXPERIMENT or plan.get("prompt") != PROMPT:
        raise RuntimeError("E041 plan does not match the hotfix experiment")
    phase_a_rows, phase_b_bases = _validate_phase_a(output_dir)

    parent_image = Image.open(parent_path).convert("RGB")
    if parent_image.size != (736, 736):
        raise RuntimeError(f"E041 parent must be 736x736, got {parent_image.size}")
    blueprint = generate_diffqrcoder_qr(
        PAYLOAD,
        ERROR_CORRECTION,
        version=QR_VERSION,
        mask_pattern=QR_MASK_PATTERN,
        module_size=QR_MODULE_SIZE,
    )

    _backup_invalid(output_dir, "phase-b-functional")
    _backup_invalid(output_dir, "phase-b-scoring")
    phase_b_root = output_dir / "phase-b-functional"

    phase_a_by_variant = {str(row["variant"]): row for row in phase_a_rows}
    phase_b_images: dict[str, Image.Image] = {}
    phase_b_meta: dict[str, dict[str, Any]] = {}
    for base_row in phase_b_bases:
        base_key = str(base_row["variant"])
        authoritative = phase_a_by_variant.get(base_key)
        if authoritative is None:
            raise RuntimeError(f"selected Phase-B base absent from Phase A: {base_key}")
        base_image = Image.open(str(authoritative["image_path"])).convert("RGB")
        for factor in FUNCTIONAL_TONE_FACTORS:
            factor_token = f"{int(round(factor * 100)):02d}"
            key = f"{base_key}__tone{factor_token}"
            image = _functional_tone_exact_diffqrcoder(base_image, blueprint, factor)
            if image.size != (736, 736):
                raise RuntimeError(f"hotfix produced wrong raster size for {key}: {image.size}")
            image_path = phase_b_root / base_key / f"tone-{factor_token}.png"
            _save_png(image_path, image)
            phase_b_images[key] = image
            phase_b_meta[key] = {
                "phase": "B",
                "base_checkpoint": base_key,
                "gamma": float(base_row["gamma"]),
                "iteration": int(base_row["iteration"]),
                "radius": LATENT_RADIUS_RMS,
                "functional_tone_factor": factor,
                "image_path": str(image_path),
                "latent_path": authoritative["latent_path"],
                "latent_delta_rms": authoritative.get("latent_delta_rms"),
                "is_parent_reference": False,
                "geometry_hotfix": "exact-736-padding78-core29x29-module20",
            }

    reference_key = "phase-b-parent-reference"
    phase_b_images[reference_key] = parent_image
    phase_b_meta[reference_key] = {
        "phase": "B_REFERENCE",
        "gamma": 0.0,
        "iteration": 0,
        "radius": 0.0,
        "functional_tone_factor": 0.0,
        "image_path": str(parent_path),
        "latent_path": str(output_dir / "parent/stage2-latent.safetensors"),
        "is_parent_reference": True,
        "geometry_hotfix": "exact-736-padding78-core29x29-module20",
    }

    capture = CaptureConfig(payload=PAYLOAD, prompt=PROMPT, seed=SEED)
    base_settings = Settings()
    settings = Settings.model_validate({**base_settings.model_dump(), **_settings_document(capture)})
    backend_proxy = SimpleNamespace(settings=settings)
    phase_b_rows_all, phase_b_surrogate = _score_rows(
        images=phase_b_images,
        metadata=phase_b_meta,
        output_dir=output_dir / "phase-b-scoring",
        backend=backend_proxy,
        blueprint=blueprint,
        parent_image=parent_image,
        trace_lpips=None,
    )
    phase_b_rows = [row for row in phase_b_rows_all if row["phase"] == "B"]
    if len(phase_b_rows) != PHASE_B_BASE_COUNT * len(FUNCTIONAL_TONE_FACTORS):
        raise RuntimeError(f"expected 18 Phase-B variants, got {len(phase_b_rows)}")
    safe_b = [row for row in phase_b_rows if row["visual_guard_pass"]]
    if not safe_b:
        raise RuntimeError("E041 recovered Phase B has no visually-safe candidate")
    winner = sorted(safe_b, key=_rank_key)[0]

    advisor = advisor_preview(
        prompt=PROMPT,
        payload_length=len(PAYLOAD),
        error_correction=ERROR_CORRECTION,
        qr_context={
            "qr_version": QR_VERSION,
            "qr_mask_pattern": QR_MASK_PATTERN,
            "qr_module_size": QR_MODULE_SIZE,
            "qr_padding_px": QR_PADDING_PX,
        },
    )
    _atomic_json(output_dir / "advisor-preview.json", advisor)

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    _save_png(pipeline_dir / "01-qr-reference.png", blueprint.image)
    _save_png(
        pipeline_dir / "02-control-condition.png",
        blueprint.image.resize((736, 736), Image.Resampling.NEAREST),
    )
    shutil.copy2(output_dir / "parent/stage1.png", pipeline_dir / "03-stage1.png")
    shutil.copy2(output_dir / "parent/stage2.png", pipeline_dir / "04-stage2.png")
    shutil.copy2(parent_path, pipeline_dir / "05-stage2-scan-ready.png")
    final_image_path = pipeline_dir / "99-FINAL-QR.png"
    final_latent_path = pipeline_dir / "99-FINAL-latent.safetensors"
    shutil.copy2(Path(winner["image_path"]), final_image_path)
    shutil.copy2(Path(winner["latent_path"]), final_latent_path)
    _atomic_json(
        pipeline_dir / "99-FINAL-raster-postprocess.json",
        {
            "base_checkpoint": winner["base_checkpoint"],
            "selected_gamma": winner["gamma"],
            "functional_tone_factor": winner["functional_tone_factor"],
            "geometry": "736x736; 78px padding; 29x29 core; 20px/module",
            "data_modules_projected": False,
            "recovered_from_existing_phase_a": True,
        },
    )

    sheet_items: list[tuple[str, Image.Image, str]] = [
        ("QR reference", blueprint.image.convert("RGB"), PAYLOAD),
        ("Stage 1", Image.open(output_dir / "parent/stage1.png").convert("RGB"), "fresh E041 prompt"),
        ("Stage 2", Image.open(output_dir / "parent/stage2.png").convert("RGB"), "fresh SRPG parent"),
        ("Stage 2 scan-ready", parent_image, "tone=0 baseline"),
    ]
    for row in phase_b_bases:
        sheet_items.append(
            (
                f"Gamma {int(row['gamma'])} i{int(row['iteration'])}",
                Image.open(str(row["image_path"])).convert("RGB"),
                f"SSR={row['qr_verify_exact_presets']}/37 safe={row['visual_guard_pass']}",
            )
        )
    sheet_items.append(
        (
            "FINAL E041",
            Image.open(final_image_path).convert("RGB"),
            f"gamma={int(winner['gamma'])} tone={winner['functional_tone_factor']:.2f} SSR={winner['qr_verify_exact_presets']}/37",
        )
    )
    _comparison_sheet(pipeline_dir / "full-pipeline-contact-sheet.png", sheet_items, columns=4)

    projection_summary = []
    for gamma in GAMMAS:
        rows = [row for row in phase_a_rows if math.isclose(float(row["gamma"]), gamma)]
        projection_summary.append(
            {
                "gamma": gamma,
                "checkpoint_count": len(rows),
                "projection_active_count": sum(bool(row.get("projection_was_active")) for row in rows),
                "mean_accepted_alpha": (
                    float(np.mean([row["accepted_alpha"] for row in rows if row.get("accepted_alpha") is not None]))
                    if any(row.get("accepted_alpha") is not None for row in rows)
                    else None
                ),
                "best_ssr_exact_presets": max(int(row["qr_verify_exact_presets"]) for row in rows),
            }
        )
    _atomic_json(output_dir / "gamma-projection-summary.json", projection_summary)

    phase_a_surrogate_path = output_dir / "phase-a-scoring/e016-surrogate-status.json"
    phase_a_surrogate = (
        json.loads(phase_a_surrogate_path.read_text(encoding="utf-8"))
        if phase_a_surrogate_path.is_file()
        else {}
    )
    verdict = {
        "experiment": EXPERIMENT,
        "prompt": PROMPT,
        "prompt_changed_from_e040": True,
        "e040_paired_comparison_allowed": False,
        "gamma_grid": list(GAMMAS),
        "historical_gamma_baseline": 1000.0,
        "selected_gamma": winner["gamma"],
        "selected_iteration": winner["iteration"],
        "selected_functional_tone_factor": winner["functional_tone_factor"],
        "winner_variant": winner["variant"],
        "winner_ssr_exact_presets": winner["qr_verify_exact_presets"],
        "winner_ssr": winner["ssr"],
        "winner_original_exact": winner["original_exact"],
        "winner_full_module_error_count": winner["full_module_error_count"],
        "winner_functional_center_error_rate": winner["functional_center_error_rate"],
        "winner_data_center_error_rate": winner["data_center_error_rate"],
        "winner_lpips": winner["lpips"],
        "winner_visual_guard_pass": winner["visual_guard_pass"],
        "winner_visual_guard_checks": winner["visual_guard_checks"],
        "phase_a_checkpoint_count": len(phase_a_rows),
        "phase_a_safe_checkpoint_count": sum(bool(row["visual_guard_pass"]) for row in phase_a_rows),
        "phase_b_variant_count": len(phase_b_rows),
        "phase_b_safe_variant_count": len(safe_b),
        "advisor_available": bool(advisor.get("available")),
        "e016_surrogate_research_usable": bool(
            phase_b_surrogate.get("research_usable") or phase_a_surrogate.get("research_usable")
        ),
        "phase_b_geometry_hotfix": "exact-736-padding78-core29x29-module20",
        "recovered_from_existing_phase_a": True,
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": "REVIEW_GAMMA_EFFECT_AND_FUNCTIONAL_PATTERN_GAIN_BEFORE_ANY_GENERALIZATION",
    }
    _atomic_json(verdict_path, verdict)
    _atomic_text(
        output_dir / "report.md",
        "\n".join(
            [
                "# E041 — gamma + motifs fonctionnels (Phase B hotfix)",
                "",
                f"- nouveau prompt : `{PROMPT}`",
                f"- gamma testés : {', '.join(str(int(x)) for x in GAMMAS)}",
                "- Phase A : réutilisée intégralement, 54 checkpoints existants",
                "- Phase B : régénérée sur géométrie exacte 736 / padding 78 / 29x29x20",
                "- data modules : jamais projetés par le hotfix",
                f"- meilleur gamma : **{winner['gamma']}**",
                f"- checkpoint : **i{winner['iteration']}**",
                f"- tone factor : **{winner['functional_tone_factor']}**",
                f"- SSR : **{winner['qr_verify_exact_presets']}/37**",
                f"- original exact : **{winner['original_exact']}**",
                f"- visual guard : **{winner['visual_guard_pass']}**",
                "- production/generalisation : **non**",
                "",
            ]
        ),
    )
    _atomic_json(output_dir / "e041-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verdict = recover_e041(output_dir=args.output_dir)
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
