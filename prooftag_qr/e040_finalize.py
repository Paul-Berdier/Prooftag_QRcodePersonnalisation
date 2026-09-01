"""Finalize an E040 run that already computed/scored all 45 checkpoints.

This command is intentionally CPU-only. It never reruns diffusion, VAE decoding, SR-MPGD,
QR-Verify, CLIP/HPS or the E016 surrogate. It validates the persisted checkpoint table, selects
the winner using the original E040 ranking, restores Stage-1 evidence, runs the optional advisor,
and writes the missing pipeline/verdict/report/manifest artifacts.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from .e035_loss_fidelity import _atomic_json, _atomic_text
from .e038_recipe_frontier import _comparison_sheet
from .e040_checkpoint_frontier import DEFAULT_RADII, EXPERIMENT, _manifest
from .e040_model_bridge import advisor_preview


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = [row for row in rows if row.get("visual_guard_pass") is True]
    return sorted(
        safe,
        key=lambda row: (
            -int(row["qr_verify_exact_presets"]),
            -int(bool(row["original_exact"])),
            int(row["full_module_error_count"]),
            -float(row.get("surrogate_mean_success_probability") or -1.0),
            float(row["lpips"]),
            int(row["iteration"]),
            abs(float(row["radius"]) - 0.20),
        ),
    )


def _validate_checkpoint_contract(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 45:
        raise RuntimeError(f"E040 partial is not finalizable: expected 45 checkpoints, got {len(rows)}")
    methods = sorted({str(row["method"]) for row in rows})
    if len(methods) != 5:
        raise RuntimeError(f"E040 partial is not finalizable: expected 5 methods, got {methods}")
    observed_radii = sorted({round(float(row["radius"]), 3) for row in rows})
    expected_radii = sorted(round(value, 3) for value in DEFAULT_RADII)
    if observed_radii != expected_radii:
        raise RuntimeError(f"E040 radii mismatch: expected={expected_radii} observed={observed_radii}")
    for method in methods:
        iterations = sorted(int(row["iteration"]) for row in rows if row["method"] == method)
        if iterations != list(range(9)):
            raise RuntimeError(f"E040 {method} checkpoints incomplete: {iterations}")
    if any(float(row["gamma"]) != 1000.0 for row in rows):
        raise RuntimeError("E040 gamma contract violated in persisted checkpoint table")
    for row in rows:
        for field in ("image_path", "latent_path"):
            path = Path(str(row[field]))
            if not path.is_file():
                raise FileNotFoundError(f"persisted E040 {field} missing: {path}")


def finalize_e040(*, parent_dir: Path, e039_results_dir: Path, output_dir: Path) -> dict[str, Any]:
    verdict_path = output_dir / "verdict.json"
    if verdict_path.is_file():
        verdict = _load_json(verdict_path)
        if verdict.get("experiment") != EXPERIMENT or verdict.get("gamma") != 1000.0:
            raise RuntimeError("existing E040 verdict does not match the expected experiment")
        return verdict

    rows = _load_json(output_dir / "checkpoint-comparison.json")
    if not isinstance(rows, list):
        raise TypeError("checkpoint-comparison.json must be a list")
    _validate_checkpoint_contract(rows)
    ranked = _rank_safe(rows)
    winner = ranked[0] if ranked else None
    if winner is None:
        raise RuntimeError("E040 has no visually-safe checkpoint; refusing to invent a final QR")

    parent_meta = _load_json(parent_dir / "parent-stage2-metadata.json")
    source = dict(parent_meta["source"])
    payload = str(source["payload"])
    prompt = str(source.get("prompt") or "")
    e039_control = _load_json(output_dir / "e039-control.json")
    if e039_control.get("verdict", {}).get("experiment") != "e039-srmpgd-limiter-scanaware-v1":
        raise RuntimeError("E040 partial has an invalid E039 control")
    if e039_control.get("verdict", {}).get("gamma") != 1000.0:
        raise RuntimeError("E039 control gamma mismatch")

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    for required in ("01-qr-reference.png", "02-control-condition.png", "04-stage2.png"):
        if not (pipeline_dir / required).is_file():
            raise FileNotFoundError(f"E040 pipeline prerequisite missing: {pipeline_dir / required}")

    stage1_asset = Path(__file__).resolve().parents[1] / "docs/e035-assets/e034-observed-stage1.png"
    if not stage1_asset.is_file():
        raise FileNotFoundError(
            "archived Stage-1 raster is missing from the rebuilt API image: " + str(stage1_asset)
        )
    shutil.copy2(stage1_asset, pipeline_dir / "03-stage1.png")

    selected_image_path = pipeline_dir / "99-FINAL-QR.png"
    selected_latent_path = pipeline_dir / "99-FINAL-latent.safetensors"
    shutil.copy2(Path(str(winner["image_path"])), selected_image_path)
    shutil.copy2(Path(str(winner["latent_path"])), selected_latent_path)

    advisor = advisor_preview(
        prompt=prompt,
        payload_length=len(payload),
        error_correction=str(source["error_correction"]),
        qr_context={
            "qr_version": int(source["qr_version"]),
            "qr_mask_pattern": int(source["qr_mask_pattern"]),
            "qr_module_size": int(source["qr_module_size"]),
            "qr_padding_px": int(source["qr_padding_px"]),
        },
    )
    _atomic_json(output_dir / "advisor-preview.json", advisor)

    surrogate_info = _load_json(output_dir / "e016-surrogate-status.json")
    winner_recipe = str(winner["method"])
    trajectory = sorted(
        [row for row in rows if str(row["method"]) == winner_recipe],
        key=lambda row: int(row["iteration"]),
    )
    if [int(row["iteration"]) for row in trajectory] != list(range(9)):
        raise RuntimeError("winner trajectory is incomplete")

    sheet_items: list[tuple[str, Image.Image, str]] = [
        ("QR reference", Image.open(pipeline_dir / "01-qr-reference.png").convert("RGB"), "exact payload"),
        ("Control condition", Image.open(pipeline_dir / "02-control-condition.png").convert("RGB"), "binary QR condition"),
        ("Stage 1", Image.open(pipeline_dir / "03-stage1.png").convert("RGB"), "Cetus-Mix + QR Monster"),
        ("Stage 2", Image.open(pipeline_dir / "04-stage2.png").convert("RGB"), "frozen exact latent parent"),
    ]
    for row in trajectory:
        sheet_items.append(
            (
                f"SR-MPGD i{int(row['iteration'])}",
                Image.open(str(row["image_path"])).convert("RGB"),
                (
                    f"SSR={int(row['qr_verify_exact_presets'])}/37 "
                    f"LPIPS={float(row['lpips']):.4f} safe={row['visual_guard_pass']}"
                ),
            )
        )
    sheet_items.append(
        (
            "FINAL selected",
            Image.open(selected_image_path).convert("RGB"),
            f"winner={winner['checkpoint']}",
        )
    )
    _comparison_sheet(pipeline_dir / "full-pipeline-contact-sheet.png", sheet_items, columns=4)

    pipeline_manifest = {
        "payload": payload,
        "prompt": prompt,
        "seed": source.get("seed"),
        "advisor_preview": advisor,
        "surrogate": surrogate_info,
        "stages": {
            "qr_reference": str(pipeline_dir / "01-qr-reference.png"),
            "control_condition": str(pipeline_dir / "02-control-condition.png"),
            "stage1": str(pipeline_dir / "03-stage1.png"),
            "stage2": str(pipeline_dir / "04-stage2.png"),
            "srmpgd_winner_trajectory": str(output_dir / winner_recipe / "images"),
            "final": str(selected_image_path),
            "final_latent": str(selected_latent_path),
        },
        "finalized_from_existing_checkpoints": True,
        "note": "advisor recommendation is prospective; E040 optimization stays on the frozen E035 Stage-2 parent",
    }
    _atomic_json(output_dir / "pipeline-manifest.json", pipeline_manifest)

    safe = [row for row in rows if row.get("visual_guard_pass") is True]
    verdict = {
        "experiment": EXPERIMENT,
        "gamma": 1000.0,
        "gamma_preserved": True,
        "radii": list(DEFAULT_RADII),
        "max_iterations": 8,
        "checkpoint_count": len(rows),
        "visual_safe_checkpoint_count": len(safe),
        "research_winner_checkpoint": winner["checkpoint"],
        "research_winner_recipe": winner["method"],
        "winner_iteration": winner["iteration"],
        "winner_radius": winner["radius"],
        "winner_ssr_exact_presets": winner["qr_verify_exact_presets"],
        "winner_ssr": winner["ssr"],
        "winner_original_exact": winner["original_exact"],
        "winner_visual_guard_checks": winner["visual_guard_checks"],
        "e039_control_ssr_exact_presets": e039_control["verdict"].get("winner_ssr_exact_presets"),
        "e016_surrogate_research_usable": bool(surrogate_info.get("research_usable")),
        "advisor_available": bool(advisor.get("available")),
        "finalized_from_existing_checkpoints": True,
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": "REVIEW_FULL_PIPELINE_AND_WINNER_THEN_DECIDE_GENERALIZATION",
    }
    _atomic_json(verdict_path, verdict)
    _atomic_text(
        output_dir / "report.md",
        (
            "# E040 — checkpoint frontier + final pipeline\n\n"
            "- gamma: **1000** (fixed)\n"
            f"- radii: {', '.join(map(str, DEFAULT_RADII))}\n"
            "- recovered/finalized from the already-computed 45 checkpoints: **yes**\n"
            f"- winner: **{verdict['research_winner_checkpoint']}**\n"
            f"- SSR: **{verdict['winner_ssr_exact_presets']}/37**\n"
            f"- E016 surrogate research-usable: **{verdict['e016_surrogate_research_usable']}**\n"
            f"- advisor available: **{verdict['advisor_available']}**\n"
            "- production ready: **no**\n"
        ),
    )
    _atomic_json(output_dir / "e040-artifact-manifest.json", _manifest(output_dir))
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--e039-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verdict = finalize_e040(
        parent_dir=args.parent_dir,
        e039_results_dir=args.e039_results_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
