"""Aggregate completed E044 prompt jobs without loading diffusion models.

E044 prompt attempts are generated under /attempts/<attempt-id> and then atomically
moved to /prompts/<prompt-id>.  The scoring rows are created before that move, so
their absolute image/latent paths may still reference the old attempt location.
This finalizer resolves those paths against the canonical prompt directory before
using or exporting them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .e044_multiprompt_best_pipeline import EXPERIMENT, GAMMAS, PROMPTS, SEED


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row.get("qr_verify_exact_presets", 0)),
        -int(bool(row.get("original_exact"))),
        int(row.get("full_module_error_count", 10**9)),
        float(row.get("lpips", 1e9)),
        -float(row.get("clip_aesthetic") or -1e9),
        int(row.get("iteration", 10**9)),
    )


def _canonical_artifact_path(root: Path, row: dict[str, Any], field: str) -> Path:
    raw = Path(str(row[field]))
    if raw.is_file():
        return raw

    prompt_dir = root / "prompts" / str(row["prompt_id"])
    parts = raw.parts
    for marker in ("trajectories", "parent", "pipeline", "scoring"):
        if marker in parts:
            index = parts.index(marker)
            candidate = prompt_dir / Path(*parts[index:])
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"E044 cannot resolve stale {field} for {row.get('prompt_id')}/"
        f"{row.get('variant')}: {raw}"
    )


def _repair_row_paths(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(row)
    repaired["image_path"] = str(_canonical_artifact_path(root, repaired, "image_path"))
    repaired["latent_path"] = str(_canonical_artifact_path(root, repaired, "latent_path"))
    return repaired


def _sheet(path: Path, items: list[tuple[str, Image.Image, str]], columns: int = 4) -> None:
    if not items:
        return
    thumb = 320
    label = 58
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image, subtitle) in enumerate(items):
        row, col = divmod(index, columns)
        tile = image.convert("RGB").copy()
        tile.thumbnail((thumb - 12, thumb - 12), Image.Resampling.LANCZOS)
        x = col * thumb + (thumb - tile.width) // 2
        y = row * (thumb + label) + (thumb - tile.height) // 2
        sheet.paste(tile, (x, y))
        text_y = row * (thumb + label) + thumb
        draw.text((col * thumb + 8, text_y + 4), title[:42], fill="black")
        draw.text((col * thumb + 8, text_y + 26), subtitle[:48], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=False, compress_level=9)


def aggregate(*, root: Path, source_commit: str) -> dict[str, Any]:
    verdicts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for prompt in PROMPTS:
        prompt_dir = root / "prompts" / prompt["id"]
        complete = prompt_dir / "COMPLETE.json"
        comparison = prompt_dir / "scoring/comparison.json"
        if not complete.is_file() or not comparison.is_file():
            raise FileNotFoundError(f"E044 prompt incomplete: {prompt['id']}")

        verdict = json.loads(complete.read_text(encoding="utf-8"))
        original_rows = json.loads(comparison.read_text(encoding="utf-8"))
        rows = [_repair_row_paths(root, row) for row in original_rows]

        # Persist the repaired canonical paths so the notebooks do not retain dead
        # /attempts/... references after atomic promotion.
        _atomic_json(comparison, rows)

        verdicts.append(verdict)
        all_rows.extend(rows)
        safe = [row for row in rows if bool(row.get("visual_guard_pass"))]
        best = sorted(safe, key=_rank_key)[0]
        summaries.append(
            {
                "prompt_id": prompt["id"],
                "family": prompt["family"],
                "prompt": prompt["text"],
                "best_variant": best["variant"],
                "best_gamma": best["gamma"],
                "best_iteration": best["iteration"],
                "best_ssr_exact_presets": best["qr_verify_exact_presets"],
                "best_ssr": best["ssr"],
                "best_original_exact": best["original_exact"],
                "best_lpips": best["lpips"],
                "best_clip_score": best.get("clip_score"),
                "best_clip_aesthetic": best.get("clip_aesthetic"),
                "best_hpsv2_1": best.get("hpsv2_1"),
                "best_module_errors": best["full_module_error_count"],
                "best_image_path": best["image_path"],
            }
        )

    safe_all = [row for row in all_rows if bool(row.get("visual_guard_pass"))]
    winner = sorted(safe_all, key=_rank_key)[0]
    raw_best = sorted(all_rows, key=_rank_key)[0]

    _atomic_json(root / "prompt-verdicts.json", verdicts)
    _atomic_json(root / "prompt-summary.json", summaries)
    _atomic_json(root / "comparison-all.json", all_rows)

    csv_rows = []
    for row in all_rows:
        flat = dict(row)
        flat["visual_guard_checks"] = json.dumps(
            flat.get("visual_guard_checks") or {}, ensure_ascii=False, sort_keys=True
        )
        flat["decoder_diagnostics"] = json.dumps(
            flat.get("decoder_diagnostics") or {}, ensure_ascii=False, sort_keys=True
        )
        csv_rows.append(flat)
    _write_csv(root / "comparison-all.csv", csv_rows)

    gamma_summary = []
    for gamma in GAMMAS:
        subset = [
            row
            for row in all_rows
            if float(row.get("gamma", 0.0)) == float(gamma)
            and bool(row.get("visual_guard_pass"))
        ]
        gamma_summary.append(
            {
                "gamma": gamma,
                "safe_checkpoint_count": len(subset),
                "prompt_count_with_any_exact": len(
                    {
                        row["prompt_id"]
                        for row in subset
                        if int(row["qr_verify_exact_presets"]) > 0
                    }
                ),
                "max_ssr_exact_presets": max(
                    (int(row["qr_verify_exact_presets"]) for row in subset), default=0
                ),
                "mean_ssr": (
                    sum(float(row["ssr"]) for row in subset) / len(subset)
                    if subset
                    else 0.0
                ),
                "projection_active_count": sum(
                    bool(row.get("projection_was_active")) for row in subset
                ),
            }
        )
    _atomic_json(root / "gamma-summary.json", gamma_summary)

    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(winner["image_path"])), pipeline / "99-FINAL-QR.png")
    shutil.copy2(Path(str(winner["latent_path"])), pipeline / "99-FINAL-latent.safetensors")
    winner_prompt = root / "prompts" / str(winner["prompt_id"])
    shutil.copy2(winner_prompt / "parent/stage1.png", pipeline / "01-WINNER-stage1.png")
    shutil.copy2(winner_prompt / "parent/stage2.png", pipeline / "02-WINNER-stage2.png")
    shutil.copy2(
        winner_prompt / "parent/stage2-exact-qz.png",
        pipeline / "03-WINNER-stage2-exact-qz.png",
    )
    _sheet(
        pipeline / "best-by-prompt-contact-sheet.png",
        [
            (
                summary["prompt_id"],
                Image.open(summary["best_image_path"]).convert("RGB"),
                (
                    f"g={int(summary['best_gamma'])} "
                    f"i={summary['best_iteration']} "
                    f"SSR={summary['best_ssr_exact_presets']}/37"
                ),
            )
            for summary in summaries
        ],
    )

    verdict = {
        "experiment": EXPERIMENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "seed": SEED,
        "prompt_count": len(PROMPTS),
        "gamma_grid": list(GAMMAS),
        "checkpoint_count": sum(int(item["checkpoint_count"]) for item in verdicts),
        "scored_image_count": len(all_rows),
        "safe_image_count": len(safe_all),
        "prompts_with_any_qrverify_success": sum(
            int(summary["best_ssr_exact_presets"] > 0) for summary in summaries
        ),
        "winner_prompt_id": winner["prompt_id"],
        "winner_prompt_family": winner["prompt_family"],
        "winner_variant": winner["variant"],
        "winner_gamma": winner["gamma"],
        "winner_iteration": winner["iteration"],
        "winner_ssr_exact_presets": winner["qr_verify_exact_presets"],
        "winner_ssr": winner["ssr"],
        "winner_original_exact": winner["original_exact"],
        "winner_visual_guard_pass": winner["visual_guard_pass"],
        "winner_lpips": winner["lpips"],
        "winner_clip_score": winner.get("clip_score"),
        "winner_clip_aesthetic": winner.get("clip_aesthetic"),
        "winner_hpsv2_1": winner.get("hpsv2_1"),
        "raw_best_prompt_id": raw_best["prompt_id"],
        "raw_best_variant": raw_best["variant"],
        "raw_best_ssr_exact_presets": raw_best["qr_verify_exact_presets"],
        "paper_comparison_kind": (
            "documented methodological comparison; E044 is not claimed paper-exact"
        ),
        "authoritative_scanner": (
            "qr-verify@0.2.0 conservative 37-preset exact-payload scoring"
        ),
        "canonical_prompt_artifact_paths": True,
        "multi_prompt_screen": True,
        "multi_seed_generalization": False,
        "production_ready": False,
        "generalization_authorized": False,
        "next_action": (
            "REVIEW_COMPLETE_NOTEBOOK_AND_PROMPT_SENSITIVITY_BEFORE_ANY_NEW_LOSS"
        ),
    }
    _atomic_json(root / "verdict.json", verdict)
    (root / "report.md").write_text(
        (
            "# E044 — multi-prompt best-pipeline benchmark\n\n"
            f"- prompts: **{len(PROMPTS)}**\n"
            f"- checkpoints SR-MPGD: **{verdict['checkpoint_count']}**\n"
            f"- scored images: **{len(all_rows)}**\n"
            "- prompts with any QR-Verify success: "
            f"**{verdict['prompts_with_any_qrverify_success']}**\n"
            f"- winner: **{winner['prompt_id']} / {winner['variant']}**\n"
            f"- SSR: **{winner['qr_verify_exact_presets']}/37**\n\n"
            "E044 is a shared-seed prompt screen, not multi-seed generalization.\n"
            "Production and generalization remain false.\n"
        ),
        encoding="utf-8",
    )
    return verdict


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(root=args.root, source_commit=args.source_commit),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
