#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/recover-e044-quality-scoring.sh" >&2
  return 2
fi
set -Eeuo pipefail

ns="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
root="${PROOFTAG_E044_RESULTS_ROOT:-/data/e044-multi-prompt-best-pipeline-v1}"
prompt_id="${1:-p01_greenhouse}"
k="${KUBECTL:-kubectl}"

[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  exit 1
}

current_commit="$(git rev-parse HEAD)"
"$k" scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
"$k" rollout status deployment/"$api" -n "$ns" --timeout=1200s
image="$("$k" get deployment "$api" -n "$ns" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"

echo "===== E044 RECOVERY — QUALITY SCORING ONLY ====="
echo "Prompt      : $prompt_id"
echo "Root        : $root"
echo "Image       : $image"
echo "Finalizer   : $current_commit"
echo "SR-MPGD     : AUCUN RECALCUL"
echo "Stage1/2    : AUCUN RECALCUL"

"$k" exec -i -n "$ns" deployment/"$api" -c api -- \
  python - "$root" "$prompt_id" "$current_commit" <<'PY'
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

from prooftag_qr.config import Settings
from prooftag_qr.diffqrcoder_backend import UpstreamDiffQRCoderBackend
from prooftag_qr.e035_loss_fidelity import _atomic_json, _atomic_text
from prooftag_qr.e035_parent_capture import _settings_document
from prooftag_qr.e040_checkpoint_frontier import CheckpointResult
from prooftag_qr.e044_multiprompt_best_pipeline import (
    ERROR_CORRECTION,
    EXPECTED_CHECKPOINTS_PER_PROMPT,
    EXPERIMENT,
    GAMMAS,
    LATENT_RADIUS_RMS,
    MAX_ITERATIONS,
    PAYLOAD,
    QR_MASK_PATTERN,
    QR_MODULE_SIZE,
    QR_VERSION,
    _capture_config,
    _gamma_recipe,
    _prompt,
    _rank_key,
    _score_prompt,
)
from prooftag_qr.quality_scoring import _installed_distribution_source
from prooftag_qr.qr import generate_diffqrcoder_qr

root = Path(sys.argv[1])
prompt_id = sys.argv[2]
finalizer_commit = sys.argv[3]
prompt_item = _prompt(prompt_id)

version, revision, error = _installed_distribution_source("hpsv2")
if error:
    raise RuntimeError(f"HPS provenance still invalid: {error}")
if version != "1.2.0":
    raise RuntimeError(f"unexpected HPS version: {version}")
if revision != "866735ecaae999fa714bd9edfa05aa2672669ee3":
    raise RuntimeError(f"unexpected HPS source revision: {revision}")
print("HPS provenance OK:", version, revision)

final_dir = root / "prompts" / prompt_id
if (final_dir / "COMPLETE.json").is_file():
    print("Prompt already complete:", final_dir)
    print((final_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    raise SystemExit(0)
if final_dir.exists():
    raise RuntimeError(
        f"canonical prompt dir exists without COMPLETE marker; preserve and inspect: {final_dir}"
    )

candidates = sorted(
    [
        path
        for path in (root / "attempts").glob(f"{prompt_id}-*")
        if path.is_dir()
    ],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not candidates:
    raise FileNotFoundError(f"no failed E044 attempt found for {prompt_id}")

def load_recoverable_attempt(attempt: Path):
    plan_path = attempt / "plan.json"
    parent_exact = attempt / "parent/stage2-exact-qz.png"
    parent_latent = attempt / "parent/stage2-latent.safetensors"
    if not plan_path.is_file() or not parent_exact.is_file() or not parent_latent.is_file():
        return None

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment") != EXPERIMENT or plan.get("prompt_id") != prompt_id:
        return None
    if tuple(float(x) for x in plan.get("gamma_grid", [])) != tuple(GAMMAS):
        return None
    if not math.isclose(float(plan.get("latent_radius_rms")), LATENT_RADIUS_RMS):
        return None
    if int(plan.get("max_iterations")) != MAX_ITERATIONS:
        return None

    checkpoints = []
    for gamma in GAMMAS:
        recipe = _gamma_recipe(prompt_id, gamma)
        recipe_root = attempt / "trajectories" / recipe.name
        trace_path = recipe_root / "trace.json"
        if not trace_path.is_file():
            return None
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        by_iteration = {int(row["iteration"]): row for row in trace}
        if sorted(by_iteration) != list(range(MAX_ITERATIONS + 1)):
            return None
        for iteration in range(MAX_ITERATIONS + 1):
            image_path = recipe_root / "images" / f"iteration-{iteration:03d}.png"
            latent_path = recipe_root / "latents" / f"iteration-{iteration:03d}.safetensors"
            if not image_path.is_file() or not latent_path.is_file():
                return None
            with Image.open(image_path) as opened:
                if opened.size != (736, 736):
                    return None
            step = dict(by_iteration[iteration])
            if not math.isclose(float(step["gamma"]), float(gamma)):
                return None
            checkpoints.append(
                CheckpointResult(
                    recipe=recipe.name,
                    iteration=iteration,
                    image_path=str(image_path),
                    latent_path=str(latent_path),
                    trace_step=step,
                )
            )
    if len(checkpoints) != EXPECTED_CHECKPOINTS_PER_PROMPT:
        return None
    return plan, checkpoints

selected = None
for candidate in candidates:
    loaded = load_recoverable_attempt(candidate)
    if loaded is not None:
        selected = (candidate, *loaded)
        break

if selected is None:
    raise RuntimeError(
        "failed attempts exist, but none contains the complete 18-checkpoint E044 state"
    )

attempt, plan, checkpoints = selected
compute_commit = str(plan.get("source_commit") or "")
print("Recovering complete scientific state:", attempt)
print("Compute commit:", compute_commit)
print("Finalizer commit:", finalizer_commit)
print("Checkpoint count:", len(checkpoints))

config = _capture_config(prompt_item["text"])
base = Settings()
settings = Settings.model_validate({**base.model_dump(), **_settings_document(config)})
backend = UpstreamDiffQRCoderBackend(settings)
blueprint = generate_diffqrcoder_qr(
    PAYLOAD,
    ERROR_CORRECTION,
    version=QR_VERSION,
    mask_pattern=QR_MASK_PATTERN,
    module_size=QR_MODULE_SIZE,
)
parent_exact = Image.open(attempt / "parent/stage2-exact-qz.png").convert("RGB")

# This is the phase that failed previously. No diffusion pipeline is loaded here.
rows = _score_prompt(
    output_dir=attempt,
    prompt_item=prompt_item,
    backend=backend,
    blueprint=blueprint,
    checkpoints=checkpoints,
    parent_exact=parent_exact,
)

safe = [row for row in rows if bool(row.get("visual_guard_pass"))]
if not safe:
    raise RuntimeError("recovered E044 prompt produced no visually-safe candidate")
winner = sorted(safe, key=_rank_key)[0]
raw_best = sorted(rows, key=_rank_key)[0]

best_per_gamma = []
for gamma in GAMMAS:
    options = [
        row
        for row in rows
        if math.isclose(float(row.get("gamma", 0.0)), float(gamma))
        and bool(row.get("visual_guard_pass"))
    ]
    if options:
        best_per_gamma.append(sorted(options, key=_rank_key)[0])
_atomic_json(attempt / "best-per-gamma.json", best_per_gamma)

pipeline_dir = attempt / "pipeline"
pipeline_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(attempt / "parent/stage1.png", pipeline_dir / "01-stage1.png")
shutil.copy2(attempt / "parent/stage2.png", pipeline_dir / "02-stage2.png")
shutil.copy2(
    attempt / "parent/stage2-exact-qz.png",
    pipeline_dir / "03-stage2-exact-qz.png",
)
shutil.copy2(Path(str(winner["image_path"])), pipeline_dir / "99-FINAL-QR.png")
shutil.copy2(
    Path(str(winner["latent_path"])),
    pipeline_dir / "99-FINAL-latent.safetensors",
)
_atomic_json(
    pipeline_dir / "99-FINAL-metadata.json",
    {
        "prompt_id": prompt_id,
        "winner_variant": winner["variant"],
        "gamma": winner["gamma"],
        "iteration": winner["iteration"],
        "ssr_exact_presets": winner["qr_verify_exact_presets"],
        "original_exact": winner["original_exact"],
        "visual_guard_pass": winner["visual_guard_pass"],
        "recovered_from_complete_pre_scoring_attempt": True,
        "compute_source_commit": compute_commit,
        "finalizer_source_commit": finalizer_commit,
    },
)

verdict = {
    "experiment": EXPERIMENT,
    "prompt_id": prompt_id,
    "prompt_family": prompt_item["family"],
    "prompt": prompt_item["text"],
    "seed": int(plan["seed"]),
    "source_commit": compute_commit,
    "finalizer_source_commit": finalizer_commit,
    "recovered_from_complete_pre_scoring_attempt": True,
    "recovery_reason": "HPS_PEP610_PROVENANCE_ONLY",
    "checkpoint_count": len(checkpoints),
    "scored_image_count": len(rows),
    "safe_image_count": len(safe),
    "gamma_grid": list(GAMMAS),
    "winner_variant": winner["variant"],
    "winner_gamma": float(winner["gamma"]),
    "winner_iteration": int(winner["iteration"]),
    "winner_ssr_exact_presets": int(winner["qr_verify_exact_presets"]),
    "winner_ssr": float(winner["ssr"]),
    "winner_original_exact": bool(winner["original_exact"]),
    "winner_lpips": float(winner["lpips"]),
    "winner_clip_score": winner.get("clip_score"),
    "winner_clip_aesthetic": winner.get("clip_aesthetic"),
    "winner_hpsv2_1": winner.get("hpsv2_1"),
    "winner_full_module_error_count": int(winner["full_module_error_count"]),
    "winner_visual_guard_pass": bool(winner["visual_guard_pass"]),
    "raw_best_variant": raw_best["variant"],
    "raw_best_ssr_exact_presets": int(raw_best["qr_verify_exact_presets"]),
    "exact_quiet_zone_geometry": True,
    "production_ready": False,
    "generalization_authorized": False,
}
_atomic_json(attempt / "verdict.json", verdict)
_atomic_text(
    attempt / "report.md",
    "\n".join(
        [
            f"# E044 recovered — {prompt_id}",
            "",
            f"- scientific compute commit: `{compute_commit}`",
            f"- scoring/finalizer commit: `{finalizer_commit}`",
            f"- checkpoints reused: **{len(checkpoints)}**",
            "- Stage1/Stage2/SR-MPGD recomputed: **NO**",
            f"- winner: `{winner['variant']}`",
            f"- SSR: **{winner['qr_verify_exact_presets']}/37**",
            f"- original exact: **{winner['original_exact']}**",
            f"- visual guard: **{winner['visual_guard_pass']}**",
            "",
            "Recovery changed provenance/scoring infrastructure only.",
        ]
    ),
)
_atomic_json(attempt / "COMPLETE.json", verdict)

final_dir.parent.mkdir(parents=True, exist_ok=True)
os.replace(attempt, final_dir)
print("RECOVERY COMPLETE:", final_dir)
print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
PY

"$k" exec -n "$ns" deployment/"$api" -c api -- \
  test -f "$root/prompts/$prompt_id/COMPLETE.json"

echo "===== E044 PROMPT RECUPERE SANS RECALCUL GPU ====="
echo "Tu peux maintenant relancer :"
echo "  bash scripts/run-e044-multiprompt-benchmark.sh"
