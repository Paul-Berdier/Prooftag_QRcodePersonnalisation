#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/resume-e046-terminal-srl.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
output_root="${PROOFTAG_E046_OUTPUT_ROOT:-/data/e046-controlled-best-generator-v1}"
plan_id="${PROOFTAG_E046_PLAN_ID:-}"
kubectl_bin="${KUBECTL:-kubectl}"
timeout_seconds="${PROOFTAG_E046_REFINEMENT_TIMEOUT_SECONDS:-21600}"
terminal_pattern="local upstream SRL port diverged from the pinned official class"

require_file() {
  [[ -f "$1" ]] || {
    echo "Fichier requis absent : $1" >&2
    exit 1
  }
}

require_file deploy/k8s/e046-refinement-job.yaml

running_jobs() {
  "$kubectl_bin" get jobs -n "$namespace" \
    -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' \
    2>/dev/null \
    | awk '($2 + 0) > 0 { print $1 }' \
    || true
}

ensure_api() {
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" \
    -n "$namespace" --timeout=1200s
}

stop_gpu_services() {
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment vllm \
    -n vllm --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=0 >/dev/null 2>&1 || true
}

restore_runtime() {
  if [[ -n "$(running_jobs)" ]]; then
    echo "Un Job E046 est encore actif : l'API reste à 0 pour ne pas prendre la RTX." >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment vllm \
    -n vllm --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=1 >/dev/null 2>&1 || true
  "$kubectl_bin" rollout status deployment/"$api_deployment" \
    -n "$namespace" --timeout=1200s >/dev/null 2>&1 || true
}

on_exit() {
  local code="$?"
  rm -f "${pending_file:-}" "${terminal_file:-}" 2>/dev/null || true
  if [[ -z "$(running_jobs)" ]]; then
    restore_runtime >/dev/null 2>&1 || true
  fi
  return "$code"
}
trap on_exit EXIT

active="$(running_jobs)"
if [[ -n "$active" ]]; then
  echo "Un Job E046 est déjà actif :" >&2
  printf '%s\n' "$active" >&2
  echo "Attendre sa fin avant de lancer le rescue." >&2
  exit 1
fi

ensure_api

if [[ -z "$plan_id" ]]; then
  plan_id="$(
    "$kubectl_bin" exec -i -n "$namespace" \
      deployment/"$api_deployment" -c api -- \
      python - "$output_root/LATEST.json" <<'PY'
import json
import sys
from pathlib import Path

latest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(latest["plan_id"])
PY
  )"
fi

readarray -t identity < <(
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python - "$output_root" "$plan_id" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan_id = sys.argv[2]
plan_dir = root / plan_id
plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
print(plan_dir)
print(plan["source_commit"])
print(plan["profile"])
PY
)

plan_dir="${identity[0]}"
plan_source_commit="${identity[1]}"
profile="${identity[2]}"

image="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
)"
deployed_commit="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
)"
image_digest="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}'
)"

if [[ "$deployed_commit" != "$plan_source_commit" ]]; then
  echo "Le plan et l'image déployée ne correspondent pas." >&2
  echo "Plan  : $plan_source_commit" >&2
  echo "Image : $deployed_commit" >&2
  echo "Ce rescue doit utiliser exactement l'image qui a créé le plan." >&2
  exit 1
fi
if [[ ! "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest OCI invalide : $image_digest" >&2
  exit 1
fi

echo "===== E046 RESCUE ====="
echo "Plan        : $plan_id"
echo "Profil      : $profile"
echo "Plan dir    : $plan_dir"
echo "Image       : $image"
echo "Commit plan : $plan_source_commit"
echo
echo "La tolérance SRL n'est PAS relâchée."
echo "Les branches dont le port local diverge de la classe officielle sont mises en quarantaine."

terminal_file="$(mktemp)"
pending_file="$(mktemp)"

classify_fidelity_failures() {
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python - "$output_root" "$plan_id" "$terminal_pattern" <<'PY'
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

root = Path(sys.argv[1])
plan_id = sys.argv[2]
pattern = sys.argv[3]
plan_dir = root / plan_id
terminal_root = plan_dir / "terminal-refinements"
terminal_root.mkdir(parents=True, exist_ok=True)

found = {}
for path in sorted((plan_dir / "failures").glob("*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if payload.get("task_kind") != "refinement":
        continue
    error = str(payload.get("error") or "")
    if pattern not in error:
        continue
    task_id = str(payload.get("task_id") or "")
    if "__" not in task_id:
        continue
    candidate_id, recipe_id = task_id.rsplit("__", 1)
    key = (candidate_id, recipe_id)
    marker = terminal_root / candidate_id / f"{recipe_id}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    terminal = {
        "schema": "e046-terminal-refinement-v1",
        "experiment": "e046-controlled-best-generator-v1",
        "plan_id": plan_id,
        "candidate_id": candidate_id,
        "srmpgd_recipe_id": recipe_id,
        "classification": "scientific_fidelity_mismatch",
        "retryable": False,
        "usable": False,
        "reason": error,
        "source_failure_file": str(path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, marker)
    found[key] = marker

for candidate_id, recipe_id in sorted(found):
    print(f"{candidate_id}\t{recipe_id}")
PY
}

classify_fidelity_failures >"$terminal_file"

if [[ ! -s "$terminal_file" ]]; then
  echo "Aucune divergence SRL officielle connue à mettre en quarantaine." >&2
  exit 1
fi

echo
echo "===== BRANCHES TERMINALES QUARANTAINÉES ====="
cat "$terminal_file"

"$kubectl_bin" exec -i -n "$namespace" \
  deployment/"$api_deployment" -c api -- \
  python - "$output_root" "$plan_id" <<'PY' >"$pending_file"
import json
import sys
from pathlib import Path

from prooftag_qr.e046_campaign import load_plan, refinement_tasks

root = Path(sys.argv[1])
plan_id = sys.argv[2]
plan_dir, plan = load_plan(root, plan_id)

for candidate_id, recipe_id in refinement_tasks(plan_dir, plan):
    complete = (
        plan_dir
        / "refinements"
        / candidate_id
        / recipe_id
        / "GENERATION_COMPLETE.json"
    )
    terminal = (
        plan_dir
        / "terminal-refinements"
        / candidate_id
        / f"{recipe_id}.json"
    )
    if complete.is_file() or terminal.is_file():
        continue
    print(f"{candidate_id}\t{recipe_id}")
PY

pending_count="$(grep -c . "$pending_file" 2>/dev/null || true)"
echo
echo "Refinements restant à générer : ${pending_count:-0}"

stop_gpu_services

wait_job() {
  local job="$1"
  local started elapsed active succeeded failed
  started="$(date +%s)"
  while true; do
    active="$("$kubectl_bin" get job "$job" -n "$namespace" \
      -o jsonpath='{.status.active}' 2>/dev/null || true)"
    succeeded="$("$kubectl_bin" get job "$job" -n "$namespace" \
      -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$("$kubectl_bin" get job "$job" -n "$namespace" \
      -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    elapsed="$(( $(date +%s) - started ))"

    printf '[E046-rescue:%s] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
      "$job" "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"

    if [[ "${succeeded:-0}" -ge 1 ]]; then
      return 0
    fi
    if [[ "${failed:-0}" -ge 1 ]]; then
      return 1
    fi
    if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
      echo "Timeout opérateur. Le Job n'est pas supprimé." >&2
      return 2
    fi
    sleep 30
  done
}

mark_terminal_from_job() {
  local candidate_id="$1"
  local recipe_id="$2"
  local job="$3"
  local log_file
  log_file="$(mktemp)"
  "$kubectl_bin" logs -n "$namespace" job/"$job" \
    --all-containers=true --tail=2000 >"$log_file" 2>&1 || true

  if ! grep -Fq "$terminal_pattern" "$log_file"; then
    cat "$log_file" >&2
    rm -f "$log_file"
    return 1
  fi

  ensure_api
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python - "$output_root" "$plan_id" "$candidate_id" "$recipe_id" "$job" <<'PY'
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

root = Path(sys.argv[1])
plan_id = sys.argv[2]
candidate_id = sys.argv[3]
recipe_id = sys.argv[4]
job = sys.argv[5]
plan_dir = root / plan_id

failures = []
for path in sorted((plan_dir / "failures").glob("*.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if payload.get("task_id") == f"{candidate_id}__{recipe_id}":
        failures.append((path, payload))

if not failures:
    raise RuntimeError("No persisted E046 failure record for the failed Job")

path, payload = failures[-1]
marker = (
    plan_dir
    / "terminal-refinements"
    / candidate_id
    / f"{recipe_id}.json"
)
marker.parent.mkdir(parents=True, exist_ok=True)
terminal = {
    "schema": "e046-terminal-refinement-v1",
    "experiment": "e046-controlled-best-generator-v1",
    "plan_id": plan_id,
    "candidate_id": candidate_id,
    "srmpgd_recipe_id": recipe_id,
    "classification": "scientific_fidelity_mismatch",
    "retryable": False,
    "usable": False,
    "reason": payload.get("error"),
    "source_failure_file": str(path),
    "kubernetes_job": job,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
tmp = marker.with_suffix(".json.tmp")
tmp.write_text(
    json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(tmp, marker)
print(json.dumps(terminal, ensure_ascii=False, indent=2))
PY
  stop_gpu_services
  rm -f "$log_file"
  return 0
}

start_refinement() {
  local candidate_id="$1"
  local recipe_id="$2"
  local hash job tmp existing_failed existing_succeeded

  hash="$(
    printf '%s' "refinement:${candidate_id}:${recipe_id}:${plan_id}" \
      | sha256sum \
      | cut -c1-12
  )"
  job="prooftag-qr-e046-r-${hash}"

  existing_failed="$("$kubectl_bin" get job "$job" -n "$namespace" \
    -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  existing_succeeded="$("$kubectl_bin" get job "$job" -n "$namespace" \
    -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"

  if [[ "${existing_succeeded:-0}" -ge 1 ]]; then
    echo "Job déjà réussi : $job"
    return 0
  fi
  if [[ "${existing_failed:-0}" -ge 1 ]]; then
    echo "Job déjà en échec non terminal : $job" >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job" \
      --all-containers=true --tail=500 >&2 || true
    return 1
  fi

  if ! "$kubectl_bin" get job "$job" -n "$namespace" >/dev/null 2>&1; then
    tmp="$(mktemp)"
    sed \
      -e "s|__JOB_NAME__|$job|g" \
      -e "s|__NAMESPACE__|$namespace|g" \
      -e "s|__IMAGE__|$image|g" \
      -e "s|__OUTPUT_ROOT__|$output_root|g" \
      -e "s|__PLAN_ID__|$plan_id|g" \
      -e "s|__CANDIDATE_ID__|$candidate_id|g" \
      -e "s|__RECIPE_ID__|$recipe_id|g" \
      -e "s|__SOURCE_COMMIT__|$plan_source_commit|g" \
      -e "s|__IMAGE_DIGEST__|$image_digest|g" \
      deploy/k8s/e046-refinement-job.yaml >"$tmp"
    "$kubectl_bin" apply -f "$tmp" >/dev/null
    rm -f "$tmp"
  fi

  if wait_job "$job"; then
    "$kubectl_bin" logs -n "$namespace" job/"$job" \
      --all-containers=true --tail=200 || true
    return 0
  fi

  if mark_terminal_from_job "$candidate_id" "$recipe_id" "$job"; then
    echo "Branche $candidate_id / $recipe_id mise en quarantaine ; poursuite automatique."
    return 0
  fi

  echo "Échec non classé : arrêt sans supprimer les résultats déjà produits." >&2
  return 1
}

echo
echo "===== POURSUITE DES REFINEMENTS ====="
while IFS=$'\t' read -r candidate_id recipe_id; do
  [[ -n "$candidate_id" && -n "$recipe_id" ]] || continue
  echo
  echo "----- $candidate_id / $recipe_id -----"
  start_refinement "$candidate_id" "$recipe_id"
done <"$pending_file"

ensure_api

echo
echo "===== SCORING ET AGRÉGATION HORS BRANCHES TERMINALES ====="
"$kubectl_bin" exec -i -n "$namespace" \
  deployment/"$api_deployment" -c api -- \
  python - "$output_root" "$plan_id" <<'PY'
import json
import sys
from pathlib import Path

import prooftag_qr.e046_campaign as campaign
from prooftag_qr.resilient_experiment import atomic_write_json

root = Path(sys.argv[1])
plan_id = sys.argv[2]
plan_dir, plan = campaign.load_plan(root, plan_id)

all_tasks = campaign.refinement_tasks(plan_dir, plan)
terminal = {}
for path in sorted((plan_dir / "terminal-refinements").glob("*/*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = (
        str(payload["candidate_id"]),
        str(payload["srmpgd_recipe_id"]),
    )
    terminal[key] = payload

completed = []
pending = []
for candidate_id, recipe_id in all_tasks:
    complete = (
        plan_dir
        / "refinements"
        / candidate_id
        / recipe_id
        / "GENERATION_COMPLETE.json"
    )
    if complete.is_file():
        completed.append((candidate_id, recipe_id))
    elif (candidate_id, recipe_id) not in terminal:
        pending.append((candidate_id, recipe_id))

if pending:
    raise RuntimeError(
        "Non-terminal refinements are still missing: "
        + json.dumps(pending, ensure_ascii=False)
    )

terminal_summary = {
    "schema": "e046-terminal-refinements-summary-v1",
    "plan_id": plan_id,
    "expected_refinement_count": len(all_tasks),
    "completed_generation_count": len(completed),
    "terminal_failure_count": len(terminal),
    "terminal_failures": list(terminal.values()),
}
atomic_write_json(
    plan_dir / "TERMINAL_REFINEMENT_FAILURES.json",
    terminal_summary,
)

original_refinement_tasks = campaign.refinement_tasks

def filtered_refinement_tasks(plan_dir_arg, plan_arg):
    return [
        task
        for task in original_refinement_tasks(plan_dir_arg, plan_arg)
        if task not in terminal
    ]

campaign.refinement_tasks = filtered_refinement_tasks

scoring = campaign.score_all_refinements(
    output_root=root,
    plan_id=plan_id,
)
complete = campaign.aggregate(
    output_root=root,
    plan_id=plan_id,
)
verification = campaign.verify(
    output_root=root,
    plan_id=plan_id,
)

print(json.dumps({
    "terminal_summary": terminal_summary,
    "scoring": scoring,
    "complete": complete,
    "verification": verification,
}, ensure_ascii=False, indent=2, default=str))
PY

trap - EXIT
rm -f "$pending_file" "$terminal_file"
restore_runtime >/dev/null

echo
echo "===== E046 RESCUE TERMINÉ ====="
echo "Plan : $plan_id"
echo "La branche SRL divergente reste exclue et documentée."
echo "Toutes les générations valides restantes ont été scorées et agrégées."
