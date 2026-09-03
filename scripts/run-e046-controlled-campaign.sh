#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e046-controlled-campaign.sh [run|resume|status|logs|verify|restore-runtime]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-run}"
profile="${PROOFTAG_E046_PROFILE:-smoke}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
output_root="${PROOFTAG_E046_OUTPUT_ROOT:-/data/e046-controlled-best-generator-v1}"
e045_root="${PROOFTAG_E045_OUTPUT_ROOT:-/data/e045-foundation-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
parent_timeout="${PROOFTAG_E046_PARENT_TIMEOUT_SECONDS:-21600}"
refinement_timeout="${PROOFTAG_E046_REFINEMENT_TIMEOUT_SECONDS:-21600}"
retry_failed_jobs="${PROOFTAG_E046_RETRY_FAILED:-0}"

require_repo() {
  for file in \
    deploy/k8s/e046-parent-job.yaml \
    deploy/k8s/e046-refinement-job.yaml \
    prooftag_qr/e046_catalog.py \
    prooftag_qr/e046_quiet_zone.py \
    prooftag_qr/e046_campaign.py; do
    [[ -f "$file" ]] || {
      echo "Fichier E046 absent : $file" >&2
      exit 1
    }
  done
}

ensure_api() {
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
}

stop_gpu_services() {
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
  "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0 >/dev/null || true
}

running_e046_jobs() {
  "$kubectl_bin" get jobs -n "$namespace" \
    -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' \
    2>/dev/null \
    | awk '($2 + 0) > 0 { print $1 }' \
    || true
}

restore_runtime_if_idle() {
  if [[ -n "$(running_e046_jobs)" ]]; then
    echo "Un Job E046 est encore actif : API laissée à 0 pour ne pas voler la RTX." >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null 2>&1 || true
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s >/dev/null 2>&1 || true
  echo "Runtime QR restauré : API=1, notebook=0, vLLM=0."
}

show_status() {
  echo "===== JOBS E046 ====="
  "$kubectl_bin" get jobs -n "$namespace" \
    -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
    -o wide 2>/dev/null || true
  echo
  echo "===== PODS E046 ====="
  "$kubectl_bin" get pods -n "$namespace" \
    -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
    -o wide 2>/dev/null || true

  active="$(running_e046_jobs)"
  if [[ -n "$active" ]]; then
    echo
    echo "===== JOB ACTIF ====="
    printf '%s\n' "$active"
    echo "L'API reste arrêtée pour laisser la RTX au Job."
    return
  fi

  echo
  echo "===== ETAT PERSISTANT ====="
  ensure_api
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python -m prooftag_qr.e046_campaign status \
      --output-root "$output_root" || true
}

show_logs() {
  jobs="$(
    "$kubectl_bin" get jobs -n "$namespace" \
      -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
      --sort-by=.metadata.creationTimestamp \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true
  )"
  if [[ -z "$jobs" ]]; then
    echo "Aucun Job E046."
    return
  fi
  while IFS= read -r job; do
    [[ -n "$job" ]] || continue
    echo
    echo "===== $job ====="
    "$kubectl_bin" logs -n "$namespace" job/"$job" \
      --all-containers=true --tail=250 2>/dev/null || true
  done <<<"$jobs"
}

case "$action" in
  status)
    require_repo
    show_status
    exit 0
    ;;
  logs)
    require_repo
    show_logs
    exit 0
    ;;
  verify)
    require_repo
    if [[ -n "$(running_e046_jobs)" ]]; then
      echo "Un Job E046 est actif : ne pas démarrer l'API sur la RTX." >&2
      exit 1
    fi
    ensure_api
    "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
      python -m prooftag_qr.e046_campaign verify \
        --output-root "$output_root"
    exit 0
    ;;
  restore-runtime)
    require_repo
    restore_runtime_if_idle
    exit $?
    ;;
  run|resume)
    ;;
  *)
    echo "Action inconnue : $action" >&2
    exit 2
    ;;
esac

require_repo
if [[ "$profile" != "smoke" && "$profile" != "pilot" && "$profile" != "full" ]]; then
  echo "PROOFTAG_E046_PROFILE doit être smoke, pilot ou full." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E046." >&2
  exit 1
fi

existing_active_jobs="$(running_e046_jobs)"
if [[ -n "$existing_active_jobs" ]]; then
  echo "Un Job E046 est déjà actif :" >&2
  printf '%s\n' "$existing_active_jobs" >&2
  echo "Ne pas démarrer l'API sur la RTX. Utiliser status, attendre la fin, puis resume." >&2
  exit 1
fi

source_commit="$(git rev-parse HEAD)"
ensure_api
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
if [[ ! "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest runtime API invalide : $image_digest" >&2
  exit 1
fi
if [[ "$deployed_commit" != "$source_commit" ]]; then
  echo "Image API et dépôt désynchronisés : API=$deployed_commit Git=$source_commit" >&2
  echo "Lancer : bash scripts/deploy-e046-notebooks.sh" >&2
  exit 1
fi

echo "===== E046 PLAN ====="
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign plan \
    --output-root "$output_root" \
    --e045-root "$e045_root" \
    --profile "$profile" \
    --source-commit "$source_commit"

plan_id="$(
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python - "$output_root/LATEST.json" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["plan_id"])
PY
)"
echo "PLAN_ID=$plan_id"
echo "PROFILE=$profile"
echo "IMAGE=$image"
echo "COMMIT=$source_commit"

parent_list="$(mktemp)"
refinement_list="$(mktemp)"
cleanup() {
  rm -f "$parent_list" "$refinement_list"
}

on_exit() {
  local exit_code="$?"
  cleanup
  if [[ -z "$(running_e046_jobs)" ]]; then
    restore_runtime_if_idle >/dev/null 2>&1 || true
  else
    echo "Un Job E046 est encore actif : API laissée à 0 pour ne pas voler la RTX." >&2
    echo "Après la fin du Job : bash scripts/run-e046-controlled-campaign.sh restore-runtime" >&2
  fi
  return "$exit_code"
}
trap on_exit EXIT

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign list-parents \
    --output-root "$output_root" \
    --plan-id "$plan_id" \
    --pending-only >"$parent_list"

stop_gpu_services

wait_job() {
  local job="$1"
  local timeout="$2"
  local started elapsed active succeeded failed
  started="$(date +%s)"
  while true; do
    active="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
    succeeded="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    elapsed="$(( $(date +%s) - started ))"
    printf '[E046:%s] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
      "$job" "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
    if [[ "${succeeded:-0}" -ge 1 ]]; then
      return 0
    fi
    if [[ "${failed:-0}" -ge 1 ]]; then
      echo "Job $job en échec. Les tentatives et checkpoints restent dans /data." >&2
      "$kubectl_bin" logs -n "$namespace" job/"$job" \
        --all-containers=true --tail=1200 || true
      return 1
    fi
    if [[ "$elapsed" -ge "$timeout" ]]; then
      echo "Timeout opérateur $job après ${elapsed}s. Le Job n'est pas supprimé." >&2
      return 1
    fi
    sleep 30
  done
}

start_or_attach_job() {
  local kind="$1"
  local task_id="$2"
  local recipe_id="${3:-}"
  local hash job template tmp active succeeded failed exists
  hash="$(printf '%s' "${kind}:${task_id}:${recipe_id}:${plan_id}" | sha256sum | cut -c1-12)"
  job="prooftag-qr-e046-${kind:0:1}-${hash}"
  template="deploy/k8s/e046-parent-job.yaml"
  [[ "$kind" == "refinement" ]] && template="deploy/k8s/e046-refinement-job.yaml"

  exists=0
  if "$kubectl_bin" get job "$job" -n "$namespace" >/dev/null 2>&1; then
    exists=1
  fi
  active="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  succeeded="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$("$kubectl_bin" get job "$job" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"

  if [[ "${active:-0}" -ge 1 ]]; then
    echo "Job existant actif : $job"
  elif [[ "${failed:-0}" -ge 1 ]]; then
    echo "Le Job $job a déjà échoué avec cette spécification." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job"       --all-containers=true --tail=300 >&2 || true
    if [[ "$retry_failed_jobs" != "1" ]]; then
      echo "Aucune relance identique automatique." >&2
      echo "Après diagnostic d'un incident réellement transitoire seulement :" >&2
      echo "  PROOFTAG_E046_RETRY_FAILED=1 PROOFTAG_E046_PROFILE=$profile \\" >&2
      echo "  bash scripts/run-e046-controlled-campaign.sh resume" >&2
      echo "Pour un OOM, une erreur de contrat ou de données : corriger la spécification dans un nouveau commit/plan." >&2
      return 1
    fi
    echo "Relance explicite autorisée par PROOFTAG_E046_RETRY_FAILED=1."
    "$kubectl_bin" delete job "$job" -n "$namespace" --ignore-not-found >/dev/null
    exists=0
  elif [[ "${succeeded:-0}" -ge 1 ]]; then
    # La liste pending ne devrait jamais contenir une tâche dont le Job a
    # réussi. Cela indique une promotion absente/incomplète : ne pas recalculer
    # sans diagnostic.
    echo "Job $job réussi mais marqueur GENERATION_COMPLETE absent." >&2
    echo "État incohérent : tentative conservée, relance automatique refusée." >&2
    return 1
  elif [[ "$exists" -eq 1 ]]; then
    echo "Job existant en attente de planification : $job"
  fi

  if [[ "$exists" -eq 0 ]]; then
    tmp="$(mktemp)"
    sed \
      -e "s|__JOB_NAME__|$job|g" \
      -e "s|__NAMESPACE__|$namespace|g" \
      -e "s|__IMAGE__|$image|g" \
      -e "s|__OUTPUT_ROOT__|$output_root|g" \
      -e "s|__PLAN_ID__|$plan_id|g" \
      -e "s|__CANDIDATE_ID__|$task_id|g" \
      -e "s|__RECIPE_ID__|$recipe_id|g" \
      -e "s|__SOURCE_COMMIT__|$source_commit|g" \
      -e "s|__IMAGE_DIGEST__|$image_digest|g" \
      "$template" >"$tmp"
    "$kubectl_bin" apply -f "$tmp" >/dev/null
    rm -f "$tmp"
  fi

  if [[ "$kind" == "parent" ]]; then
    wait_job "$job" "$parent_timeout"
  else
    wait_job "$job" "$refinement_timeout"
  fi
  "$kubectl_bin" logs -n "$namespace" job/"$job" \
    --all-containers=true --tail=250 || true
}

echo "===== E046 PARENTS GPU ====="
while IFS= read -r candidate_id; do
  [[ -n "$candidate_id" ]] || continue
  echo
  echo "----- parent $candidate_id -----"
  start_or_attach_job "parent" "$candidate_id"
done <"$parent_list"

echo "===== E046 SCORING PARENTS CPU ====="
ensure_api
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign score-parents \
    --output-root "$output_root" \
    --plan-id "$plan_id"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign select \
    --output-root "$output_root" \
    --plan-id "$plan_id"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign list-refinements \
    --output-root "$output_root" \
    --plan-id "$plan_id" \
    --pending-only >"$refinement_list"

stop_gpu_services

echo "===== E046 SR-MPGD GPU ====="
while IFS=$'\t' read -r candidate_id recipe_id; do
  [[ -n "$candidate_id" && -n "$recipe_id" ]] || continue
  echo
  echo "----- refinement $candidate_id / $recipe_id -----"
  start_or_attach_job "refinement" "$candidate_id" "$recipe_id"
done <"$refinement_list"

echo "===== E046 SCORING + AGRÉGATION CPU ====="
ensure_api
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign score-refinements \
    --output-root "$output_root" \
    --plan-id "$plan_id"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign aggregate \
    --output-root "$output_root" \
    --plan-id "$plan_id"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign verify \
    --output-root "$output_root" \
    --plan-id "$plan_id"

trap - EXIT
cleanup
restore_runtime_if_idle >/dev/null

echo "===== E046 TERMINÉ ====="
echo "Plan    : $plan_id"
echo "Résultat: $output_root/$plan_id"
echo "Main    : .\\scripts\\e046-remote.ps1"
echo "Atlas   : .\\scripts\\e046-remote.ps1 -Atlas"
