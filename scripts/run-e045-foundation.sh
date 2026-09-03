#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e045-foundation.sh [run|status|logs|verify]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-run}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E045_JOB_NAME:-prooftag-qr-e045}"
output_root="${PROOFTAG_E045_OUTPUT_ROOT:-/data/e045-foundation-v1}"
max_files="${PROOFTAG_E045_MAX_FILES:-200000}"
max_hash_mb="${PROOFTAG_E045_MAX_HASH_MB:-64}"
max_parse_mb="${PROOFTAG_E045_MAX_PARSE_MB:-64}"
kubectl_bin="${KUBECTL:-kubectl}"

require_repo() {
  [[ -f deploy/k8s/e045-foundation-job.yaml ]] || {
    echo "Lancer depuis la racine du dépôt." >&2
    exit 1
  }
}

ensure_api() {
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
}

api_status() {
  ensure_api
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python -m prooftag_qr.e045_foundation status --output-root "$output_root"
}

case "$action" in
  status)
    require_repo
    echo "===== JOB E045 ====="
    "$kubectl_bin" get job "$job_name" -n "$namespace" -o wide 2>/dev/null || true
    "$kubectl_bin" get pods -n "$namespace" -l "job-name=$job_name" -o wide 2>/dev/null || true
    echo
    echo "===== ETAT PERSISTANT ====="
    api_status || true
    exit 0
    ;;
  logs)
    require_repo
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=1000
    exit 0
    ;;
  verify)
    require_repo
    ensure_api
    "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
      python -m prooftag_qr.e045_foundation verify --output-root "$output_root"
    exit 0
    ;;
  run|resume)
    ;;
  *)
    echo "Action inconnue : $action" >&2
    echo "Utiliser run, resume, status, logs ou verify." >&2
    exit 2
    ;;
esac

require_repo
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E045." >&2
  exit 1
fi

ensure_api
source_commit="$(git rev-parse HEAD)"
image="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
)"
deployed_commit="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
)"
if [[ "$deployed_commit" != "$source_commit" ]]; then
  echo "L'image API ne correspond pas au commit courant." >&2
  echo "API=$deployed_commit Git=$source_commit" >&2
  echo "Lancer : bash scripts/deploy-e045-notebook.sh api" >&2
  exit 1
fi

echo "===== PREFLIGHT E045 ====="
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - <<'PY'
import tempfile
from pathlib import Path
from prooftag_qr.e045_registry import EXPERIMENTS
from prooftag_qr.e045_parameter_space import PARAMETERS
from prooftag_qr.e045_foundation import EXPERIMENT, run_resilience_selftest

assert len(EXPERIMENTS) == 45
assert len(PARAMETERS) >= 90
with tempfile.TemporaryDirectory(prefix="e045-preflight-") as tmp:
    result = run_resilience_selftest(Path(tmp))
    assert result["passed"] is True
print("E045 preflight OK:", EXPERIMENT, len(EXPERIMENTS), len(PARAMETERS))
PY

# Si un Job est encore actif, ne jamais le supprimer.
active="$(
  "$kubectl_bin" get job "$job_name" -n "$namespace" \
    -o jsonpath='{.status.active}' 2>/dev/null || true
)"
if [[ "${active:-0}" -ge 1 ]]; then
  echo "Un Job E045 est déjà actif. Aucun second Job n'est créé."
else
  # Un ancien objet Job terminé/échoué peut être supprimé; les données /data restent intactes.
  "$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found

  tmp_job="$(mktemp)"
  trap 'rm -f "$tmp_job"' EXIT
  sed \
    -e "s|__NAMESPACE__|$namespace|g" \
    -e "s|__JOB_NAME__|$job_name|g" \
    -e "s|__IMAGE__|$image|g" \
    -e "s|__OUTPUT_ROOT__|$output_root|g" \
    -e "s|__SOURCE_COMMIT__|$source_commit|g" \
    -e "s|__MAX_FILES__|$max_files|g" \
    -e "s|__MAX_HASH_MB__|$max_hash_mb|g" \
    -e "s|__MAX_PARSE_MB__|$max_parse_mb|g" \
    deploy/k8s/e045-foundation-job.yaml >"$tmp_job"
  "$kubectl_bin" apply -f "$tmp_job"
  rm -f "$tmp_job"
  trap - EXIT
fi

echo "===== E045 FONDATION REPRENABLE ====="
echo "Image      : $image"
echo "Commit     : $source_commit"
echo "Sortie     : $output_root"
echo "GPU        : aucun"
echo "Suppression /data : interdite"
echo "Même commande après crash : bash scripts/run-e045-foundation.sh"

started="$(date +%s)"
timeout_seconds="${PROOFTAG_E045_TIMEOUT_SECONDS:-21600}"
while true; do
  succeeded="$("$kubectl_bin" get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$("$kubectl_bin" get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$("$kubectl_bin" get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  elapsed="$(( $(date +%s) - started ))"
  printf '[E045] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
    "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"

  if [[ "${succeeded:-0}" -ge 1 ]]; then
    break
  fi
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E045 en échec. Les résultats partiels sont conservés." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=1000 || true
    echo
    echo "===== ETAT PERSISTANT APRES ECHEC =====" >&2
    api_status || true
    echo
    echo "Après correction ou incident transitoire, relancer la même commande." >&2
    echo "Un OOM/contrat reste BLOCKED et exige une nouvelle spécification." >&2
    exit 1
  fi
  if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
    echo "Timeout opérateur E045 après ${elapsed}s; le Job n'est pas supprimé." >&2
    "$kubectl_bin" get job,pod -n "$namespace" -l prooftag.io/experiment=e045-foundation-resilience-v1 -o wide || true
    exit 1
  fi
  sleep 30
done

echo "===== LOGS FINAUX E045 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=1000 || true

echo "===== VERIFICATION E045 ====="
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e045_foundation verify --output-root "$output_root"

echo "===== E045 TERMINÉ ====="
echo "Notebook : .\\scripts\\e045-remote.ps1"
