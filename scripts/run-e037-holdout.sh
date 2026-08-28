#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e037-holdout.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E037_JOB_NAME:-prooftag-qr-e037}"
e036_results_dir="${PROOFTAG_E037_E036_RESULTS_DIR:-/data/e036-gamma1000-trust-region-v1}"
results_dir="${PROOFTAG_E037_RESULTS_DIR:-/data/e037-prospective-mini-holdout-v1}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f deploy/k8s/e037-holdout-job.yaml ]] || {
  echo "Lancer depuis la racine du dépôt." >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E037." >&2
  exit 1
}

source_commit="$(git rev-parse HEAD)"
previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
if [[ "${previous_api_replicas:-0}" -lt 1 ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi

image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"

echo "===== PRÉCONDITION E036 ====="
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$e036_results_dir/verdict.json" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"verdict E036 absent: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["research_winner"] == "e036_gamma1000_global_trust", data
assert data["gamma"] == 1000.0, data
assert data["gamma_preserved"] is True, data
assert data["decision"] == "PREPARE_MINI_HOLDOUT_WITH_WINNER", data
print("E036 winner vérifié: e036_gamma1000_global_trust, gamma=1000")
PY

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c \
  "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E037 est déjà terminé: $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0
    ;;
  PARTIAL)
    echo "Sortie E037 partielle détectée: $results_dir" >&2
    echo "Aucune suppression automatique. Inspecter avant de relancer." >&2
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
      find "$results_dir" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null || true
    exit 1
    ;;
esac

job_file="$(mktemp)"
restored=0
cleanup() {
  code="$?"
  rm -f "$job_file"
  if [[ "$restored" -eq 0 ]]; then
    "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}" >/dev/null || true
    if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
      "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s || true
    fi
  fi
  exit "$code"
}
trap cleanup EXIT

sed \
  -e "s|__NAMESPACE__|$namespace|g" \
  -e "s|__JOB_NAME__|$job_name|g" \
  -e "s|__IMAGE__|$image|g" \
  -e "s|__RESULTS_DIR__|$results_dir|g" \
  -e "s|__E036_RESULTS_DIR__|$e036_results_dir|g" \
  -e "s|__SOURCE_COMMIT__|$source_commit|g" \
  deploy/k8s/e037-holdout-job.yaml >"$job_file"

echo "===== E037 prospective mini-holdout ====="
echo "Image          : $image"
echo "Source commit  : $source_commit"
echo "E036 référence : $e036_results_dir"
echo "Sortie         : $results_dir"
echo "Cas            : 10 prompts/seeds pré-enregistrés"
echo "Policy         : e036_gamma1000_global_trust"
echo "Gamma          : 1000"

echo "Libération de la RTX (API -> 0)."
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$job_file"

# Polling volontaire: pas de `kubectl logs -f`, pour éviter l'erreur fsnotify/inotify
# rencontrée pendant E036 sur pcIA.
echo "===== ATTENTE E037 (polling sans watcher de logs) ====="
started="$(date +%s)"
timeout_seconds="${PROOFTAG_E037_TIMEOUT_SECONDS:-21600}"
while true; do
  succeeded="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  now="$(date +%s)"
  elapsed="$((now - started))"
  printf '[E037] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
    "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
  if [[ "${succeeded:-0}" -ge 1 ]]; then
    break
  fi
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E037 en échec." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=300 || true
    "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
    exit 1
  fi
  if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
    echo "Timeout E037 après ${elapsed}s." >&2
    "$kubectl_bin" get job,pod -n "$namespace" -l app=prooftag-qr-e037 -o wide || true
    exit 1
  fi
  sleep 30
done

echo "===== LOGS FINAUX E037 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=300 || true

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$results_dir/verdict.json" "$results_dir/holdout-summary.csv" "$results_dir/e037-final-contact-sheet.png" <<'PY'
import json
from pathlib import Path
import sys
verdict, summary, sheet = map(Path, sys.argv[1:])
assert verdict.is_file(), verdict
assert summary.is_file(), summary
assert sheet.is_file(), sheet
data = json.loads(verdict.read_text(encoding="utf-8"))
assert data["gamma"] == 1000.0
assert data["gamma_preserved"] is True
assert data["case_count"] == 10
assert data["production_ready"] is False
print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
print("\n===== HOLDOUT SUMMARY =====")
print(summary.read_text(encoding="utf-8"))
PY

rm -f "$job_file"
trap - EXIT

echo "===== E037 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "Archive   : ${results_dir}.tar.gz"
echo "Dans Jupyter : Run > Run All Cells sur 32_e037_prospective_global_trust_holdout.ipynb"
