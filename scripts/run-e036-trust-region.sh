#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e036-trust-region.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E036_JOB_NAME:-prooftag-qr-e036}"
parent_dir="${PROOFTAG_E036_PARENT_DIR:-/data/e035-parent-v1}"
e035_results_dir="${PROOFTAG_E036_E035_RESULTS_DIR:-/data/e035-loss-fidelity-gate-v1}"
results_dir="${PROOFTAG_E036_RESULTS_DIR:-/data/e036-gamma1000-trust-region-v1}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f deploy/k8s/e036-trust-region-job.yaml ]] || {
  echo "Lancer depuis la racine du dépôt." >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E036." >&2
  exit 1
}

previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
if [[ "${previous_api_replicas:-0}" -lt 1 ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi

image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
parent_commit="$($kubectl_bin exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$parent_dir/parent-stage2-metadata.json" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"parent E036 absent: {p}")
metadata = json.loads(p.read_text(encoding="utf-8"))
print(metadata["source"]["source_commit"])
PY
)"

[[ "$parent_commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Commit parent invalide: $parent_commit" >&2
  exit 1
}

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e035_parent_artifact "$parent_dir" >/dev/null
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  test -f "$e035_results_dir/verdict.json"

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c \
  "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E036 est déjà terminé: $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0
    ;;
  PARTIAL)
    echo "Sortie E036 partielle détectée: $results_dir" >&2
    echo "Aucune suppression automatique. Inspecter avant de relancer." >&2
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
  -e "s|__PARENT_DIR__|$parent_dir|g" \
  -e "s|__E035_RESULTS_DIR__|$e035_results_dir|g" \
  -e "s|__RESULTS_DIR__|$results_dir|g" \
  -e "s|__EXPECTED_PARENT_COMMIT__|$parent_commit|g" \
  deploy/k8s/e036-trust-region-job.yaml >"$job_file"

echo "===== E036 gamma=1000 trust-region ====="
echo "Image          : $image"
echo "Parent commit  : $parent_commit"
echo "Parent         : $parent_dir"
echo "E035 référence : $e035_results_dir"
echo "Sortie         : $results_dir"

echo "Libération de la RTX (API -> 0)."
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$job_file"

# Évite le faux BadRequest ContainerCreating rencontré sur E035.
"$kubectl_bin" wait --for=condition=Ready pod -n "$namespace" -l "job-name=$job_name" --timeout=1200s || true
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true -f &
logs_pid=$!

if ! "$kubectl_bin" wait --for=condition=complete job/"$job_name" -n "$namespace" --timeout="${PROOFTAG_E036_TIMEOUT:-10800s}"; then
  kill "$logs_pid" 2>/dev/null || true
  wait "$logs_pid" 2>/dev/null || true
  "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=-1 || true
  "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
  exit 1
fi
wait "$logs_pid" 2>/dev/null || true

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$results_dir/verdict.json" "$results_dir/branch-summary.csv" <<'PY'
import json
from pathlib import Path
import sys
verdict = Path(sys.argv[1])
summary = Path(sys.argv[2])
assert verdict.is_file(), verdict
assert summary.is_file(), summary
data = json.loads(verdict.read_text(encoding="utf-8"))
assert data["gamma"] == 1000.0
assert data["gamma_preserved"] is True
assert data["production_ready"] is False
print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
print("\n===== BRANCH SUMMARY =====")
print(summary.read_text(encoding="utf-8"))
PY

rm -f "$job_file"
trap - EXIT

echo "===== E036 TERMINÉ ====="
echo "Dans Jupyter: Run > Run All Cells sur 31_e036_gamma1000_trust_region.ipynb"
