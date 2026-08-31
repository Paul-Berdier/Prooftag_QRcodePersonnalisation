#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e038-recipe-frontier.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E038_JOB_NAME:-prooftag-qr-e038}"
parent_dir="${PROOFTAG_E038_PARENT_DIR:-/data/e035-parent-v1}"
e035_results_dir="${PROOFTAG_E038_E035_RESULTS_DIR:-/data/e035-loss-fidelity-gate-v1}"
e036_results_dir="${PROOFTAG_E038_E036_RESULTS_DIR:-/data/e036-gamma1000-trust-region-v1}"
results_dir="${PROOFTAG_E038_RESULTS_DIR:-/data/e038-srmpgd-ssr-aesthetic-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f deploy/k8s/e038-recipe-frontier-job.yaml ]] || {
  echo "Lancer depuis la racine du dépôt." >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E038." >&2
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
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"parent E038 absent: {path}")
metadata = json.loads(path.read_text(encoding="utf-8"))
print(metadata["source"]["source_commit"])
PY
)"
[[ "$parent_commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Commit parent invalide: $parent_commit" >&2
  exit 1
}

# E038 est volontairement mono-parent et dépend des contrôles E035/E036 déjà exécutés.
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  test -f "$e035_results_dir/verdict.json"
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  test -f "$e036_results_dir/verdict.json"
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$e036_results_dir/verdict.json" <<'PY'
import json
from pathlib import Path
import sys
v = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert v["gamma"] == 1000.0
assert v["gamma_preserved"] is True
print("Contrôle E036 vérifié: gamma=1000")
PY

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c \
  "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E038 est déjà terminé: $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0
    ;;
  PARTIAL)
    echo "Sortie E038 partielle détectée: $results_dir" >&2
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
  -e "s|__PARENT_DIR__|$parent_dir|g" \
  -e "s|__E035_RESULTS_DIR__|$e035_results_dir|g" \
  -e "s|__E036_RESULTS_DIR__|$e036_results_dir|g" \
  -e "s|__RESULTS_DIR__|$results_dir|g" \
  -e "s|__EXPECTED_PARENT_COMMIT__|$parent_commit|g" \
  deploy/k8s/e038-recipe-frontier-job.yaml >"$job_file"

echo "===== E038 SR-MPGD SSR/aesthetic frontier ====="
echo "Image          : $image"
echo "Parent commit  : $parent_commit"
echo "Parent         : $parent_dir"
echo "E035 contrôle  : $e035_results_dir"
echo "E036 contrôle  : $e036_results_dir"
echo "Sortie         : $results_dir"
echo "Gamma          : 1000 (figé)"
echo "Nouvelles recettes : 10"
echo "Rayons purs    : .075 .100 .125 .150 .200 .300"
echo "Objectifs      : full / robust / hybrid"

echo "Libération de la RTX (API -> 0)."
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$job_file"

# Polling volontaire: aucun suivi continu des logs, donc pas de watcher fsnotify/inotify.
echo "===== ATTENTE E038 (polling) ====="
started="$(date +%s)"
timeout_seconds="${PROOFTAG_E038_TIMEOUT_SECONDS:-28800}"
while true; do
  succeeded="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  now="$(date +%s)"
  elapsed="$((now - started))"
  printf '[E038] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
    "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
  if [[ "${succeeded:-0}" -ge 1 ]]; then
    break
  fi
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E038 en échec." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=400 || true
    "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
    exit 1
  fi
  if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
    echo "Timeout E038 après ${elapsed}s." >&2
    "$kubectl_bin" get job,pod -n "$namespace" -l app=prooftag-qr-e038 -o wide || true
    exit 1
  fi
  sleep 30
done

echo "===== LOGS FINAUX E038 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=400 || true

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python - "$results_dir/verdict.json" "$results_dir/method-comparison.csv" "$results_dir/e038-all-methods-contact-sheet.png" <<'PY'
import json
from pathlib import Path
import sys
verdict, table, sheet = map(Path, sys.argv[1:])
assert verdict.is_file(), verdict
assert table.is_file(), table
assert sheet.is_file(), sheet
data = json.loads(verdict.read_text(encoding="utf-8"))
assert data["gamma"] == 1000.0
assert data["gamma_preserved"] is True
assert data["recipe_count"] == 10
assert data["production_ready"] is False
print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
print("\n===== METHOD COMPARISON =====")
print(table.read_text(encoding="utf-8"))
PY

rm -f "$job_file"
trap - EXIT

echo "===== E038 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "Dans Jupyter : Run > Run All Cells sur 33_e038_srmpgd_ssr_aesthetic_frontier.ipynb"
