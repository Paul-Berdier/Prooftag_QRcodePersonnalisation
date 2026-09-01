#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e041-gamma-functional-frontier.sh" >&2
  return 2
fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
job_name="${PROOFTAG_E041_JOB_NAME:-prooftag-qr-e041}"
e040_results_dir="${PROOFTAG_E041_E040_RESULTS_DIR:-/data/e040-srmpgd-checkpoint-frontier-v1}"
results_dir="${PROOFTAG_E041_RESULTS_DIR:-/data/e041-gamma-functional-pattern-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
[[ -f deploy/k8s/e041-gamma-functional-frontier-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
source_commit="$(git rev-parse HEAD)"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$e040_results_dir/verdict.json" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"E040 verdict absent: {p}")
v = json.loads(p.read_text(encoding='utf-8'))
assert v['experiment'] == 'e040-srmpgd-checkpoint-frontier-v1'
assert v['gamma_preserved'] is True
assert v['checkpoint_count'] == 45
print('Contrôle E040 finalisé vérifié:', v['research_winner_checkpoint'], f"SSR={v['winner_ssr_exact_presets']}/37")
PY

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E041 est déjà terminé: $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0
    ;;
  PARTIAL)
    echo "Sortie E041 partielle détectée: $results_dir" >&2
    echo "Aucune suppression automatique. Inspecter avant de relancer." >&2
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- find "$results_dir" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null || true
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
  -e "s|__E040_RESULTS_DIR__|$e040_results_dir|g" \
  -e "s|__SOURCE_COMMIT__|$source_commit|g" \
  deploy/k8s/e041-gamma-functional-frontier-job.yaml >"$job_file"

echo "===== E041 gamma + motifs fonctionnels ====="
echo "Image          : $image"
echo "Sortie         : $results_dir"
echo "Nouveau prompt : botanical reading room / conservatory"
echo "Gamma          : 50 100 250 500 1000 2000"
echo "Rayon          : 0.20"
echo "Checkpoints    : i0..i8 pour chaque gamma (54 états)"
echo "Phase B        : top 3 gamma/checkpoints x 6 tone factors"
echo "Gamma 1000     : baseline historique, PAS valeur imposée"

"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
echo "Libération de la RTX (API -> 0)."
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$job_file"

echo "===== ATTENTE E041 (polling uniquement) ====="
started="$(date +%s)"
timeout_seconds="${PROOFTAG_E041_TIMEOUT_SECONDS:-21600}"
while true; do
  succeeded="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  elapsed="$(( $(date +%s) - started ))"
  printf '[E041] elapsed=%ss active=%s succeeded=%s failed=%s\n' "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
  if [[ "${succeeded:-0}" -ge 1 ]]; then break; fi
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E041 en échec." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=700 || true
    "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
    exit 1
  fi
  if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
    echo "Timeout E041 après ${elapsed}s." >&2
    "$kubectl_bin" get job,pod -n "$namespace" -l app=prooftag-qr-e041 -o wide || true
    exit 1
  fi
  sleep 30
done

echo "===== LOGS FINAUX E041 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=700 || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir/verdict.json" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
assert p.is_file(), p
v = json.loads(p.read_text(encoding='utf-8'))
assert v['experiment'] == 'e041-gamma-functional-pattern-frontier-v1'
assert v['historical_gamma_baseline'] == 1000.0
assert len(v['gamma_grid']) == 6
assert v['phase_a_checkpoint_count'] == 54
assert v['phase_b_variant_count'] == 18
assert v['production_ready'] is False
assert v['generalization_authorized'] is False
print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
PY
rm -f "$job_file"
trap - EXIT
echo "===== E041 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "Ensuite depuis PowerShell : .\\scripts\\e041-remote.ps1"
