#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e043-scanner-cell-frontier.sh" >&2
  return 2
fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
job_name="${PROOFTAG_E043_JOB_NAME:-prooftag-qr-e043}"
e041_results_dir="${PROOFTAG_E043_E041_RESULTS_DIR:-/data/e041-gamma-functional-pattern-frontier-v1}"
e042_results_dir="${PROOFTAG_E043_E042_RESULTS_DIR:-/data/e042-decoder-failure-localization-v1}"
results_dir="${PROOFTAG_E043_RESULTS_DIR:-/data/e043-scanner-cell-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
[[ -f deploy/k8s/e043-scanner-cell-frontier-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
source_commit="$(git rev-parse HEAD)"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$e041_results_dir/verdict.json" "$e042_results_dir/verdict.json" <<'PY2'
import json, sys
from pathlib import Path
e41=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
e42=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
assert e41['experiment']=='e041-gamma-functional-pattern-frontier-v1'
assert e41['phase_a_checkpoint_count']==54
assert e42['experiment']=='e042-decoder-failure-localization-v1'
assert e42['primary_blocker']=='GRID_DETECTION_OR_INTRA_MODULE_TEXTURE'
assert e42['grid_reconstruction_rescue_count']>=1
print('E041/E042 préflight OK:', 'E041=', f"{e41['winner_ssr_exact_presets']}/37", 'E042=', e42['primary_blocker'], 'grid_rescue=', e42['grid_reconstruction_rescue_count'])
PY2

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E043 est déjà terminé : $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0
    ;;
  PARTIAL)
    echo "Sortie E043 partielle détectée : $results_dir" >&2
    echo "Aucune suppression automatique. Conserver les preuves et diagnostiquer avant toute reprise." >&2
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- find "$results_dir" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null || true
    exit 1
    ;;
esac

tmp_job="$(mktemp)"
restored=0
cleanup() {
  code="$?"
  rm -f "$tmp_job"
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
  -e "s|__E041_RESULTS_DIR__|$e041_results_dir|g" \
  -e "s|__E042_RESULTS_DIR__|$e042_results_dir|g" \
  -e "s|__SOURCE_COMMIT__|$source_commit|g" \
  deploy/k8s/e043-scanner-cell-frontier-job.yaml >"$tmp_job"

echo "===== E043 — scanner-cell frontier ====="
echo "Image       : $image"
echo "Sortie      : $results_dir"
echo "Parent      : E041 même prompt / même Stage2"
echo "Gamma       : 500 (pairé, pas une constante globale)"
echo "Rayon       : 0.20"
echo "A           : contrôle E041 gamma500, 9 latents réutilisés"
echo "B           : whole-cell + intra-module variance"
echo "C           : B + grid/sub-cell consistency"
echo "D           : C + format + data/ECC-risk proxy"
echo "Checkpoints : 36"
echo "Nouveaux updates SR-MPGD : 24"
echo "Quiet zone  : géométrie exacte 736 / 78 / 580"
echo "Généralisation : NON"

"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$tmp_job"

started="$(date +%s)"
timeout_seconds="${PROOFTAG_E043_TIMEOUT_SECONDS:-21600}"
echo "===== ATTENTE E043 (polling) ====="
while true; do
  succeeded="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  elapsed="$(( $(date +%s) - started ))"
  printf '[E043] elapsed=%ss active=%s succeeded=%s failed=%s\n' "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
  if [[ "${succeeded:-0}" -ge 1 ]]; then break; fi
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E043 en échec." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=800 || true
    "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
    exit 1
  fi
  if [[ "$elapsed" -ge "$timeout_seconds" ]]; then
    echo "Timeout E043 après ${elapsed}s." >&2
    "$kubectl_bin" get job,pod -n "$namespace" -l prooftag.io/experiment=e043-scanner-cell-frontier-v1 -o wide || true
    exit 1
  fi
  sleep 30
done

echo "===== LOGS FINAUX E043 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=800 || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir/verdict.json" <<'PY2'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); assert p.is_file(), p
v=json.loads(p.read_text(encoding='utf-8'))
assert v['experiment']=='e043-scanner-cell-frontier-v1'
assert v['gamma']==500.0
assert v['checkpoint_count']==36
assert v['recipe_count']==4
assert v['exact_diffqrcoder_quiet_zone_geometry'] is True
assert v['legacy_quiet_zone_core_overwrite'] is False
assert v['production_ready'] is False
assert v['generalization_authorized'] is False
print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
PY2
fi
rm -f "$tmp_job"
trap - EXIT
echo "===== E043 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "PowerShell : .\\scripts\\e043-remote.ps1"
