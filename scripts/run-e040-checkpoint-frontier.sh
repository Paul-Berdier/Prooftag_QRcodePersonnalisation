#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e040-checkpoint-frontier.sh" >&2
  return 2
fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E040_JOB_NAME:-prooftag-qr-e040}"
parent_dir="${PROOFTAG_E040_PARENT_DIR:-/data/e035-parent-v1}"
e039_results_dir="${PROOFTAG_E040_E039_RESULTS_DIR:-/data/e039-srmpgd-limiter-scanaware-v1}"
results_dir="${PROOFTAG_E040_RESULTS_DIR:-/data/e040-srmpgd-checkpoint-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f deploy/k8s/e040-checkpoint-frontier-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
if [[ "${previous_api_replicas:-0}" -lt 1 ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
parent_commit="$($kubectl_bin exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$parent_dir/parent-stage2-metadata.json" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1]); data=json.loads(path.read_text(encoding='utf-8')); print(data['source']['source_commit'])
PY
)"
[[ "$parent_commit" =~ ^[0-9a-f]{40}$ ]] || { echo "Commit parent invalide: $parent_commit" >&2; exit 1; }

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$e039_results_dir/verdict.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); v=json.loads(p.read_text(encoding='utf-8'))
assert v['experiment']=='e039-srmpgd-limiter-scanaware-v1'
assert v['gamma']==1000.0 and v['gamma_preserved'] is True
assert v['research_winner']=='e039_scanaware_r200_i08'
print('Contrôle E039 vérifié: winner=e039_scanaware_r200_i08 gamma=1000')
PY

state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c "if [ ! -d '$results_dir' ] || [ -z \"\$(ls -A '$results_dir' 2>/dev/null)\" ]; then echo EMPTY; elif [ -f '$results_dir/verdict.json' ]; then echo COMPLETE; else echo PARTIAL; fi")"
case "$state" in
  COMPLETE)
    echo "E040 est déjà terminé: $results_dir/verdict.json"
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
    exit 0 ;;
  PARTIAL)
    echo "Sortie E040 partielle détectée: $results_dir" >&2
    echo "Aucune suppression automatique. Inspecter avant de relancer." >&2
    "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- find "$results_dir" -maxdepth 3 -type f -printf '%p\n' 2>/dev/null || true
    exit 1 ;;
esac

job_file="$(mktemp)"; restored=0
cleanup(){ code="$?"; rm -f "$job_file"; if [[ "$restored" -eq 0 ]]; then "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}" >/dev/null || true; fi; exit "$code"; }
trap cleanup EXIT
sed -e "s|__NAMESPACE__|$namespace|g" -e "s|__JOB_NAME__|$job_name|g" -e "s|__IMAGE__|$image|g" -e "s|__PARENT_DIR__|$parent_dir|g" -e "s|__E039_RESULTS_DIR__|$e039_results_dir|g" -e "s|__RESULTS_DIR__|$results_dir|g" -e "s|__EXPECTED_PARENT_COMMIT__|$parent_commit|g" deploy/k8s/e040-checkpoint-frontier-job.yaml >"$job_file"

echo "===== E040 SR-MPGD checkpoint frontier ====="
echo "Image          : $image"
echo "Parent         : $parent_dir"
echo "E039 contrôle  : $e039_results_dir"
echo "Sortie         : $results_dir"
echo "Gamma          : 1000 (figé)"
echo "Rayons         : .150 .175 .200 .225 .250"
echo "Checkpoints    : i0..i8 pour chaque rayon (45 états)"
echo "Modèles        : advisor E026/E031 + surrogate E016 si disponibles"

echo "Libération de la RTX (API -> 0)."
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
"$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$job_file"

echo "===== ATTENTE E040 (polling) ====="
started="$(date +%s)"; timeout_seconds="${PROOFTAG_E040_TIMEOUT_SECONDS:-43200}"
while true; do
  succeeded="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  failed="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  active="$($kubectl_bin get job "$job_name" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
  elapsed="$(( $(date +%s) - started ))"
  printf '[E040] elapsed=%ss active=%s succeeded=%s failed=%s\n' "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
  [[ "${succeeded:-0}" -ge 1 ]] && break
  if [[ "${failed:-0}" -ge 1 ]]; then
    echo "Job E040 en échec." >&2
    "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=600 || true
    "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
    exit 1
  fi
  [[ "$elapsed" -ge "$timeout_seconds" ]] && { echo "Timeout E040 après ${elapsed}s." >&2; exit 1; }
  sleep 30
done

echo "===== LOGS FINAUX E040 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=600 || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s; fi
restored=1

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir/verdict.json" "$results_dir/checkpoint-comparison.csv" "$results_dir/pipeline/full-pipeline-contact-sheet.png" "$results_dir/pipeline/99-FINAL-QR.png" <<'PY'
import json, sys
from pathlib import Path
verdict, table, sheet, final = map(Path, sys.argv[1:])
for p in (verdict,table,sheet,final): assert p.is_file(), p
v=json.loads(verdict.read_text(encoding='utf-8')); assert v['gamma']==1000.0 and v['gamma_preserved'] is True; assert v['checkpoint_count']==45
print(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True))
PY
rm -f "$job_file"; trap - EXIT
echo "===== E040 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "Notebook frontier : .\\scripts\\e040-remote.ps1"
echo "Pipeline finale    : .\\scripts\\e040-remote.ps1 -Pipeline"
