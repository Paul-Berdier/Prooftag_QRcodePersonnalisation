#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
ns="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
nb="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
k="${KUBECTL:-kubectl}"
main_nb="45_e044_multiprompt_complete_audit.ipynb"
atlas_nb="46_e044_visual_atlas.ipynb"
for f in "notebooks/$main_nb" "notebooks/$atlas_nb" prooftag_qr/e044_multiprompt_best_pipeline.py prooftag_qr/e044_aggregate.py; do [[ -f "$f" ]] || { echo "Absent: $f" >&2; exit 1; }; done
[[ -z "$(git status --porcelain)" ]] || { echo "Commit/push/pull avant E044." >&2; exit 1; }
$k scale deployment "$nb" -n "$ns" --replicas=0 >/dev/null || true
$k scale deployment vllm -n vllm --replicas=0 >/dev/null || true
$k scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
$k rollout status deployment/"$api" -n "$ns" --timeout=1200s
echo "===== E044 : build $(git rev-parse HEAD) ====="
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/$main_nb"
$k scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
$k rollout status deployment/"$api" -n "$ns" --timeout=1200s
$k exec -i -n "$ns" deployment/"$api" -c api -- python - <<'PY'
import prooftag_qr.e044_multiprompt_best_pipeline as e
assert e.EXPERIMENT=='e044-multi-prompt-best-pipeline-v1'
assert e.GAMMAS==(500.0,1000.0)
assert len(e.PROMPTS)==7
assert e.EXPECTED_CHECKPOINTS_PER_PROMPT==18
assert e.QR_PADDING_PX==78
print('E044 runtime OK:', e.EXPERIMENT, 'prompts=',len(e.PROMPTS),'gammas=',e.GAMMAS)
PY
$k scale deployment "$nb" -n "$ns" --replicas=0 >/dev/null || true
echo "===== E044 PRÊT ====="
echo "Main  : $main_nb"
echo "Atlas : $atlas_nb"
echo "Run   : bash scripts/run-e044-multiprompt-benchmark.sh"
