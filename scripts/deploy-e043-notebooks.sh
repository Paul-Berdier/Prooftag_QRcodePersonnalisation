#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
frontier="41_e043_scanner_cell_frontier.ipynb"
pipeline="42_e043_final_pipeline_visualizer.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"
for nb in "$frontier" "$pipeline"; do [[ -f "notebooks/$nb" ]] || { echo "Notebook E043 absent: $nb" >&2; exit 1; }; done
[[ -f prooftag_qr/e043_scanner_cell_frontier.py ]] || { echo "Runner E043 absent." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Commit/push/pull avant E043." >&2; exit 1; }
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

git_sha="$(git rev-parse HEAD)"
echo "===== E043 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/$frontier"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - <<'PY2'
import prooftag_qr.e043_scanner_cell_frontier as e
assert e.GAMMA == 500.0
assert e.LATENT_RADIUS_RMS == 0.2
assert len(e.RECIPES) == 4
assert e.EXPECTED_CHECKPOINT_COUNT == 36
assert e.RECIPES[0].name == 'e043_A_control_e041_g500'
assert e.RECIPES[-1].format_weight > 0
assert e.RECIPES[-1].data_ecc_risk_weight > 0
print('E043 runtime OK:', e.EXPERIMENT, 'gamma=', e.GAMMA, 'checkpoints=', e.EXPECTED_CHECKPOINT_COUNT)
PY2
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
echo "===== E043 PRÊT ====="
echo "Commit   : $git_sha"
echo "Frontier : $frontier"
echo "Pipeline : $pipeline"
echo "Calcul   : bash scripts/run-e043-scanner-cell-frontier.sh"
