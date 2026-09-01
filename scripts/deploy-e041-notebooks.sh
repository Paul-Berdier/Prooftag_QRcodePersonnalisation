#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
frontier="37_e041_gamma_functional_frontier.ipynb"
pipeline="38_e041_final_pipeline_visualizer.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"
for nb in "$frontier" "$pipeline"; do [[ -f "notebooks/$nb" ]] || { echo "Notebook E041 absent: $nb" >&2; exit 1; }; done
[[ -f prooftag_qr/e041_gamma_functional_frontier.py ]] || { echo "Runner E041 absent." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Commit/push/pull avant E041." >&2; exit 1; }

"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

git_sha="$(git rev-parse HEAD)"
echo "===== E041 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/$frontier"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- python - <<'PY'
import prooftag_qr.e041_gamma_functional_frontier as e
assert e.GAMMAS == (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)
assert e.LATENT_RADIUS_RMS == 0.2
assert len(e.FUNCTIONAL_TONE_FACTORS) == 6
assert e.PROMPT != 'a sunlit greenhouse filled with tomato plants and terracotta pots, botanical photograph'
print('E041 runtime OK:', e.PROMPT)
PY
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
echo "===== E041 PRÊT ====="
echo "Commit   : $git_sha"
echo "Frontier : $frontier"
echo "Pipeline : $pipeline"
echo "Calcul GPU : bash scripts/run-e041-gamma-functional-frontier.sh"
