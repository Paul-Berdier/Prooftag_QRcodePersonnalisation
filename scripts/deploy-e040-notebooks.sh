#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
frontier="35_e040_srmpgd_checkpoint_frontier.ipynb"
pipeline="36_final_qr_pipeline_visualizer.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}}"
kubectl_bin="${KUBECTL:-kubectl}"
for nb in "$frontier" "$pipeline"; do [[ -f "notebooks/$nb" ]] || { echo "Notebook E040 absent: $nb" >&2; exit 1; }; done
[[ -f prooftag_qr/e040_checkpoint_frontier.py ]] || { echo "Runner E040 absent." >&2; exit 1; }
[[ -f prooftag_qr/e040_finalize.py ]] || { echo "Finalizer E040 absent." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Commit/push/pull avant E040." >&2; exit 1; }

current="$($kubectl_bin get deployment "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0')"
if [[ "${current:-0}" -gt 0 ]]; then "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0; "$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr-notebook --timeout=300s || true; fi
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
advisor_patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"advisor-cpu"}},"spec":{"runtimeClassName":null,"containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
"$kubectl_bin" patch deployment "$notebook_deployment" -n "$namespace" --type=strategic -p "$advisor_patch" >/dev/null

git_sha="$(git rev-parse HEAD)"; tag="$(git rev-parse --short=12 HEAD)"
echo "===== E040 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/$frontier"
"$kubectl_bin" patch deployment "$notebook_deployment" -n "$namespace" --type=strategic -p "$advisor_patch" >/dev/null
if ! "$kubectl_bin" get secret prooftag-qr-notebook -n "$namespace" >/dev/null 2>&1; then token="$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"; "$kubectl_bin" create secret generic prooftag-qr-notebook -n "$namespace" --from-literal="token=$token" >/dev/null; fi
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$notebook_deployment" -n "$namespace" --timeout=1200s
pod="$($kubectl_bin get pods -n "$namespace" -l app=prooftag-qr-notebook --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
for nb in "$frontier" "$pipeline"; do "$kubectl_bin" exec -n "$namespace" "$pod" -c notebook -- test -f "/workspace/notebooks/$nb"; done
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- python -c 'import joblib, sklearn; import prooftag_qr.e040_checkpoint_frontier as m; import prooftag_qr.e040_finalize; assert len(m.DEFAULT_RECIPES)==5; assert m.E039Config().gamma==1000; print("E040 runtime + advisor deps + finalizer OK", joblib.__version__, sklearn.__version__)'
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- test -f /app/docs/e035-assets/e034-observed-stage1.png
echo "===== E040 PRÊT ====="
echo "Commit : $git_sha"
echo "Frontier : $frontier"
echo "Pipeline : $pipeline"
echo "Calcul GPU neuf : bash scripts/run-e040-checkpoint-frontier.sh"
echo "Run partiel déjà calculé : bash scripts/finalize-e040-partial.sh"
