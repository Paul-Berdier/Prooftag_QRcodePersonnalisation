#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e038-notebook.sh" >&2
  return 2
fi
set -Eeuo pipefail

notebook="33_e038_srmpgd_ssr_aesthetic_frontier.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f "notebooks/$notebook" ]] || { echo "Notebook E038 absent." >&2; exit 1; }
[[ -f prooftag_qr/e038_recipe_frontier.py ]] || { echo "Runner E038 absent." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E038." >&2
  exit 1
}

# Arrêt du notebook existant, API conservée pour le build, vLLM libéré.
current_notebook="$($kubectl_bin get deployment "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0')"
if [[ "${current_notebook:-0}" -gt 0 ]]; then
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0
  "$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr-notebook --timeout=300s || true
fi
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" wait --for=delete pod -n vllm -l app=vllm --timeout=300s >/dev/null 2>&1 || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

advisor_patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"advisor-cpu"}},"spec":{"runtimeClassName":null,"containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
"$kubectl_bin" patch deployment "$notebook_deployment" -n "$namespace" --type=strategic -p "$advisor_patch" >/dev/null

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:$git_tag"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:$git_tag"

echo "===== E038 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
[[ "$(git rev-parse HEAD)" == "$git_sha" ]] || { echo "Le commit a changé pendant le build API." >&2; exit 1; }
bash scripts/deploy-notebook-image.sh "notebooks/$notebook"
[[ "$(git rev-parse HEAD)" == "$git_sha" ]] || { echo "Le commit a changé pendant le build notebook." >&2; exit 1; }

"$kubectl_bin" patch deployment "$notebook_deployment" -n "$namespace" --type=strategic -p "$advisor_patch" >/dev/null
if ! "$kubectl_bin" get secret prooftag-qr-notebook -n "$namespace" >/dev/null 2>&1; then
  token="$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
  "$kubectl_bin" create secret generic prooftag-qr-notebook -n "$namespace" --from-literal="token=$token" >/dev/null
fi
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
"$kubectl_bin" rollout status deployment/"$notebook_deployment" -n "$namespace" --timeout=1200s

api_image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
notebook_image="$($kubectl_bin get deployment "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="notebook")].image}')"
runtime_mode="$($kubectl_bin get deployment "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}')"
[[ "$api_image" == "$expected_api" ]] || { echo "Image API inattendue: $api_image != $expected_api" >&2; exit 1; }
[[ "$notebook_image" == "$expected_notebook" ]] || { echo "Image notebook inattendue: $notebook_image != $expected_notebook" >&2; exit 1; }
[[ "$runtime_mode" == "advisor-cpu" ]] || { echo "Mode notebook inattendu: $runtime_mode" >&2; exit 1; }

notebook_pod="$($kubectl_bin get pods -n "$namespace" -l app=prooftag-qr-notebook --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$notebook_pod" ]] || { echo "Pod notebook E038 introuvable." >&2; exit 1; }
"$kubectl_bin" exec -n "$namespace" "$notebook_pod" -c notebook -- test -f "/workspace/notebooks/$notebook"
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -c 'import prooftag_qr.e038_recipe_frontier as m; assert len(m.DEFAULT_RECIPES)==10; print("E038 runtime import OK")'

encoded="$($kubectl_bin get secret prooftag-qr-notebook -n "$namespace" -o jsonpath='{.data.token}')"
token="$(printf '%s' "$encoded" | base64 --decode)"
target_ip="$($kubectl_bin get service "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.clusterIP}')"
target_port="$($kubectl_bin get service "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.ports[0].port}')"

echo "===== E038 PRÊT ====="
echo "Commit       : $git_sha"
echo "API          : $api_image"
echo "Notebook     : $notebook_image"
echo "Runtime      : notebook CPU, API GPU, vLLM arrêté"
echo "Notebook     : /workspace/notebooks/$notebook"
echo "JUPYTER_TOKEN=$token"
echo "JUPYTER_TARGET=$target_ip:$target_port"
echo "Depuis PowerShell : .\\scripts\\e038-remote.ps1"
echo "Calcul GPU : bash scripts/run-e038-recipe-frontier.sh"
