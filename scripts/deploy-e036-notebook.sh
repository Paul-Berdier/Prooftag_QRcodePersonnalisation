#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e036-notebook.sh" >&2
  return 2
fi
set -Eeuo pipefail

notebook="31_e036_gamma1000_trust_region.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}}"
kubectl_bin="${KUBECTL:-kubectl}"

failure_line=0
failure_command=""
runtime_prepared=0
previous_api_replicas=1
previous_vllm_replicas=0

record_failure() {
  failure_line="$1"
  failure_command="$2"
}

replicas_or_zero() {
  "$kubectl_bin" get deployment "$1" -n "$2" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
}

wait_stop() {
  "$kubectl_bin" wait --for=delete pod -n "$1" -l "app=$2" --timeout=300s >/dev/null 2>&1 || true
}

restore_runtime() {
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null 2>&1 || true
  wait_stop "$namespace" prooftag-qr-notebook
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="$previous_api_replicas" >/dev/null 2>&1 || true
  "$kubectl_bin" scale deployment vllm -n vllm --replicas="$previous_vllm_replicas" >/dev/null 2>&1 || true
  if [[ "$previous_api_replicas" -gt 0 ]]; then
    "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s || true
  fi
  if [[ "$previous_vllm_replicas" -gt 0 ]]; then
    "$kubectl_bin" rollout status deployment/vllm -n vllm --timeout=1200s || true
  fi
}

report_exit() {
  code="$?"
  if [[ "$code" -eq 0 ]]; then
    return
  fi
  if [[ "$runtime_prepared" -eq 1 ]]; then
    echo "Restauration de l'état GPU antérieur au déploiement E036 incomplet." >&2
    restore_runtime
  fi
  if [[ "$failure_line" -gt 0 ]]; then
    echo "ÉCHEC E036 à la ligne $failure_line: $failure_command" >&2
  else
    echo "ÉCHEC E036 (code $code)." >&2
  fi
}
trap 'record_failure "$LINENO" "$BASH_COMMAND"' ERR
trap report_exit EXIT

required=(
  "notebooks/$notebook"
  "prooftag_qr/e036_trust_region.py"
  "scripts/deploy-app-image.sh"
  "scripts/deploy-notebook-image.sh"
  "scripts/run-e036-trust-region.sh"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Fichier E036 absent: $path" >&2; exit 1; }
done
[[ -z "$(git status --porcelain)" ]] || {
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E036." >&2
  exit 1
}

previous_notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
if [[ "${previous_notebook_replicas:-0}" -gt 0 ]]; then
  echo "Arrêt du notebook actif avant E036."
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0
  wait_stop "$namespace" prooftag-qr-notebook
fi
previous_api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
previous_vllm_replicas="$(replicas_or_zero vllm vllm)"
runtime_prepared=1

"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null
wait_stop vllm vllm
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

advisor_patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"advisor-cpu"}},"spec":{"runtimeClassName":null,"containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
"$kubectl_bin" patch deployment "$notebook_deployment" -n "$namespace" --type=strategic -p "$advisor_patch" >/dev/null

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:$git_tag"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:$git_tag"

echo "===== E036 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
[[ "$(git rev-parse HEAD)" == "$git_sha" ]] || { echo "Le commit a changé pendant le build API." >&2; exit 1; }
bash scripts/deploy-notebook-image.sh "notebooks/$notebook"
[[ "$(git rev-parse HEAD)" == "$git_sha" ]] || { echo "Le commit a changé pendant le build notebook." >&2; exit 1; }

# deploy-notebook-image met à jour l'image mais ne garantit pas le mode CPU : on le réaffirme.
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
[[ -n "$notebook_pod" ]] || { echo "Pod notebook E036 introuvable." >&2; exit 1; }
"$kubectl_bin" exec -n "$namespace" "$notebook_pod" -c notebook -- test -f "/workspace/notebooks/$notebook"
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- python -c 'import prooftag_qr.e036_trust_region; print("E036 runtime import OK")'

encoded="$($kubectl_bin get secret prooftag-qr-notebook -n "$namespace" -o jsonpath='{.data.token}')"
token="$(printf '%s' "$encoded" | base64 --decode)"
target_ip="$($kubectl_bin get service "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.clusterIP}')"
target_port="$($kubectl_bin get service "$notebook_deployment" -n "$namespace" -o jsonpath='{.spec.ports[0].port}')"

runtime_prepared=0
trap - ERR
trap - EXIT

echo "===== E036 PRÊT ====="
echo "Commit       : $git_sha"
echo "API          : $api_image"
echo "Notebook     : $notebook_image"
echo "Runtime      : notebook CPU, API GPU, vLLM arrêté"
echo "Notebook     : /workspace/notebooks/$notebook"
echo "JUPYTER_TOKEN=$token"
echo "JUPYTER_TARGET=$target_ip:$target_port"
echo "Depuis PowerShell : .\\scripts\\e036-remote.ps1"
echo "Pour lancer le calcul GPU E036 : bash scripts/run-e036-trust-region.sh"
