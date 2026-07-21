#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-status}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
state_file="${TMPDIR:-/tmp}/prooftag-qr-notebook-previous-state"

replicas_or_zero() {
  kubectl get deployment "$1" -n "$2" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
}

wait_for_pods_to_stop() {
  kubectl wait --for=delete pod -n "$1" -l "app=$2" --timeout=300s || true
}

ensure_token() {
  if ! kubectl get secret prooftag-qr-notebook -n "$namespace" >/dev/null 2>&1; then
    local token
    token="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    kubectl create secret generic prooftag-qr-notebook -n "$namespace" \
      --from-literal="token=${token}" >/dev/null
  fi
}

print_token() {
  local encoded
  encoded="$(kubectl get secret prooftag-qr-notebook -n "$namespace" -o jsonpath='{.data.token}')"
  printf 'JUPYTER_TOKEN=%s\n' "$(printf '%s' "$encoded" | base64 --decode)"
}

restore_previous_state() {
  local api_replicas=1
  local vllm_replicas=0
  if [[ -f "$state_file" ]]; then
    # shellcheck disable=SC1090
    source "$state_file"
  fi
  kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=0 >/dev/null
  wait_for_pods_to_stop "$namespace" prooftag-qr-notebook
  kubectl scale "deployment/${api_deployment}" -n "$namespace" \
    --replicas="${api_replicas}" >/dev/null
  kubectl scale deployment/vllm -n vllm --replicas="${vllm_replicas}" >/dev/null
  if [[ "$api_replicas" -gt 0 ]]; then
    kubectl rollout status "deployment/${api_deployment}" -n "$namespace" --timeout=900s
  fi
  if [[ "$vllm_replicas" -gt 0 ]]; then
    kubectl rollout status deployment/vllm -n vllm --timeout=900s
  fi
}

case "$command_name" in
  start)
    ensure_token
    notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
    if [[ "$notebook_replicas" -gt 0 ]]; then
      print_token
      exit 0
    fi
    api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
    vllm_replicas="$(replicas_or_zero vllm vllm)"
    printf 'api_replicas=%q\nvllm_replicas=%q\n' "$api_replicas" "$vllm_replicas" > "$state_file"
    rollback() {
      echo "Échec du notebook, restauration de l'état GPU précédent" >&2
      restore_previous_state
    }
    trap rollback ERR
    kubectl scale "deployment/${api_deployment}" -n "$namespace" --replicas=0 >/dev/null
    kubectl scale deployment/vllm -n vllm --replicas=0 >/dev/null
    wait_for_pods_to_stop "$namespace" prooftag-qr
    wait_for_pods_to_stop vllm vllm
    kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=1 >/dev/null
    kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
    trap - ERR
    print_token
    ;;
  stop)
    restore_previous_state
    ;;
  status)
    kubectl get deployment,pod,service -n "$namespace" -l app=prooftag-qr-notebook
    ;;
  *)
    echo "Usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
