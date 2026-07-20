#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-status}"

case "$command_name" in
  pause-vllm)
    replicas="$(kubectl get deployment vllm -n vllm -o jsonpath='{.spec.replicas}')"
    printf '%s\n' "$replicas" > "${TMPDIR:-/tmp}/prooftag-vllm-replicas"
    kubectl scale deployment/vllm -n vllm --replicas=0
    kubectl wait --for=delete pod -n vllm -l app=vllm --timeout=300s || true
    nvidia-smi
    ;;
  resume-vllm)
    replicas_file="${TMPDIR:-/tmp}/prooftag-vllm-replicas"
    replicas=1
    if [[ -f "$replicas_file" ]]; then
      replicas="$(<"$replicas_file")"
    fi
    kubectl scale deployment/vllm -n vllm --replicas="$replicas"
    kubectl rollout status deployment/vllm -n vllm --timeout=900s
    nvidia-smi
    ;;
  status)
    kubectl get deployment,pod -n vllm -l app=vllm
    kubectl get deployment,pod -n qr-core -l app=prooftag-qr 2>/dev/null || true
    nvidia-smi
    ;;
  *)
    echo "Usage: $0 {status|pause-vllm|resume-vllm}" >&2
    exit 2
    ;;
esac
