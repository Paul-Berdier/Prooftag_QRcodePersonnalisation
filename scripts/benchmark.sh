#!/usr/bin/env bash
set -euo pipefail

port_forward_pid=""

cleanup() {
  if [[ -n "$port_forward_pid" ]]; then
    kill "$port_forward_pid" 2>/dev/null || true
    wait "$port_forward_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -n "${PROOFTAG_QR_API_URL:-}" ]]; then
  api_url="$PROOFTAG_QR_API_URL"
else
  benchmark_port="${PROOFTAG_QR_BENCHMARK_PORT:-18081}"
  api_url="http://127.0.0.1:${benchmark_port}"
  port_forward_log="${TMPDIR:-/tmp}/prooftag-qr-benchmark-port-forward.log"
  kubectl port-forward -n qr-core service/prooftag-qr-svc \
    "${benchmark_port}:8080" >"$port_forward_log" 2>&1 &
  port_forward_pid="$!"
fi

python3 scripts/benchmark.py \
  --api-url "$api_url" \
  --output-root "${PROOFTAG_QR_BENCHMARK_DIR:-benchmark-results}"
