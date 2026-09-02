#!/usr/bin/env bash
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results_dir="${PROOFTAG_E043_RESULTS_DIR:-/data/e043-scanner-cell-frontier-v1}"
out_dir="${PROOFTAG_E043_DOWNLOAD_DIR:-artifacts/e043-download}"
archive="$out_dir/e043-scanner-cell-frontier-v1.tar.gz"
mkdir -p "$out_dir"
kubectl scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
kubectl rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
kubectl exec -n "$namespace" deployment/"$api_deployment" -c api -- test -f "$results_dir/verdict.json"
kubectl exec -n "$namespace" deployment/"$api_deployment" -c api -- tar -C "$(dirname "$results_dir")" -czf - "$(basename "$results_dir")" >"$archive"
ls -lh "$archive"
sha256sum "$archive"
