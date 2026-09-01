#!/usr/bin/env bash
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results_name="e041-gamma-functional-pattern-frontier-v1"
results_dir="${PROOFTAG_E041_RESULTS_DIR:-/data/$results_name}"
out_dir="${PROOFTAG_E041_DOWNLOAD_DIR:-artifacts/e041-download}"
kubectl_bin="${KUBECTL:-kubectl}"
mkdir -p "$out_dir"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s >/dev/null
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- test -f "$results_dir/verdict.json"
archive="$out_dir/${results_name}.tar.gz"
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- tar -C /data -czf - "$results_name" >"$archive"
sha256sum "$archive"
echo "Archive E041 : $archive"
