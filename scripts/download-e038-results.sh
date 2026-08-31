#!/usr/bin/env bash
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results_dir="${PROOFTAG_E038_RESULTS_DIR:-/data/e038-srmpgd-ssr-aesthetic-frontier-v1}"
destination="${PROOFTAG_E038_DOWNLOAD_DIR:-artifacts/e038-download}"
kubectl_bin="${KUBECTL:-kubectl}"

mkdir -p "$destination"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
pod="$($kubectl_bin get pods -n "$namespace" -l app=prooftag-qr --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
archive_remote="/tmp/e038-srmpgd-ssr-aesthetic-frontier-v1.tar.gz"
archive_local="$destination/e038-srmpgd-ssr-aesthetic-frontier-v1.tar.gz"
"$kubectl_bin" exec -n "$namespace" "$pod" -c api -- test -f "$results_dir/verdict.json"
"$kubectl_bin" exec -n "$namespace" "$pod" -c api -- tar -czf "$archive_remote" "$results_dir"
"$kubectl_bin" cp "$namespace/$pod:$archive_remote" "$archive_local" -c api
"$kubectl_bin" exec -n "$namespace" "$pod" -c api -- rm -f "$archive_remote"
sha256sum "$archive_local" | tee "$archive_local.sha256"
echo "Archive E038: $archive_local"
