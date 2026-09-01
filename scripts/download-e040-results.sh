#!/usr/bin/env bash
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results="${PROOFTAG_E040_RESULTS_DIR:-/data/e040-srmpgd-checkpoint-frontier-v1}"
dest="${PROOFTAG_E040_DOWNLOAD_DIR:-artifacts/e040-download}"
k="${KUBECTL:-kubectl}"
mkdir -p "$dest"
"$k" scale deployment "$api" -n "$namespace" --replicas=1 >/dev/null
"$k" rollout status deployment/"$api" -n "$namespace" --timeout=1200s
pod="$($k get pods -n "$namespace" -l app=prooftag-qr --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
remote=/tmp/e040-srmpgd-checkpoint-frontier-v1.tar.gz
local="$dest/e040-srmpgd-checkpoint-frontier-v1.tar.gz"
"$k" exec -n "$namespace" "$pod" -c api -- test -f "$results/verdict.json"
"$k" exec -n "$namespace" "$pod" -c api -- tar -czf "$remote" "$results"
"$k" cp "$namespace/$pod:$remote" "$local" -c api
"$k" exec -n "$namespace" "$pod" -c api -- rm -f "$remote"
sha256sum "$local" | tee "$local.sha256"
echo "Archive E040: $local"
