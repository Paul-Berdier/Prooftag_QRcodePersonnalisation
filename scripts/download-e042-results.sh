#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results_dir="${PROOFTAG_E042_RESULTS_DIR:-/data/e042-decoder-failure-localization-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
out="artifacts/e042-download"
mkdir -p "$out"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s >/dev/null
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- test -f "$results_dir/verdict.json"
archive="$out/e042-decoder-failure-localization-v1.tar.gz"
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- tar -C "$(dirname "$results_dir")" -czf - "$(basename "$results_dir")" > "$archive"
sha256sum "$archive" | tee "$archive.sha256"
echo "Archive E042 : $archive"
