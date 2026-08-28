#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/download-e037-results.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
results_dir="${PROOFTAG_E037_RESULTS_DIR:-/data/e037-prospective-mini-holdout-v1}"
remote_archive="${results_dir}.tar.gz"
destination="${PROOFTAG_E037_DOWNLOAD_DIR:-artifacts/e037-download}"
kubectl_bin="${KUBECTL:-kubectl}"

mkdir -p "$destination"
replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
if [[ "${replicas:-0}" -lt 1 ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
pod="$($kubectl_bin get pods -n "$namespace" -l app=prooftag-qr --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$pod" ]] || { echo "Pod API introuvable." >&2; exit 1; }
"$kubectl_bin" exec -n "$namespace" "$pod" -c api -- test -f "$remote_archive"
local_archive="$destination/$(basename "$remote_archive")"
"$kubectl_bin" cp "$namespace/$pod:$remote_archive" "$local_archive" -c api
sha256sum "$local_archive" | tee "$local_archive.sha256"
echo "Archive E037 téléchargée sur le serveur : $local_archive"
