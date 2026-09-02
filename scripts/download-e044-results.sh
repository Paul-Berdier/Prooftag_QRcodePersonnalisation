#!/usr/bin/env bash
set -Eeuo pipefail
ns="${PROOFTAG_QR_NAMESPACE:-qr-core}"; api="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"; root="${PROOFTAG_E044_RESULTS_ROOT:-/data/e044-multi-prompt-best-pipeline-v1}"; out="${PROOFTAG_E044_DOWNLOAD_DIR:-artifacts/e044-download}"; mkdir -p "$out"; archive="$out/e044-multi-prompt-best-pipeline-v1.tar.gz"
kubectl scale deployment "$api" -n "$ns" --replicas=1 >/dev/null; kubectl rollout status deployment/"$api" -n "$ns" --timeout=1200s
kubectl exec -n "$ns" deployment/"$api" -c api -- test -f "$root/verdict.json"
kubectl exec -n "$ns" deployment/"$api" -c api -- tar -C "$(dirname "$root")" -czf - "$(basename "$root")" > "$archive"
ls -lh "$archive"; sha256sum "$archive"
