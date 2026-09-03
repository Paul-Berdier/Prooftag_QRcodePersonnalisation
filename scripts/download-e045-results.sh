#!/usr/bin/env bash
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
output_root="${PROOFTAG_E045_OUTPUT_ROOT:-/data/e045-foundation-v1}"
destination="${PROOFTAG_E045_DOWNLOAD_DIR:-artifacts/e045-download}"
kubectl_bin="${KUBECTL:-kubectl}"

mkdir -p "$destination"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

latest="$(
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python - "$output_root/LATEST.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
payload=json.loads(p.read_text(encoding="utf-8"))
assert payload["status"]=="complete", payload
print(payload["plan_dir"])
PY
)"
plan_id="$(basename "$latest")"
archive="$destination/${plan_id}-e045-foundation-v1.tar.gz"

"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e045_foundation verify --output-root "$output_root" >/dev/null

"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  tar -C "$(dirname "$latest")" -czf - "$plan_id" >"$archive"

ls -lh "$archive"
sha256sum "$archive"
