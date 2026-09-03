#!/usr/bin/env bash
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
output_root="${PROOFTAG_E046_OUTPUT_ROOT:-/data/e046-controlled-best-generator-v1}"
destination="${PROOFTAG_E046_DOWNLOAD_DIR:-artifacts/e046-download}"
kubectl_bin="${KUBECTL:-kubectl}"

active="$(
  "$kubectl_bin" get jobs -n "$namespace" \
    -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' \
    2>/dev/null \
    | awk '($2 + 0) > 0 { print $1 }' \
    || true
)"
if [[ -n "$active" ]]; then
  echo "Un Job E046 est actif; attendre avant l'archive." >&2
  exit 1
fi

mkdir -p "$destination"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

latest="$(
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python - "$output_root/LATEST.json" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["status"]=="complete", payload
print(payload["plan_dir"])
PY
)"
plan_id="$(basename "$latest")"
archive="$destination/${plan_id}-e046-controlled-best-generator-v1.tar.gz"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_campaign verify \
    --output-root "$output_root" \
    --plan-id "$plan_id" >/dev/null

"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  tar -C "$(dirname "$latest")" -czf - "$plan_id" >"$archive"

ls -lh "$archive"
sha256sum "$archive"
