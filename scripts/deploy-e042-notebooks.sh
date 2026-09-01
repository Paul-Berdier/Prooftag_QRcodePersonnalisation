#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
frontier="39_e042_decoder_failure_localization.ipynb"
pipeline="40_e042_diagnostic_pipeline_visualizer.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"
for nb in "$frontier" "$pipeline"; do [[ -f "notebooks/$nb" ]] || { echo "Notebook E042 absent: $nb" >&2; exit 1; }; done
[[ -f prooftag_qr/e042_decoder_failure_localization.py ]] || { echo "Runner E042 absent." >&2; exit 1; }
[[ -f deploy/k8s/e042-decode-selected-latents-job.yaml ]] || { echo "Job decode E042 absent." >&2; exit 1; }
[[ -f deploy/k8s/e042-decoder-diagnostic-job.yaml ]] || { echo "Job diagnose E042 absent." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Commit/push/pull avant E042." >&2; exit 1; }

"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s

git_sha="$(git rev-parse HEAD)"
echo "===== E042 : build commit $git_sha ====="
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/$frontier"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - <<'PY'
import numpy as np
from prooftag_qr.qr import generate_diffqrcoder_qr
import prooftag_qr.e042_decoder_failure_localization as e
assert e.EXPERIMENT == 'e042-decoder-failure-localization-v1'
assert e.QR_CANVAS_PX == 736
assert e.QR_CORE_PX == 580
assert e.QR_PADDING_PX == 78
assert len(e.SELECTED_STATES) == 9
b = generate_diffqrcoder_qr(e.PAYLOAD, e.ERROR_CORRECTION, version=3, mask_pattern=4, module_size=20)
ref = e._exact_reference_image(b)
assert ref.size == (736, 736)
masks = e._region_masks(b)
assert masks['format'].sum() >= 30
assert masks['data'].shape == (29, 29)
print('E042 runtime OK:', e.EXPERIMENT, 'states=', len(e.SELECTED_STATES))
PY
"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true

echo "===== E042 PRÊT ====="
echo "Commit     : $git_sha"
echo "Diagnostic : $frontier"
echo "Pipeline   : $pipeline"
echo "Lancer     : bash scripts/run-e042-decoder-diagnostic.sh"
