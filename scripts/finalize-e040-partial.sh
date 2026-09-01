#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/finalize-e040-partial.sh" >&2
  return 2
fi
set -Eeuo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E040_FINALIZE_JOB_NAME:-prooftag-qr-e040-finalize}"
parent_dir="${PROOFTAG_E040_PARENT_DIR:-/data/e035-parent-v1}"
e039_results_dir="${PROOFTAG_E040_E039_RESULTS_DIR:-/data/e039-srmpgd-limiter-scanaware-v1}"
results_dir="${PROOFTAG_E040_RESULTS_DIR:-/data/e040-srmpgd-checkpoint-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"

[[ -f deploy/k8s/e040-finalize-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
if (root / 'verdict.json').is_file():
    print('E040_ALREADY_COMPLETE')
    raise SystemExit(0)
comparison = root / 'checkpoint-comparison.json'
if not comparison.is_file():
    raise SystemExit(f'E040 non finalisable: {comparison} absent')
rows = json.loads(comparison.read_text(encoding='utf-8'))
if len(rows) != 45:
    raise SystemExit(f'E040 non finalisable: {len(rows)} checkpoints au lieu de 45')
if any(float(row['gamma']) != 1000.0 for row in rows):
    raise SystemExit('E040 non finalisable: gamma != 1000 trouvé')
for row in rows:
    for key in ('image_path', 'latent_path'):
        if not Path(row[key]).is_file():
            raise SystemExit(f'E040 non finalisable: {key} absent: {row[key]}')
print('E040_PARTIAL_FINALIZABLE: 45/45 checkpoints présents, gamma=1000')
PY

"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -c 'import joblib, sklearn; import prooftag_qr.e040_finalize; print("advisor runtime + finalizer OK", joblib.__version__, sklearn.__version__)'
"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- \
  test -f /app/docs/e035-assets/e034-observed-stage1.png

job_file="$(mktemp)"
trap 'rm -f "$job_file"' EXIT
sed \
  -e "s|__NAMESPACE__|$namespace|g" \
  -e "s|__JOB_NAME__|$job_name|g" \
  -e "s|__IMAGE__|$image|g" \
  -e "s|__PARENT_DIR__|$parent_dir|g" \
  -e "s|__E039_RESULTS_DIR__|$e039_results_dir|g" \
  -e "s|__RESULTS_DIR__|$results_dir|g" \
  deploy/k8s/e040-finalize-job.yaml >"$job_file"

"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found >/dev/null
"$kubectl_bin" apply -f "$job_file"
"$kubectl_bin" wait --for=condition=complete job/"$job_name" -n "$namespace" --timeout=600s || {
  "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=500 || true
  "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
  exit 1
}

echo "===== LOGS FINALISATION E040 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=500 || true

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
required = [
    root / 'verdict.json',
    root / 'checkpoint-comparison.json',
    root / 'advisor-preview.json',
    root / 'pipeline-manifest.json',
    root / 'pipeline/03-stage1.png',
    root / 'pipeline/99-FINAL-QR.png',
    root / 'pipeline/99-FINAL-latent.safetensors',
    root / 'pipeline/full-pipeline-contact-sheet.png',
    root / 'e040-artifact-manifest.json',
]
for path in required:
    assert path.is_file(), path
v = json.loads((root / 'verdict.json').read_text(encoding='utf-8'))
assert v['experiment'] == 'e040-srmpgd-checkpoint-frontier-v1'
assert v['gamma'] == 1000.0 and v['gamma_preserved'] is True
assert v['checkpoint_count'] == 45
assert v['finalized_from_existing_checkpoints'] is True
print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "===== E040 FINALISÉ SANS RECALCUL SR-MPGD ====="
echo "Résultats : $results_dir"
echo "Ensuite : .\\scripts\\e040-remote.ps1 puis .\\scripts\\e040-remote.ps1 -Pipeline"
