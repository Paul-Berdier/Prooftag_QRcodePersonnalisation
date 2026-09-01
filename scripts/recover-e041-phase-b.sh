#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/recover-e041-phase-b.sh" >&2
  return 2
fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
job_name="${PROOFTAG_E041_RECOVERY_JOB_NAME:-prooftag-qr-e041-recover}"
results_dir="${PROOFTAG_E041_RESULTS_DIR:-/data/e041-gamma-functional-pattern-frontier-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
[[ -f deploy/k8s/e041-phase-b-recovery-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir" <<'PY'
import json
from pathlib import Path
import sys
from PIL import Image
root = Path(sys.argv[1])
if (root / 'verdict.json').is_file():
    print('E041_ALREADY_COMPLETE')
    raise SystemExit(0)
comparison = root / 'phase-a-scoring/comparison.json'
bases = root / 'phase-b-selected-bases.json'
assert comparison.is_file(), comparison
assert bases.is_file(), bases
rows = json.loads(comparison.read_text(encoding='utf-8'))
selected = json.loads(bases.read_text(encoding='utf-8'))
assert len(rows) == 54, len(rows)
assert len(selected) == 3, len(selected)
for row in rows:
    ip = Path(row['image_path']); lp = Path(row['latent_path'])
    assert ip.is_file(), ip
    assert lp.is_file(), lp
    with Image.open(ip) as im:
        assert im.size == (736, 736), (ip, im.size)
print('E041_PHASE_A_RECOVERABLE: 54/54 checkpoints + 3 bases présents')
PY

"$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- python - <<'PY'
from PIL import Image
from prooftag_qr.qr import generate_diffqrcoder_qr
from prooftag_qr.e041_gamma_functional_frontier import _functional_tone_exact_diffqrcoder
import prooftag_qr.e041_recover_phase_b
b = generate_diffqrcoder_qr('https://ptag.io/t/e041', 'M', version=3, mask_pattern=4, module_size=20)
i = Image.new('RGB', (736, 736), (127, 160, 100))
o = _functional_tone_exact_diffqrcoder(i, b, 0.20)
assert o.size == (736, 736)
print('E041 Phase-B geometry hotfix runtime OK: 736x736')
PY

job_file="$(mktemp)"
trap 'rm -f "$job_file"' EXIT
sed \
  -e "s|__NAMESPACE__|$namespace|g" \
  -e "s|__JOB_NAME__|$job_name|g" \
  -e "s|__IMAGE__|$image|g" \
  -e "s|__RESULTS_DIR__|$results_dir|g" \
  deploy/k8s/e041-phase-b-recovery-job.yaml >"$job_file"

"$kubectl_bin" delete job "$job_name" -n "$namespace" --ignore-not-found >/dev/null
"$kubectl_bin" apply -f "$job_file"
echo "===== RECUPERATION E041 PHASE B (CPU, aucun Stage1/Stage2/SR-MPGD) ====="
"$kubectl_bin" wait --for=condition=complete job/"$job_name" -n "$namespace" --timeout=2400s || {
  "$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=700 || true
  "$kubectl_bin" describe job "$job_name" -n "$namespace" || true
  exit 1
}
echo "===== LOGS FINAUX RECUPERATION E041 ====="
"$kubectl_bin" logs -n "$namespace" job/"$job_name" --all-containers=true --tail=700 || true

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir/verdict.json" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1]); assert p.is_file(), p
v = json.loads(p.read_text(encoding='utf-8'))
assert v['experiment'] == 'e041-gamma-functional-pattern-frontier-v1'
assert v['phase_a_checkpoint_count'] == 54
assert v['phase_b_variant_count'] == 18
assert v['recovered_from_existing_phase_a'] is True
assert v['phase_b_geometry_hotfix'] == 'exact-736-padding78-core29x29-module20'
assert v['production_ready'] is False
assert v['generalization_authorized'] is False
print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
PY
rm -f "$job_file"
trap - EXIT
echo "===== E041 RECUPERE SANS RECALCUL SR-MPGD ====="
echo "Résultats : $results_dir"
echo "Ensuite : .\\scripts\\e041-remote.ps1 puis .\\scripts\\e041-remote.ps1 -Pipeline"
