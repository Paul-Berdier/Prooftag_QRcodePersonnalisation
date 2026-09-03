#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e046-notebooks.sh [all|api|notebook|check]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-all}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
notebook_container="${PROOFTAG_QR_NOTEBOOK_CONTAINER:-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"
main_notebook="48_e046_controlled_best_generator.ipynb"
atlas_notebook="49_e046_visual_atlas.ipynb"

host_python="${PROOFTAG_HOST_PYTHON:-}"
if [[ -n "$host_python" ]]; then
  command -v "$host_python" >/dev/null 2>&1 || {
    echo "PROOFTAG_HOST_PYTHON absent : $host_python" >&2
    exit 1
  }
elif command -v python3 >/dev/null 2>&1; then
  host_python="python3"
elif command -v python >/dev/null 2>&1; then
  host_python="python"
else
  echo "Python hôte introuvable." >&2
  exit 1
fi

failure_line=0
failure_command=""
record_failure() {
  failure_line="$1"
  failure_command="$2"
}
report_exit() {
  code="$?"
  [[ "$code" -eq 0 ]] && return
  echo "ÉCHEC DÉPLOIEMENT E046 ligne ${failure_line}: ${failure_command}" >&2
  echo "Aucun résultat /data n'a été supprimé." >&2
}
trap 'record_failure "$LINENO" "$BASH_COMMAND"' ERR
trap report_exit EXIT

required=(
  "prooftag_qr/e046_catalog.py"
  "prooftag_qr/e046_quiet_zone.py"
  "prooftag_qr/e046_campaign.py"
  "notebooks/$main_notebook"
  "notebooks/$atlas_notebook"
  "deploy/k8s/e046-parent-job.yaml"
  "deploy/k8s/e046-refinement-job.yaml"
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || {
    echo "Absent : $file" >&2
    exit 1
  }
done
[[ -z "$(git status --porcelain)" ]] || {
  echo "Commit/push/pull avant de construire E046." >&2
  exit 1
}

"$host_python" -m py_compile \
  prooftag_qr/e046_catalog.py \
  prooftag_qr/e046_quiet_zone.py \
  prooftag_qr/e046_campaign.py \
  scripts/build_e046_notebooks.py

case "$action" in
  api)
    bash scripts/deploy-app-image.sh
    ;;
  notebook)
    bash scripts/deploy-notebook-image.sh "notebooks/$main_notebook"
    ;;
  all)
    "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
    "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true
    bash scripts/deploy-app-image.sh
    bash scripts/deploy-notebook-image.sh "notebooks/$main_notebook"
    ;;
  check)
    ;;
  *)
    echo "Action inconnue : $action" >&2
    exit 2
    ;;
esac

git_sha="$(git rev-parse HEAD)"

if [[ "$action" == "api" || "$action" == "all" || "$action" == "check" ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python - <<'PY'
from prooftag_qr.e046_catalog import (
    EXPERIMENT, PARENT_RECIPES, PROMPTS, SRMPGD_RECIPES, build_candidates
)
from prooftag_qr.e046_quiet_zone import compose_scene_preserving_quiet_zone
from PIL import Image

assert EXPERIMENT == "e046-controlled-best-generator-v1"
assert len(PROMPTS) == 8
assert len(PARENT_RECIPES) == 8
assert len(SRMPGD_RECIPES) == 4
assert len(build_candidates("pilot")) == 48
image = Image.new("RGB", (736, 736), (80, 120, 160))
output, evidence = compose_scene_preserving_quiet_zone(image)
assert output.size == image.size
assert evidence["core_byte_identical"] is True
assert evidence["uniform_flat_replacement"] is False
print("Runtime API E046 OK:", EXPERIMENT, len(build_candidates("pilot")))
PY
fi

if [[ "$action" == "notebook" || "$action" == "all" ]]; then
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$notebook_deployment" -n "$namespace" --timeout=1200s
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$notebook_deployment" \
    -c "$notebook_container" -- \
    python - <<'PY'
from pathlib import Path
from prooftag_qr.e046_catalog import PROMPTS, PARENT_RECIPES

for name in (
    "48_e046_controlled_best_generator.ipynb",
    "49_e046_visual_atlas.ipynb",
):
    path = Path("/workspace/notebooks") / name
    assert path.is_file(), path
assert len(PROMPTS) == 8
assert len(PARENT_RECIPES) == 8
print("Runtime notebook E046 OK")
PY
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null
fi

trap - ERR
trap - EXIT
echo "===== E046 PRÊT ====="
echo "Commit : $git_sha"
echo "Run    : PROOFTAG_E046_PROFILE=smoke bash scripts/run-e046-controlled-campaign.sh"
echo "Pilot  : PROOFTAG_E046_PROFILE=pilot bash scripts/run-e046-controlled-campaign.sh"
echo "Full   : PROOFTAG_E046_PROFILE=full bash scripts/run-e046-controlled-campaign.sh"
