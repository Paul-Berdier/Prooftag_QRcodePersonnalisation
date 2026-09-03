#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e045-notebook.sh [all|api|notebook|check]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-all}"
notebook="47_e045_foundation_and_resilience.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
notebook_container="${PROOFTAG_QR_NOTEBOOK_CONTAINER:-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"

host_python="${PROOFTAG_HOST_PYTHON:-}"
if [[ -n "$host_python" ]]; then
  command -v "$host_python" >/dev/null 2>&1 || {
    echo "PROOFTAG_HOST_PYTHON pointe vers une commande absente : $host_python" >&2
    exit 1
  }
elif command -v python3 >/dev/null 2>&1; then
  host_python="python3"
elif command -v python >/dev/null 2>&1; then
  host_python="python"
else
  echo "Python introuvable sur l'hôte. Installer python3 ou définir PROOFTAG_HOST_PYTHON." >&2
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
  if [[ "$code" -eq 0 ]]; then
    return
  fi
  echo "ÉCHEC DÉPLOIEMENT E045 à la ligne ${failure_line}: ${failure_command}" >&2
  echo "Aucun résultat /data n'a été supprimé." >&2
  echo "Après correction, relancer seulement l'étape nécessaire :" >&2
  echo "  bash scripts/deploy-e045-notebook.sh api" >&2
  echo "  bash scripts/deploy-e045-notebook.sh notebook" >&2
}
trap 'record_failure "$LINENO" "$BASH_COMMAND"' ERR
trap report_exit EXIT

[[ -f "notebooks/$notebook" ]] || { echo "Notebook E045 absent." >&2; exit 1; }
for file in \
  prooftag_qr/resilient_experiment.py \
  prooftag_qr/e045_registry.py \
  prooftag_qr/e045_parameter_space.py \
  prooftag_qr/e045_phone_labels.py \
  prooftag_qr/e045_foundation.py; do
  [[ -f "$file" ]] || { echo "Fichier E045 absent: $file" >&2; exit 1; }
done
[[ -z "$(git status --porcelain)" ]] || {
  echo "Commit/push/pull avant de construire les images E045." >&2
  exit 1
}

echo "Python hôte E045 : $host_python ($("$host_python" --version 2>&1))"

"$host_python" -m py_compile \
  prooftag_qr/resilient_experiment.py \
  prooftag_qr/e045_registry.py \
  prooftag_qr/e045_parameter_space.py \
  prooftag_qr/e045_phone_labels.py \
  prooftag_qr/e045_foundation.py \
  scripts/build_e045_foundation_notebook.py \
  scripts/e045-import-phone-captures.py

case "$action" in
  api)
    bash scripts/deploy-app-image.sh
    ;;
  notebook)
    bash scripts/deploy-notebook-image.sh "notebooks/$notebook"
    ;;
  all)
    bash scripts/deploy-app-image.sh
    bash scripts/deploy-notebook-image.sh "notebooks/$notebook"
    ;;
  check)
    ;;
  *)
    echo "Action inconnue: $action" >&2
    exit 2
    ;;
esac

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
api_image="${PROOFTAG_QR_IMAGE:-prooftag-qr}:${git_tag}"
notebook_image="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"

if [[ "$action" == "api" || "$action" == "all" || "$action" == "check" ]]; then
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
    python - <<'PY'
from prooftag_qr.e045_foundation import EXPERIMENT
from prooftag_qr.e045_registry import EXPERIMENTS
from prooftag_qr.e045_parameter_space import PARAMETERS
from prooftag_qr.resilient_experiment import classify_failure

assert len(EXPERIMENTS) == 45
assert len(PARAMETERS) >= 90
assert classify_failure(RuntimeError("CUDA out of memory")).kind == "resource"
assert classify_failure(TimeoutError("timeout")).retryable is True
print("Runtime API E045 OK:", EXPERIMENT, len(EXPERIMENTS), len(PARAMETERS))
PY
fi

if [[ "$action" == "notebook" || "$action" == "all" ]]; then
  # Le Deployment notebook vient déjà d'être mis à jour et vérifié. Le dernier
  # contrôle se fait dans k3s, pas via `docker run`, afin qu'une indisponibilité
  # ponctuelle du socket containerd de l'hôte ne transforme pas un déploiement
  # réussi en faux échec.
  "$kubectl_bin" rollout status deployment/"$notebook_deployment"     -n "$namespace" --timeout=1200s
  "$kubectl_bin" exec -i -n "$namespace"     deployment/"$notebook_deployment"     -c "$notebook_container" --     python - <<'PY'
from pathlib import Path
from prooftag_qr.e045_registry import EXPERIMENTS
from prooftag_qr.e045_parameter_space import PARAMETERS

notebook = Path("/workspace/notebooks/47_e045_foundation_and_resilience.ipynb")
assert notebook.is_file(), notebook
assert len(EXPERIMENTS) == 45
assert len(PARAMETERS) >= 90
print("Runtime notebook E045 OK:", notebook, len(EXPERIMENTS), len(PARAMETERS))
PY
fi

trap - ERR
trap - EXIT
echo "===== E045 PRÊT ====="
echo "Commit   : $git_sha"
echo "API      : $api_image"
echo "Notebook : $notebook_image"
echo "Calcul CPU reprenable : bash scripts/run-e045-foundation.sh"
