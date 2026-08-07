#!/usr/bin/env bash
set -euo pipefail

notebook="21_e026_prompt_parameter_advisor.ipynb"

if [[ ! -f "notebooks/${notebook}" || ! -f scripts/notebook-server.sh ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E026." >&2
  exit 1
fi

rollback() {
  echo "Échec du déploiement E026 : restauration de l'état précédent." >&2
  bash scripts/notebook-server.sh stop || true
}
trap rollback ERR

echo "===== MODE E026 : NOTEBOOK CPU + API GPU ====="
notebook_replicas="$(
  kubectl get deployment prooftag-qr-notebook -n qr-core \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
)"
if [[ "${notebook_replicas:-0}" -gt 0 ]]; then
  bash scripts/notebook-server.sh reset "$notebook"
else
  bash scripts/notebook-server.sh start "$notebook"
fi

echo "===== API DU MÊME COMMIT ====="
bash scripts/deploy-app-image.sh

echo "===== IMAGE NOTEBOOK DU MÊME COMMIT ====="
bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"

echo "===== POD NOTEBOOK FRAIS ====="
bash scripts/notebook-server.sh reset "$notebook"

trap - ERR
echo "===== E026 DÉPLOYÉ ====="
echo "Depuis PowerShell sur le PC :"
echo ".\\scripts\\notebook-remote.ps1 -Notebook ${notebook}"
echo "Dans Jupyter : modifier COLLECTION_PAYLOAD puis Run > Run All Cells."
