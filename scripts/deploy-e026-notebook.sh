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

echo "===== MODE E026 : NOTEBOOK CPU + API GPU ====="
notebook_replicas="$(
  kubectl get deployment prooftag-qr-notebook -n qr-core \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
)"

echo "===== IMAGE NOTEBOOK DU MÊME COMMIT ====="
bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"

started_by_this_script=0
rollback() {
  echo "Échec du déploiement E026." >&2
  if [[ "$started_by_this_script" -eq 1 ]]; then
    echo "Restauration des charges arrêtées au démarrage du notebook." >&2
    bash scripts/notebook-server.sh stop || true
  else
    echo "Le notebook était déjà actif : il reste démarré pour ne pas détruire sa session." >&2
  fi
}
trap rollback ERR

echo "===== POD NOTEBOOK FRAIS ====="
if [[ "${notebook_replicas:-0}" -gt 0 ]]; then
  bash scripts/notebook-server.sh reset "$notebook"
else
  started_by_this_script=1
  bash scripts/notebook-server.sh start "$notebook"
fi

echo "===== API DU MÊME COMMIT ====="
bash scripts/deploy-app-image.sh

echo "===== CONTRÔLE FINAL DU SHA ====="
git_tag="$(git rev-parse --short=12 HEAD)"
expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:${git_tag}"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"
deployed_api="$(
  kubectl get deployment prooftag-qr -n qr-core \
    -o 'jsonpath={.spec.template.spec.containers[?(@.name=="api")].image}'
)"
deployed_notebook="$(
  kubectl get deployment prooftag-qr-notebook -n qr-core \
    -o 'jsonpath={.spec.template.spec.containers[?(@.name=="notebook")].image}'
)"
if [[ "$deployed_api" != "$expected_api" || "$deployed_notebook" != "$expected_notebook" ]]; then
  echo "Versions incohérentes :" >&2
  echo "  API      : $deployed_api (attendu $expected_api)" >&2
  echo "  Notebook : $deployed_notebook (attendu $expected_notebook)" >&2
  exit 1
fi
echo "Commit     : $(git rev-parse HEAD)"
echo "API        : $deployed_api"
echo "Notebook   : $deployed_notebook"

trap - ERR
echo "===== E026 DÉPLOYÉ ====="
echo "Depuis PowerShell sur le PC :"
echo ".\\scripts\\notebook-remote.ps1 -Notebook ${notebook}"
echo "Dans Jupyter : modifier COLLECTION_PAYLOAD puis Run > Run All Cells."
