#!/usr/bin/env bash
set -euo pipefail

notebook="25_e030_reliable_qrverify_cascade.ipynb"

if [[ ! -f "notebooks/${notebook}" || ! -f scripts/notebook-server.sh ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E030." >&2
  exit 1
fi

echo "===== E030 : RESCORING CPU, NOTEBOOK IMMUTABLE ====="
notebook_replicas="$(
  kubectl get deployment prooftag-qr-notebook -n qr-core \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
)"

bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"

started_by_this_script=0
rollback() {
  echo "Échec du démarrage E030." >&2
  if [[ "$started_by_this_script" -eq 1 ]]; then
    bash scripts/notebook-server.sh stop || true
  fi
}
trap rollback ERR

if [[ "${notebook_replicas:-0}" -gt 0 ]]; then
  bash scripts/notebook-server.sh reset "$notebook"
else
  started_by_this_script=1
  bash scripts/notebook-server.sh start "$notebook"
fi

echo "===== CONTRÔLE FINAL DE VERSION ====="
git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"
deployed_notebook="$(
  kubectl get deployment prooftag-qr-notebook -n qr-core \
    -o 'jsonpath={.spec.template.spec.containers[?(@.name=="notebook")].image}'
)"
if [[ "$deployed_notebook" != "$expected_notebook" ]]; then
  echo "Versions incohérentes :" >&2
  echo "  Notebook : $deployed_notebook (attendu $expected_notebook)" >&2
  exit 1
fi

trap - ERR
echo "Commit     : $git_sha"
echo "Notebook   : $deployed_notebook"
echo "===== E030 PRÊT — AUCUNE GÉNÉRATION ET AUCUNE CHARGE GPU MODIFIÉE ====="
echo "Depuis PowerShell sur le PC :"
echo ".\\scripts\\notebook-remote.ps1 -Notebook ${notebook}"
echo "Dans Jupyter : Run > Run All Cells. Le notebook reprend son cache par hash."
