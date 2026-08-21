#!/usr/bin/env bash
set -euo pipefail

notebook="26_e031_prospective_stage2_holdout.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"

if [[ ! -f "notebooks/${notebook}" || ! -f scripts/notebook-server.sh ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E031." >&2
  exit 1
fi

echo "===== E031 : API ET NOTEBOOK DU MÊME COMMIT ====="
notebook_replicas="$(
  kubectl get deployment "$notebook_deployment" -n "$namespace" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
)"

# E031 est un orchestrateur CPU. L'API du même commit conserve seule la RTX et
# exécute les campagnes prospectives ; vLLM reste arrêté pendant l'expérience.
bash scripts/deploy-app-image.sh
bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"

started_by_this_script=0
rollback() {
  echo "Échec du démarrage E031." >&2
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

echo "===== CONTRÔLE FINAL DE VERSION ET DE RUNTIME ====="
git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:${git_tag}"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"

deployment_value() {
  local deployment="$1"
  local expression="$2"
  kubectl get deployment "$deployment" -n "$namespace" -o "jsonpath=${expression}"
}

deployed_api="$(
  deployment_value "$api_deployment" \
    '{.spec.template.spec.containers[?(@.name=="api")].image}'
)"
deployed_notebook="$(
  deployment_value "$notebook_deployment" \
    '{.spec.template.spec.containers[?(@.name=="notebook")].image}'
)"
api_commit="$(
  deployment_value "$api_deployment" \
    '{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
)"
notebook_commit="$(
  deployment_value "$notebook_deployment" \
    '{.spec.template.spec.containers[?(@.name=="notebook")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
)"
api_runtime_image="$(
  deployment_value "$api_deployment" \
    '{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE")].value}'
)"
notebook_runtime_image="$(
  deployment_value "$notebook_deployment" \
    '{.spec.template.spec.containers[?(@.name=="notebook")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE")].value}'
)"
api_digest="$(
  deployment_value "$api_deployment" \
    '{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}'
)"
notebook_digest="$(
  deployment_value "$notebook_deployment" \
    '{.spec.template.spec.containers[?(@.name=="notebook")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}'
)"
runtime_mode="$(
  deployment_value "$notebook_deployment" \
    '{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}'
)"

if [[ "$deployed_api" != "$expected_api" || "$deployed_notebook" != "$expected_notebook" ]]; then
  echo "Versions incohérentes :" >&2
  echo "  API      : $deployed_api (attendu $expected_api)" >&2
  echo "  Notebook : $deployed_notebook (attendu $expected_notebook)" >&2
  exit 1
fi
if [[ "$api_commit" != "$git_sha" || "$notebook_commit" != "$git_sha" ]]; then
  echo "Commits runtime incohérents : api=$api_commit notebook=$notebook_commit attendu=$git_sha" >&2
  exit 1
fi
if [[ "$api_runtime_image" != "$expected_api" || "$notebook_runtime_image" != "$expected_notebook" ]]; then
  echo "Images runtime incohérentes : api=$api_runtime_image notebook=$notebook_runtime_image" >&2
  exit 1
fi
if [[ ! "$api_digest" =~ ^sha256:[0-9a-f]{64}$ || ! "$notebook_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest runtime absent ou invalide : api=$api_digest notebook=$notebook_digest" >&2
  exit 1
fi
if [[ "$runtime_mode" != "advisor-cpu" ]]; then
  echo "Mode notebook inattendu : $runtime_mode (attendu advisor-cpu)" >&2
  exit 1
fi

api_replicas="$(deployment_value "$api_deployment" '{.spec.replicas}')"
active_notebook_replicas="$(deployment_value "$notebook_deployment" '{.spec.replicas}')"
vllm_replicas="$(
  kubectl get deployment vllm -n vllm -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
)"
if [[ "${api_replicas:-0}" -ne 1 || "${active_notebook_replicas:-0}" -ne 1 || "${vllm_replicas:-0}" -ne 0 ]]; then
  echo "Réplicas inattendus : api=$api_replicas notebook=$active_notebook_replicas vllm=$vllm_replicas" >&2
  exit 1
fi

kubectl rollout status "deployment/${api_deployment}" -n "$namespace" --timeout=1200s
kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
kubectl exec -n "$namespace" "deployment/${notebook_deployment}" -c notebook -- \
  test -f "/workspace/notebooks/${notebook}"
kubectl exec -n "$namespace" "deployment/${api_deployment}" -c api -- \
  python -c \
  "import json, urllib.request; ready=json.load(urllib.request.urlopen('http://127.0.0.1:8080/readyz')); schema=json.load(urllib.request.urlopen('http://127.0.0.1:8080/v1/lab/schema')); assert ready['status']=='ready'; assert schema['notes']['upstream_revision']=='e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'; print('API E031 prête, DiffQRCoder épinglé')"

trap - ERR
echo "Commit       : $git_sha"
echo "API          : $deployed_api ($api_digest)"
echo "Notebook     : $deployed_notebook ($notebook_digest)"
echo "Runtime      : notebook CPU, API GPU, vLLM arrêté"
echo "===== E031 PRÊT ====="
echo "Depuis PowerShell sur le PC :"
echo ".\\scripts\\notebook-remote.ps1 -Notebook ${notebook}"
echo "Dans Jupyter : Run > Run All Cells. La reprise utilise le plan persistant E031."
