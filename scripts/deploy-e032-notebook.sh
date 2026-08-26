#!/usr/bin/env bash

# Ne jamais appliquer `set -e` au shell SSH interactif de l'opérateur.
# Ce fichier doit être exécuté avec `bash scripts/deploy-e032-notebook.sh`,
# et non chargé avec `source` ou `.`.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e032-notebook.sh" >&2
  return 2
fi

set -Eeuo pipefail

notebook="27_e032_srmpgd_paper_reconstruction.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
default_notebook_deployment="${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-$default_notebook_deployment}"

# Les scripts historiques emploient deux noms de variables pour la même cible.
# Les exporter ici empêche un déploiement croisé si le namespace ou le nom du
# Deployment ont été personnalisés.
export PROOFTAG_QR_NAMESPACE="$namespace"
export PROOFTAG_NOTEBOOK_NAMESPACE="$namespace"
export PROOFTAG_QR_NOTEBOOK_DEPLOYMENT="$notebook_deployment"
export PROOFTAG_NOTEBOOK_DEPLOYMENT="$notebook_deployment"

failure_line=0
failure_command=""
started_by_this_script=0
record_failure() {
  failure_line="$1"
  failure_command="$2"
}
report_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -eq 0 ]]; then
    return
  fi
  if [[ "$started_by_this_script" -eq 1 ]]; then
    echo "Arrêt du notebook démarré par ce déploiement incomplet." >&2
    bash scripts/notebook-server.sh stop || true
  fi
  if [[ "$failure_line" -gt 0 ]]; then
    echo "ÉCHEC E032 à la ligne ${failure_line}: ${failure_command}" >&2
  else
    echo "ÉCHEC E032 (code ${exit_code})." >&2
  fi
  echo "La connexion SSH reste ouverte : le script a été exécuté dans son propre Bash." >&2
}
trap 'record_failure "$LINENO" "$BASH_COMMAND"' ERR
trap report_exit EXIT

if [[ ! -f "notebooks/${notebook}" || ! -f scripts/notebook-server.sh ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E032." >&2
  exit 1
fi

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:${git_tag}"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"

echo "===== E032 : IMAGES IMMUABLES DU COMMIT ${git_sha} ====="
echo "Les images sont importées par flux Docker -> k3s ; aucune archive volumineuse n'est écrite dans /tmp."

notebook_replicas="$(${KUBECTL:-kubectl} get deployment "$notebook_deployment" \
  -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0')"

bash scripts/deploy-app-image.sh
if [[ "$(git rev-parse HEAD)" != "$git_sha" ]]; then
  echo "Le commit a changé pendant la construction de l'API." >&2
  exit 1
fi

bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"
if [[ "$(git rev-parse HEAD)" != "$git_sha" ]]; then
  echo "Le commit a changé pendant la construction du notebook." >&2
  exit 1
fi

if [[ "${notebook_replicas:-0}" -gt 0 ]]; then
  bash scripts/notebook-server.sh reset "$notebook"
else
  started_by_this_script=1
  bash scripts/notebook-server.sh start "$notebook"
fi

echo "===== E032 : CONTRÔLE FINAL DU COMMIT ET DES PROFILS ====="

deployment_value() {
  local deployment="$1"
  local expression="$2"
  kubectl get deployment "$deployment" -n "$namespace" -o "jsonpath=${expression}"
}

deployed_api="$(deployment_value "$api_deployment" \
  '{.spec.template.spec.containers[?(@.name=="api")].image}')"
deployed_notebook="$(deployment_value "$notebook_deployment" \
  '{.spec.template.spec.containers[?(@.name=="notebook")].image}')"
api_commit="$(deployment_value "$api_deployment" \
  '{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}')"
notebook_commit="$(deployment_value "$notebook_deployment" \
  '{.spec.template.spec.containers[?(@.name=="notebook")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}')"
api_digest="$(deployment_value "$api_deployment" \
  '{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}')"
notebook_digest="$(deployment_value "$notebook_deployment" \
  '{.spec.template.spec.containers[?(@.name=="notebook")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}')"
runtime_mode="$(deployment_value "$notebook_deployment" \
  '{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}')"

if [[ "$deployed_api" != "$expected_api" || "$deployed_notebook" != "$expected_notebook" ]]; then
  echo "Images incohérentes : API=$deployed_api notebook=$deployed_notebook" >&2
  echo "Attendues : API=$expected_api notebook=$expected_notebook" >&2
  exit 1
fi
if [[ "$api_commit" != "$git_sha" || "$notebook_commit" != "$git_sha" ]]; then
  echo "Commits incohérents : API=$api_commit notebook=$notebook_commit attendu=$git_sha" >&2
  exit 1
fi
if [[ ! "$api_digest" =~ ^sha256:[0-9a-f]{64}$ || ! "$notebook_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest absent ou invalide : API=$api_digest notebook=$notebook_digest" >&2
  exit 1
fi
if [[ "$runtime_mode" != "advisor-cpu" ]]; then
  echo "Mode notebook inattendu : $runtime_mode (attendu advisor-cpu)" >&2
  exit 1
fi

api_replicas="$(deployment_value "$api_deployment" '{.spec.replicas}')"
active_notebook_replicas="$(deployment_value "$notebook_deployment" '{.spec.replicas}')"
vllm_replicas="$(kubectl get deployment vllm -n vllm \
  -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0')"
if [[ "${api_replicas:-0}" -ne 1 || "${active_notebook_replicas:-0}" -ne 1 || "${vllm_replicas:-0}" -ne 0 ]]; then
  echo "Réplicas inattendus : API=$api_replicas notebook=$active_notebook_replicas vLLM=$vllm_replicas" >&2
  exit 1
fi

kubectl rollout status "deployment/${api_deployment}" -n "$namespace" --timeout=1200s
kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
kubectl exec -n "$namespace" "deployment/${notebook_deployment}" -c notebook -- \
  test -f "/workspace/notebooks/${notebook}"

kubectl exec -i -n "$namespace" "deployment/${api_deployment}" -c api -- \
  python - <<'PY'
import json
import urllib.request

ready = json.load(urllib.request.urlopen("http://127.0.0.1:8080/readyz"))
schema = json.load(urllib.request.urlopen("http://127.0.0.1:8080/v1/lab/schema"))
profiles = {item["id"]: item for item in schema["profiles"]}
required = {
    "diffqrcoder_stage1",
    "diffqrcoder_paper_srpg",
    "diffqrcoder_paper_srmpgd_guarded",
    "diffqrcoder_paper_srmpgd",
}
assert ready["status"] == "ready"
assert required <= set(profiles), sorted(required - set(profiles))

paper = profiles["diffqrcoder_paper_srmpgd"]
guarded = profiles["diffqrcoder_paper_srmpgd_guarded"]
paper_settings = paper["tools"]["settings"]
guarded_settings = guarded["tools"]["settings"]
assert paper["output_variant"] == "srmpgd" and paper["enabled"] is False
assert guarded["output_variant"] == "srmpgd" and guarded["enabled"] is False
assert paper_settings["srmpgd_protocol"] == "paper_equations"
assert guarded_settings["srmpgd_protocol"] == "guarded_production"
assert (
    paper_settings["srmpgd_max_iterations"]
    == guarded_settings["srmpgd_max_iterations"]
    == 20
)
assert (
    paper_settings["srmpgd_step_size"]
    == guarded_settings["srmpgd_step_size"]
    == 1000.0
)
assert (
    paper_settings["srmpgd_lpips_weight"]
    == guarded_settings["srmpgd_lpips_weight"]
    == 0.01
)
assert (
    paper_settings["srmpgd_crop_padding_px"]
    == guarded_settings["srmpgd_crop_padding_px"]
    == 78
)
assert 736 - 2 * paper_settings["srmpgd_crop_padding_px"] == 29 * 20
assert (
    paper_settings["diffqrcoder_stage2_target_mode"]
    == guarded_settings["diffqrcoder_stage2_target_mode"]
    == "qart_url_fragment"
)
print("Schéma E032 vérifié:", sorted(required))
PY

trap - ERR
trap - EXIT

echo "Commit       : $git_sha"
echo "API          : $deployed_api ($api_digest)"
echo "Notebook     : $deployed_notebook ($notebook_digest)"
echo "Runtime      : notebook CPU, API GPU, vLLM arrêté"
echo "Notebook pod : /workspace/notebooks/${notebook}"
echo "===== E032 PRÊT ====="
echo "Depuis PowerShell sur le PC :"
echo ".\\scripts\\notebook-remote.ps1 -Notebook ${notebook}"
echo "Dans Jupyter : Run > Run All Cells. Les galeries et l'état de reprise sont persistants."
