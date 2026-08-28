#!/usr/bin/env bash

# Exécuter dans un Bash enfant : `bash scripts/deploy-e035-notebook.sh`.
# Le sourcer appliquerait le mode strict au shell SSH de l'opérateur.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e035-notebook.sh" >&2
  return 2
fi

set -Eeuo pipefail

notebook="30_e035_srmpgd_loss_fidelity_gate.ipynb"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
default_notebook_deployment="${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-$default_notebook_deployment}"
kubectl_bin="${KUBECTL:-kubectl}"

state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/prooftag-qr"
state_file="${PROOFTAG_NOTEBOOK_STATE_FILE:-${state_dir}/notebook-previous-state}"

export PROOFTAG_QR_NAMESPACE="$namespace"
export PROOFTAG_NOTEBOOK_NAMESPACE="$namespace"
export PROOFTAG_QR_NOTEBOOK_DEPLOYMENT="$notebook_deployment"
export PROOFTAG_NOTEBOOK_DEPLOYMENT="$notebook_deployment"

failure_line=0
failure_command=""
runtime_prepared_by_this_script=0
previous_api_replicas=1
previous_vllm_replicas=0

record_failure() {
  failure_line="$1"
  failure_command="$2"
}

replicas_or_zero() {
  "$kubectl_bin" get deployment "$1" -n "$2" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
}

wait_for_pods_to_stop() {
  "$kubectl_bin" wait --for=delete pod -n "$1" -l "app=$2" \
    --timeout=300s >/dev/null 2>&1 || true
}

write_previous_state() {
  local api_replicas="$1"
  local vllm_replicas="$2"
  local temporary

  if [[ ! "$api_replicas" =~ ^[0-9]+$ || ! "$vllm_replicas" =~ ^[0-9]+$ ]]; then
    echo "État de réplica invalide : API=$api_replicas vLLM=$vllm_replicas" >&2
    exit 1
  fi

  install -d -m 700 "$state_dir"
  temporary="$(mktemp "${state_dir}/.notebook-previous-state.XXXXXX")"
  chmod 600 "$temporary"

  printf 'api_replicas=%s\nvllm_replicas=%s\n' \
    "$api_replicas" "$vllm_replicas" >"$temporary"

  mv -f -- "$temporary" "$state_file"
}

restore_previous_state() {
  "$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" \
    --replicas=0 >/dev/null 2>&1 || true

  wait_for_pods_to_stop "$namespace" prooftag-qr-notebook

  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" \
    --replicas="$previous_api_replicas" >/dev/null 2>&1 || true

  "$kubectl_bin" scale deployment vllm -n vllm \
    --replicas="$previous_vllm_replicas" >/dev/null 2>&1 || true

  if [[ "$previous_api_replicas" -gt 0 ]]; then
    "$kubectl_bin" rollout status "deployment/${api_deployment}" -n "$namespace" \
      --timeout=900s || true
  fi

  if [[ "$previous_vllm_replicas" -gt 0 ]]; then
    "$kubectl_bin" rollout status deployment/vllm -n vllm \
      --timeout=900s || true
  fi
}

report_exit() {
  local exit_code="$?"

  if [[ "$exit_code" -eq 0 ]]; then
    return
  fi

  if [[ "$runtime_prepared_by_this_script" -eq 1 ]]; then
    echo "Restauration de l'état GPU antérieur au déploiement incomplet." >&2
    restore_previous_state
  fi

  if [[ "$failure_line" -gt 0 ]]; then
    echo "ÉCHEC E035 à la ligne ${failure_line}: ${failure_command}" >&2
  else
    echo "ÉCHEC E035 (code ${exit_code})." >&2
  fi

  echo "La connexion SSH reste ouverte : le script utilise son propre Bash." >&2
}

trap 'record_failure "$LINENO" "$BASH_COMMAND"' ERR
trap report_exit EXIT

required_files=(
  "notebooks/${notebook}"
  "scripts/notebook-server.sh"
  "scripts/deploy-app-image.sh"
  "scripts/deploy-notebook-image.sh"
  "prooftag_qr/e035_loss_fidelity.py"
  "prooftag_qr/e035_losses.py"
  "prooftag_qr/e035_parent_artifact.py"
  "prooftag_qr/e035_parent_capture.py"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Fichier requis absent : $required_file" >&2
    echo "Lancer ce script depuis la racine du dépôt E035 à jour." >&2
    exit 1
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de déployer E035." >&2
  exit 1
fi

command -v git >/dev/null 2>&1 || {
  echo "Commande absente : git" >&2
  exit 1
}

command -v "$kubectl_bin" >/dev/null 2>&1 || {
  echo "Commande absente : $kubectl_bin" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || {
  echo "Commande absente : docker" >&2
  exit 1
}

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"

expected_api="${PROOFTAG_QR_IMAGE:-prooftag-qr}:${git_tag}"
expected_notebook="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}:${git_tag}"

echo "===== E035 : IMAGES IMMUABLES DU COMMIT ${git_sha} ====="
echo "Import Docker -> k3s par flux : aucune archive volumineuse dans /tmp."
echo "Runtime cible : notebook CPU, API GPU, vLLM arrêté."

notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"

if [[ "${notebook_replicas:-0}" -gt 0 ]]; then
  echo "Arrêt propre du notebook actif avant le déploiement E035."
  bash scripts/notebook-server.sh stop
fi

previous_api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
previous_vllm_replicas="$(replicas_or_zero vllm vllm)"

write_previous_state \
  "$previous_api_replicas" \
  "$previous_vllm_replicas"

runtime_prepared_by_this_script=1

echo "État GPU mémorisé avant déploiement : API=${previous_api_replicas}, vLLM=${previous_vllm_replicas}."

echo "Arrêt de vLLM pour libérer la RTX."

"$kubectl_bin" scale deployment vllm -n vllm \
  --replicas=0 >/dev/null

wait_for_pods_to_stop vllm vllm

echo "Configuration du notebook E035 en mode advisor-cpu."

advisor_patch='{
  "spec": {
    "template": {
      "metadata": {
        "annotations": {
          "prooftag.io/notebook-mode": "advisor-cpu"
        }
      },
      "spec": {
        "runtimeClassName": null,
        "containers": [
          {
            "name": "notebook",
            "resources": {
              "$patch": "replace",
              "requests": {
                "cpu": "1",
                "memory": "2Gi"
              },
              "limits": {
                "cpu": "4",
                "memory": "8Gi"
              }
            }
          }
        ]
      }
    }
  }
}'

"$kubectl_bin" patch deployment "$notebook_deployment" \
  -n "$namespace" \
  --type=strategic \
  -p "$advisor_patch" >/dev/null

echo "Activation de l'API GPU pour E035."

"$kubectl_bin" scale deployment "$api_deployment" \
  -n "$namespace" \
  --replicas=1 >/dev/null

"$kubectl_bin" rollout status \
  "deployment/${api_deployment}" \
  -n "$namespace" \
  --timeout=1200s

echo "===== E035 : CONSTRUCTION API ====="

bash scripts/deploy-app-image.sh

if [[ "$(git rev-parse HEAD)" != "$git_sha" ]]; then
  echo "Le commit a changé pendant la construction de l'API." >&2
  exit 1
fi

echo "===== E035 : CONSTRUCTION NOTEBOOK ====="

bash scripts/deploy-notebook-image.sh \
  "notebooks/${notebook}"

if [[ "$(git rev-parse HEAD)" != "$git_sha" ]]; then
  echo "Le commit a changé pendant la construction du notebook." >&2
  exit 1
fi

# La première version E035 du notebook-server ne connaît pas encore
# le notebook 30 comme advisor-cpu.
#
# Le runtime est donc préparé explicitement dans ce script :
#   - notebook sans GPU ;
#   - API avec la RTX ;
#   - vLLM arrêté.
#
# On démarre directement le Deployment notebook déjà configuré.

if ! "$kubectl_bin" get secret prooftag-qr-notebook \
  -n "$namespace" >/dev/null 2>&1; then

  token="$(
    od -An -N24 -tx1 /dev/urandom |
      tr -d ' \n'
  )"

  "$kubectl_bin" create secret generic \
    prooftag-qr-notebook \
    -n "$namespace" \
    --from-literal="token=${token}" >/dev/null
fi

"$kubectl_bin" scale deployment "$notebook_deployment" \
  -n "$namespace" \
  --replicas=1 >/dev/null

"$kubectl_bin" rollout status \
  "deployment/${notebook_deployment}" \
  -n "$namespace" \
  --timeout=1200s

echo "===== E035 : CONTRÔLE FINAL DES IMAGES ET DU RUNTIME ====="

deployment_value() {
  local deployment="$1"
  local expression="$2"

  "$kubectl_bin" get deployment "$deployment" \
    -n "$namespace" \
    -o "jsonpath=${expression}"
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

api_replicas="$(
  deployment_value "$api_deployment" \
    '{.spec.replicas}'
)"

active_notebook_replicas="$(
  deployment_value "$notebook_deployment" \
    '{.spec.replicas}'
)"

vllm_replicas="$(
  replicas_or_zero vllm vllm
)"

if [[ "${api_replicas:-0}" -ne 1 ||
      "${active_notebook_replicas:-0}" -ne 1 ||
      "${vllm_replicas:-0}" -ne 0 ]]; then

  echo "Réplicas inattendus : API=$api_replicas notebook=$active_notebook_replicas vLLM=$vllm_replicas" >&2
  exit 1
fi

"$kubectl_bin" rollout status \
  "deployment/${api_deployment}" \
  -n "$namespace" \
  --timeout=1200s

"$kubectl_bin" rollout status \
  "deployment/${notebook_deployment}" \
  -n "$namespace" \
  --timeout=1200s

notebook_pod="$(
  "$kubectl_bin" get pods \
    -n "$namespace" \
    -l app=prooftag-qr-notebook \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
)"

if [[ -z "$notebook_pod" ]]; then
  echo "Pod notebook E035 introuvable." >&2
  exit 1
fi

"$kubectl_bin" exec \
  -n "$namespace" \
  "$notebook_pod" \
  -c notebook -- \
  test -f "/workspace/notebooks/${notebook}"

"$kubectl_bin" exec -i \
  -n "$namespace" \
  "deployment/${api_deployment}" \
  -c api -- \
  python - <<'PY'
import importlib
import json
import os
import urllib.request

ready = json.load(
    urllib.request.urlopen(
        "http://127.0.0.1:8080/readyz"
    )
)

runtime = json.load(
    urllib.request.urlopen(
        "http://127.0.0.1:8080/v1/runtime"
    )
)

schema = json.load(
    urllib.request.urlopen(
        "http://127.0.0.1:8080/v1/lab/schema"
    )
)

assert ready["status"] == "ready"

assert (
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    == "expandable_segments:True"
)

identity = runtime.get("deployment_identity") or {}

assert identity.get("configured") is True

assert (
    identity.get("git_commit")
    == os.environ["PROOFTAG_GIT_COMMIT"]
)

for module_name in (
    "prooftag_qr.e035_loss_fidelity",
    "prooftag_qr.e035_losses",
    "prooftag_qr.e035_parent_artifact",
    "prooftag_qr.e035_parent_capture",
):
    importlib.import_module(module_name)

assert (
    schema.get("validation", {}).get("engine")
    == "antfu/qr-verify@0.2.0"
)

assert (
    schema.get("notes", {}).get("upstream_revision")
    == "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"
)

print("E035 imports OK")
print(
    "QR-Verify:",
    schema["validation"]["engine"],
)
print(
    "DiffQRCoder:",
    schema["notes"]["upstream_revision"],
)
PY

encoded_token="$(
  "$kubectl_bin" get secret \
    prooftag-qr-notebook \
    -n "$namespace" \
    -o jsonpath='{.data.token}'
)"

jupyter_token="$(
  printf '%s' "$encoded_token" |
    base64 --decode
)"

jupyter_target_ip="$(
  "$kubectl_bin" get service \
    "$notebook_deployment" \
    -n "$namespace" \
    -o jsonpath='{.spec.clusterIP}'
)"

jupyter_target_port="$(
  "$kubectl_bin" get service \
    "$notebook_deployment" \
    -n "$namespace" \
    -o jsonpath='{.spec.ports[0].port}'
)"

if [[ -z "$jupyter_token" ||
      -z "$jupyter_target_ip" ||
      -z "$jupyter_target_port" ]]; then

  echo "Impossible de résoudre le token ou la cible Jupyter." >&2
  exit 1
fi

runtime_prepared_by_this_script=0

trap - ERR
trap - EXIT

echo "Commit       : $git_sha"
echo "API          : $deployed_api ($api_digest)"
echo "Notebook     : $deployed_notebook ($notebook_digest)"
echo "Runtime      : notebook CPU, API GPU, vLLM arrêté"
echo "Notebook pod : /workspace/notebooks/${notebook}"
echo "JUPYTER_TOKEN=${jupyter_token}"
echo "JUPYTER_TARGET=${jupyter_target_ip}:${jupyter_target_port}"
echo "===== E035 PRÊT ====="
echo ""
echo "Tunnel depuis PowerShell :"
echo "ssh -N -L 18888:${jupyter_target_ip}:${jupyter_target_port} $(whoami)@$(hostname)"
echo ""
echo "Puis ouvrir :"
echo "http://127.0.0.1:18888/lab/tree/notebooks/${notebook}?token=${jupyter_token}"
echo ""
echo "Pour arrêter le notebook et restaurer l'état GPU précédent :"
echo "bash scripts/notebook-server.sh stop"
