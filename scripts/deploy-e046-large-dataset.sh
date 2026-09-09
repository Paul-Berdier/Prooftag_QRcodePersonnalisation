#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e046-large-dataset.sh [deploy|check]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-deploy}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
kubectl_bin="${KUBECTL:-kubectl}"
runner_lock_file="${PROOFTAG_E046_LARGE_LOCK_FILE:-/tmp/prooftag-e046-large.runner.lock}"

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

required=(
  data/e046_prompt_catalog_v2.json
  data/e046_large_doe_v1.json
  prooftag_qr/e046_large_catalog.py
  prooftag_qr/e046_large_doe.py
  prooftag_qr/e046_large_campaign.py
  notebooks/50_e046_large_advisor_dataset.ipynb
  scripts/build_e046_large_notebook.py
  scripts/build_e046_prompt_catalog.py
  scripts/deploy-notebook-image.sh
  scripts/run-e046-large-dataset.sh
  deploy/k8s/e046-large-parent-job.yaml
  deploy/k8s/e046-large-refinement-job.yaml
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || {
    echo "Absent : $file" >&2
    exit 1
  }
done

"$host_python" -m py_compile \
  prooftag_qr/e046_large_catalog.py \
  prooftag_qr/e046_large_doe.py \
  prooftag_qr/e046_large_campaign.py \
  scripts/build_e046_prompt_catalog.py
bash -n scripts/run-e046-large-dataset.sh
bash -n scripts/deploy-e046-large-dataset.sh
bash -n scripts/deploy-notebook-image.sh
"$host_python" - "$PWD" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("e046-large-parent-job.yaml", "e046-large-refinement-job.yaml"):
    text = (root / "deploy/k8s" / name).read_text(encoding="utf-8")
    for required in (
        "prooftag.io/experiment: e046-large-advisor-dataset-v1",
        "backoffLimit: 0",
        "completions: 1",
        "parallelism: 1",
        "runtimeClassName: nvidia",
        "nvidia.com/gpu: \"1\"",
        "claimName: prooftag-qr-data",
    ):
        assert required in text, (name, required)
print("Contrats Kubernetes E046 large valides.")
PY

case "$action" in
  check)
    echo "Préflight local E046 large OK. Aucun déploiement effectué."
    exit 0
    ;;
  deploy)
    ;;
  *)
    echo "Action inconnue : $action (attendu : deploy ou check)" >&2
    exit 2
    ;;
esac

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de construire l'image E046 large." >&2
  exit 1
fi

command -v flock >/dev/null 2>&1 || {
  echo "flock est requis pour coordonner déploiement et campagne E046 large." >&2
  exit 1
}
exec 9>"$runner_lock_file"
if ! flock -n 9; then
  echo "Déploiement refusé : un plan ou ordonnanceur E046 large est actif." >&2
  exit 1
fi

active_jobs="$(
  "$kubectl_bin" get jobs -n "$namespace" \
    -l prooftag.io/experiment \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' \
    | awk '($2 + 0) > 0 { print $1 }'
)"
if [[ -n "$active_jobs" ]]; then
  echo "Déploiement refusé : un Job d'expérience GPU est actif." >&2
  printf '%s\n' "$active_jobs" >&2
  exit 1
fi

nonterminal_experiment_pods="$(
  "$kubectl_bin" get pods -n "$namespace" \
    -l prooftag.io/experiment -o json \
    | "$host_python" -c '
import json, sys
for pod in json.load(sys.stdin).get("items", []):
    phase = pod.get("status", {}).get("phase")
    deleting = pod.get("metadata", {}).get("deletionTimestamp")
    if phase in {"Pending", "Running", "Unknown"} or (
        deleting and phase not in {"Succeeded", "Failed"}
    ):
        print(pod.get("metadata", {}).get("name"))
'
)"
if [[ -n "$nonterminal_experiment_pods" ]]; then
  echo "Déploiement refusé : des pods d'expérience sont encore non terminaux." >&2
  printf '%s\n' "$nonterminal_experiment_pods" >&2
  exit 1
fi

wait_for_pods_to_stop() {
  local target_namespace="$1"
  local app_label="$2"
  local deadline=$((SECONDS + 300))
  local pods
  while ((SECONDS < deadline)); do
    pods="$(
      "$kubectl_bin" get pods -n "$target_namespace" \
        -l "app=$app_label" --field-selector=status.phase=Running \
        -o name
    )"
    [[ -z "$pods" ]] && return 0
    sleep 2
  done
  echo "Arrêt incomplet : $target_namespace/$app_label" >&2
  printf '%s\n' "$pods" >&2
  return 1
}

replicas() {
  local target_namespace="$1"
  local deployment="$2"
  local value
  value="$(
    "$kubectl_bin" get deployment "$deployment" -n "$target_namespace" \
      -o jsonpath='{.spec.replicas}'
  )"
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "Répliques invalides pour $target_namespace/$deployment : $value" >&2
    return 1
  }
  printf '%s\n' "$value"
}

foreign_gpu_pods_except() {
  local allowed_namespace="${1:-}"
  local allowed_app="${2:-}"
  "$kubectl_bin" get pods -A -o json \
    | "$host_python" -c '
import json, sys
allowed_namespace, allowed_app = sys.argv[1:3]
for pod in json.load(sys.stdin).get("items", []):
    if pod.get("status", {}).get("phase") not in {"Pending", "Running", "Unknown"}:
        continue
    metadata = pod.get("metadata", {})
    labels = metadata.get("labels", {})
    if (
        allowed_namespace
        and metadata.get("namespace") == allowed_namespace
        and labels.get("app") == allowed_app
    ):
        continue
    containers = (
        pod.get("spec", {}).get("initContainers", [])
        + pod.get("spec", {}).get("containers", [])
    )
    requested = 0
    for container in containers:
        resources = container.get("resources", {})
        for kind in ("requests", "limits"):
            raw = resources.get(kind, {}).get("nvidia.com/gpu", 0)
            try:
                requested = max(requested, int(raw))
            except (TypeError, ValueError):
                pass
    if requested:
        print("{}/{}".format(metadata.get("namespace"), metadata.get("name")))
' "$allowed_namespace" "$allowed_app"
}

previous_api="$(replicas "$namespace" "$api_deployment")"
previous_notebook="$(replicas "$namespace" "$notebook_deployment")"
previous_vllm="$(replicas vllm vllm)"
if ((previous_api + previous_notebook + previous_vllm > 1)); then
  echo "Déploiement refusé : plusieurs runtimes GPU étaient demandés simultanément." >&2
  exit 1
fi
previous_api_revision="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
)"
previous_notebook_revision="$(
  "$kubectl_bin" get deployment "$notebook_deployment" -n "$namespace" \
    -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
)"
[[ "$previous_api_revision" =~ ^[1-9][0-9]*$ ]] || {
  echo "Révision Kubernetes API invalide : $previous_api_revision" >&2
  exit 1
}
[[ "$previous_notebook_revision" =~ ^[1-9][0-9]*$ ]] || {
  echo "Révision Kubernetes notebook invalide : $previous_notebook_revision" >&2
  exit 1
}

deployment_template_sha() {
  local target_namespace="$1"
  local deployment="$2"
  "$kubectl_bin" get deployment "$deployment" -n "$target_namespace" -o json \
    | "$host_python" -c '
import hashlib, json, sys
deployment = json.load(sys.stdin)
material = json.dumps(
    deployment["spec"]["template"],
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
print(hashlib.sha256(material).hexdigest())
'
}

previous_api_template_sha="$(
  deployment_template_sha "$namespace" "$api_deployment"
)"
previous_notebook_template_sha="$(
  deployment_template_sha "$namespace" "$notebook_deployment"
)"
[[ "$previous_api_template_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Empreinte initiale du PodTemplate API invalide : $previous_api_template_sha" >&2
  exit 1
}
[[ "$previous_notebook_template_sha" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Empreinte initiale du PodTemplate notebook invalide : $previous_notebook_template_sha" >&2
  exit 1
}
deployment_complete=0
deployment_started=0
notebook_deployment_started=0

restore_previous_replicas() {
  local foreign
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null || return 1
  "$kubectl_bin" scale deployment vllm -n vllm \
    --replicas=0 >/dev/null || return 1
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=0 >/dev/null || return 1
  wait_for_pods_to_stop "$namespace" prooftag-qr-notebook || return 1
  wait_for_pods_to_stop vllm vllm || return 1
  wait_for_pods_to_stop "$namespace" prooftag-qr || return 1
  foreign="$(foreign_gpu_pods_except)"
  if [[ -n "$foreign" ]]; then
    echo "Restauration refusée : une charge GPU étrangère est active :" >&2
    printf '%s\n' "$foreign" >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas="$previous_notebook" >/dev/null || return 1
  "$kubectl_bin" scale deployment vllm -n vllm \
    --replicas="$previous_vllm" >/dev/null || return 1
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas="$previous_api" >/dev/null || return 1
}

rollback_api_if_changed() {
  local current_template_sha restored_template_sha
  if [[ "$deployment_started" -ne 1 ]]; then
    return 0
  fi
  if ! current_template_sha="$(
    deployment_template_sha "$namespace" "$api_deployment"
  )"; then
    echo "Impossible de relire le PodTemplate API; rollback de précaution vers la révision $previous_api_revision." >&2
    current_template_sha="unknown"
  fi
  if [[ "$current_template_sha" == "$previous_api_template_sha" ]]; then
    echo "Le PodTemplate API n'a pas changé; aucun rollback Kubernetes requis." >&2
    return 0
  fi

  echo "Rollback API vers la révision Kubernetes $previous_api_revision avant restauration des répliques." >&2
  if ! "$kubectl_bin" rollout undo deployment/"$api_deployment" \
    -n "$namespace" --to-revision="$previous_api_revision"; then
    echo "La commande de rollback API a échoué." >&2
    return 1
  fi
  if ! "$kubectl_bin" rollout status deployment/"$api_deployment" \
    -n "$namespace" --timeout=1200s; then
    echo "Le rollback API n'a pas atteint un état stable dans le délai imparti." >&2
    return 1
  fi
  if ! restored_template_sha="$(
    deployment_template_sha "$namespace" "$api_deployment"
  )"; then
    echo "Impossible de vérifier le PodTemplate après rollback." >&2
    return 1
  fi
  if [[ "$restored_template_sha" != "$previous_api_template_sha" ]]; then
    echo "Rollback API incohérent : template=$restored_template_sha attendu=$previous_api_template_sha" >&2
    return 1
  fi
  echo "Rollback API terminé et vérifié." >&2
}

rollback_notebook_if_changed() {
  local current_template_sha restored_template_sha
  if [[ "$notebook_deployment_started" -ne 1 ]]; then
    return 0
  fi
  if ! current_template_sha="$(
    deployment_template_sha "$namespace" "$notebook_deployment"
  )"; then
    echo "Impossible de relire le PodTemplate notebook; rollback de précaution vers la révision $previous_notebook_revision." >&2
    current_template_sha="unknown"
  fi
  if [[ "$current_template_sha" == "$previous_notebook_template_sha" ]]; then
    echo "Le PodTemplate notebook n'a pas changé; aucun rollback Kubernetes requis." >&2
    return 0
  fi

  echo "Rollback notebook vers la révision Kubernetes $previous_notebook_revision." >&2
  if ! "$kubectl_bin" rollout undo deployment/"$notebook_deployment" \
    -n "$namespace" --to-revision="$previous_notebook_revision"; then
    echo "La commande de rollback notebook a échoué." >&2
    return 1
  fi
  if ! "$kubectl_bin" rollout status deployment/"$notebook_deployment" \
    -n "$namespace" --timeout=1200s; then
    echo "Le rollback notebook n'a pas atteint un état stable dans le délai imparti." >&2
    return 1
  fi
  if ! restored_template_sha="$(
    deployment_template_sha "$namespace" "$notebook_deployment"
  )"; then
    echo "Impossible de vérifier le PodTemplate notebook après rollback." >&2
    return 1
  fi
  if [[ "$restored_template_sha" != "$previous_notebook_template_sha" ]]; then
    echo "Rollback notebook incohérent : template=$restored_template_sha attendu=$previous_notebook_template_sha" >&2
    return 1
  fi
  echo "Rollback notebook terminé et vérifié." >&2
}

on_exit() {
  local code="$?"
  local rollback_confirmed=1
  if [[ "$code" -ne 0 && "$deployment_complete" -ne 1 ]]; then
    echo "Échec du déploiement E046 large." >&2
    if ! rollback_api_if_changed; then
      rollback_confirmed=0
      echo "ALERTE : le rollback API n'a pas pu être confirmé." >&2
    fi
    if ! rollback_notebook_if_changed; then
      rollback_confirmed=0
      echo "ALERTE : le rollback notebook n'a pas pu être confirmé." >&2
    fi
    if [[ "$rollback_confirmed" -eq 1 ]]; then
      echo "Restauration des répliques précédentes après confirmation du rollback." >&2
      if ! restore_previous_replicas; then
        echo "ALERTE : restauration GPU refusée; runtimes connus laissés arrêtés." >&2
      fi
    else
      echo "Répliques notebook/vLLM laissées arrêtées : vérifier et terminer le rollback API avant restauration." >&2
    fi
    echo "Aucune donnée /data n'a été supprimée." >&2
  fi
  return "$code"
}
trap on_exit EXIT

"$kubectl_bin" scale deployment "$notebook_deployment" \
  -n "$namespace" --replicas=0 >/dev/null
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null
wait_for_pods_to_stop "$namespace" prooftag-qr-notebook
wait_for_pods_to_stop vllm vllm
foreign_gpu_pods="$(foreign_gpu_pods_except "$namespace" prooftag-qr)"
if [[ -n "$foreign_gpu_pods" ]]; then
  echo "Déploiement refusé : une charge GPU étrangère est active :" >&2
  printf '%s\n' "$foreign_gpu_pods" >&2
  exit 1
fi

deployment_started=1
bash scripts/deploy-app-image.sh

git_sha="$(git rev-parse HEAD)"
"$kubectl_bin" scale deployment "$api_deployment" \
  -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" \
  -n "$namespace" --timeout=1200s
deployed_commit="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
)"
deployed_image="$(
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
)"
api_pod="$(
  "$kubectl_bin" get pods -n "$namespace" \
    -l app=prooftag-qr --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
    | awk '$2 == "True" { print $1; exit }'
)"
[[ "$deployed_commit" == "$git_sha" && -n "$api_pod" ]] || {
  echo "Identité API invalide : commit=$deployed_commit attendu=$git_sha pod=$api_pod" >&2
  exit 1
}
pod_spec_image="$(
  "$kubectl_bin" get pod "$api_pod" -n "$namespace" \
    -o jsonpath='{.spec.containers[?(@.name=="api")].image}'
)"
pod_image_id="$(
  "$kubectl_bin" get pod "$api_pod" -n "$namespace" \
    -o jsonpath='{.status.containerStatuses[?(@.name=="api")].imageID}'
)"
embedded_commit="$(
  "$kubectl_bin" exec -i -n "$namespace" "$api_pod" -c api -- \
    python -c "from pathlib import Path; print(Path('/app/prooftag-build-commit.txt').read_text(encoding='ascii').strip())"
)"
[[ "$pod_spec_image" == "$deployed_image" ]] || {
  echo "Image API incohérente : pod=$pod_spec_image deployment=$deployed_image" >&2
  exit 1
}
[[ "$embedded_commit" == "$git_sha" && "$embedded_commit" == "$deployed_commit" ]] || {
  echo "Attestation commit incohérente : image=$embedded_commit env=$deployed_commit git=$git_sha" >&2
  exit 1
}
[[ "${pod_image_id#docker-pullable://}" =~ @sha256:[0-9a-f]{64}$ ]] || {
  echo "ImageID API non épinglable : $pod_image_id" >&2
  exit 1
}
"$kubectl_bin" exec -i -n "$namespace" \
  deployment/"$api_deployment" -c api -- python - <<'PY'
from prooftag_qr.e046_catalog import EXPERIMENT as PILOT_EXPERIMENT
from prooftag_qr.e046_large_catalog import (
    EXPERIMENT,
    load_catalog,
    load_doe,
    load_prompts,
)
from prooftag_qr.e046_large_doe import CONFIG_COUNT

assert PILOT_EXPERIMENT == "e046-controlled-best-generator-v1"
assert EXPERIMENT == "e046-large-advisor-dataset-v1"
assert EXPERIMENT != PILOT_EXPERIMENT
catalog = load_catalog()
doe = load_doe()
prompts = load_prompts()
assert len(prompts) == 256
assert CONFIG_COUNT == 12
assert len(catalog["catalog_sha256"]) == 64
assert len(doe["doe_sha256"]) == 64
print("Runtime E046 large OK:", EXPERIMENT, len(prompts), CONFIG_COUNT)
PY
"$kubectl_bin" exec -i -n "$namespace" \
  deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign --help >/dev/null

notebook_deployment_started=1
PROOFTAG_NOTEBOOK_NAMESPACE="$namespace" \
PROOFTAG_NOTEBOOK_DEPLOYMENT="$notebook_deployment" \
  bash scripts/deploy-notebook-image.sh \
    notebooks/50_e046_large_advisor_dataset.ipynb

deployment_complete=1
trap - EXIT

echo "===== E046 LARGE PRÊT ====="
echo "Commit : $git_sha"
echo "Plan smoke (aucune génération) : bash scripts/run-e046-large-dataset.sh plan"
echo "Run smoke             : bash scripts/run-e046-large-dataset.sh run"
echo "Notebook CPU           : .\\scripts\\notebook-remote.ps1 -Reset -Notebook 50_e046_large_advisor_dataset.ipynb"
echo "Pilot                 : PROOFTAG_E046_LARGE_PROFILE=pilot bash scripts/run-e046-large-dataset.sh plan"
echo "Full : plan obligatoire puis confirmation RUN-3072; jamais automatique."
