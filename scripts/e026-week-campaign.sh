#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-status}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
vllm_namespace="${PROOFTAG_VLLM_NAMESPACE:-vllm}"
vllm_deployment="${PROOFTAG_VLLM_DEPLOYMENT:-vllm}"
job_name="prooftag-e026-week"
secret_name="prooftag-e026-week"
runtime_config="prooftag-e026-week-runtime"
manifest="deploy/k8s/e026-week-job.yaml"

replicas_or_zero() {
  local deployment="$1"
  local target_namespace="$2"
  kubectl get deployment "$deployment" -n "$target_namespace" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
}

require_repository() {
  if [[ ! -f "$manifest" || ! -f pyproject.toml ]]; then
    echo "Lancer cette commande depuis la racine du dépôt." >&2
    exit 1
  fi
}

save_runtime_state() {
  local api_replicas notebook_replicas vllm_replicas debug_artifacts image
  api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
  notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
  vllm_replicas="$(replicas_or_zero "$vllm_deployment" "$vllm_namespace")"
  debug_artifacts="$(
    kubectl get configmap prooftag-qr-config -n "$namespace" \
      -o jsonpath='{.data.PROOFTAG_QR_SAVE_DEBUG_ARTIFACTS}'
  )"
  image="$(
    kubectl get deployment "$api_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
  )"
  kubectl create configmap "$runtime_config" -n "$namespace" \
    --from-literal=api_replicas="$api_replicas" \
    --from-literal=notebook_replicas="$notebook_replicas" \
    --from-literal=vllm_replicas="$vllm_replicas" \
    --from-literal=debug_artifacts="$debug_artifacts" \
    --from-literal=image="$image" \
    --dry-run=client -o yaml | kubectl apply -f -
}

pause_gpu_competitors() {
  kubectl scale deployment "$vllm_deployment" -n "$vllm_namespace" --replicas=0
  kubectl scale deployment "$notebook_deployment" -n "$namespace" --replicas=0
  kubectl wait --for=delete pod -n "$vllm_namespace" -l app=vllm --timeout=600s || true
}

configure_week_api() {
  kubectl patch configmap prooftag-qr-config -n "$namespace" --type=merge \
    -p '{"data":{"PROOFTAG_QR_SAVE_DEBUG_ARTIFACTS":"false"}}'
  kubectl scale deployment "$api_deployment" -n "$namespace" --replicas=1
  kubectl rollout restart deployment "$api_deployment" -n "$namespace"
  kubectl rollout status deployment "$api_deployment" -n "$namespace" --timeout=1200s
}

prepare_gpu_and_api() {
  pause_gpu_competitors
  configure_week_api
}

current_image() {
  kubectl get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
}

validate_deployment_and_plan() {
  local image
  image="$(current_image)"
  if [[ -z "$image" || "$image" == *':dev' ]]; then
    echo "Image API non tracable : $image" >&2
    echo "Deployer d'abord avec ./scripts/deploy-app-image.sh" >&2
    exit 1
  fi
  echo "===== VALIDATION DU PLAN E026W ====="
  kubectl exec -n "$namespace" deployment/"$api_deployment" -c api -- \
    env E026_PAYLOAD_BASE="$E026_PAYLOAD_BASE" \
    python -m prooftag_qr.week_campaign --plan-only
}

create_secret() {
  if [[ -z "${E026_PAYLOAD_BASE:-}" ]]; then
    echo "Définir E026_PAYLOAD_BASE sans l'écrire dans Git." >&2
    echo "Exemple : export E026_PAYLOAD_BASE='https://ptag.io/t/w'" >&2
    exit 1
  fi
  kubectl create secret generic "$secret_name" -n "$namespace" \
    --from-literal=payload="$E026_PAYLOAD_BASE" \
    --dry-run=client -o yaml | kubectl apply -f -
}

launch_job() {
  local image escaped_image
  image="$(current_image)"
  if [[ -z "$image" || "$image" == *':dev' ]]; then
    echo "Image API non traçable : $image" >&2
    echo "Déployer d'abord avec ./scripts/deploy-app-image.sh" >&2
    exit 1
  fi
  escaped_image="${image//|/\\|}"
  sed "s|__IMAGE__|${escaped_image}|g" "$manifest" | kubectl apply -f -
  kubectl get job "$job_name" -n "$namespace" -o wide
  echo "Runner démarré avec l'image exacte : $image"
}

restore_runtime() {
  if ! kubectl get configmap "$runtime_config" -n "$namespace" >/dev/null 2>&1; then
    echo "État antérieur absent ; aucune restauration automatique possible." >&2
    return 1
  fi
  local api_replicas notebook_replicas vllm_replicas debug_artifacts
  api_replicas="$(
    kubectl get configmap "$runtime_config" -n "$namespace" \
      -o jsonpath='{.data.api_replicas}'
  )"
  notebook_replicas="$(
    kubectl get configmap "$runtime_config" -n "$namespace" \
      -o jsonpath='{.data.notebook_replicas}'
  )"
  vllm_replicas="$(
    kubectl get configmap "$runtime_config" -n "$namespace" \
      -o jsonpath='{.data.vllm_replicas}'
  )"
  debug_artifacts="$(
    kubectl get configmap "$runtime_config" -n "$namespace" \
      -o jsonpath='{.data.debug_artifacts}'
  )"
  kubectl patch configmap prooftag-qr-config -n "$namespace" --type=merge \
    -p "{\"data\":{\"PROOFTAG_QR_SAVE_DEBUG_ARTIFACTS\":\"${debug_artifacts}\"}}"

  # The single GPU cannot serve API/notebook/vLLM concurrently. Restore the
  # exact replica counts, stopping the E026 API first when another workload won.
  if [[ "$vllm_replicas" -gt 0 || "$notebook_replicas" -gt 0 ]]; then
    kubectl scale deployment "$api_deployment" -n "$namespace" --replicas=0
  else
    kubectl scale deployment "$api_deployment" -n "$namespace" --replicas="$api_replicas"
    if [[ "$api_replicas" -gt 0 ]]; then
      kubectl rollout restart deployment "$api_deployment" -n "$namespace"
      kubectl rollout status deployment "$api_deployment" -n "$namespace" --timeout=1200s
    fi
  fi
  kubectl scale deployment "$notebook_deployment" -n "$namespace" \
    --replicas="$notebook_replicas"
  kubectl scale deployment "$vllm_deployment" -n "$vllm_namespace" \
    --replicas="$vllm_replicas"
  if [[ "$notebook_replicas" -gt 0 ]]; then
    kubectl rollout status deployment "$notebook_deployment" -n "$namespace" --timeout=1200s
  fi
  if [[ "$vllm_replicas" -gt 0 ]]; then
    kubectl rollout status deployment "$vllm_deployment" -n "$vllm_namespace" --timeout=1200s
  fi
  kubectl delete secret "$secret_name" -n "$namespace" --ignore-not-found
  kubectl delete configmap "$runtime_config" -n "$namespace" --ignore-not-found
  echo "Charge GPU précédente restaurée. Les données E026 restent sur le PVC."
}

show_status() {
  kubectl get job,pod -n "$namespace" -l app=prooftag-e026-week -o wide || true
  echo "===== DERNIERS LOGS ====="
  kubectl logs -n "$namespace" job/"$job_name" --tail=100 2>/dev/null || true
  echo "===== DISQUE ET ÉTAT PERSISTANT ====="
  if kubectl get deployment "$api_deployment" -n "$namespace" >/dev/null 2>&1; then
    kubectl exec -n "$namespace" deployment/"$api_deployment" -c api -- \
      sh -c 'df -h /data; find /data/e026-week -maxdepth 2 -name state.json -print -exec tail -n 40 {} \;' \
      2>/dev/null || true
  fi
}

case "$command_name" in
  deploy-start)
    require_repository
    if kubectl get job "$job_name" -n "$namespace" >/dev/null 2>&1; then
      echo "Le Job existe deja. Utiliser status, stop ou resume." >&2
      exit 1
    fi
    if kubectl get configmap "$runtime_config" -n "$namespace" >/dev/null 2>&1; then
      echo "Un etat E026W existe deja. Utiliser resume ou stop, sans l'ecraser." >&2
      exit 1
    fi
    if [[ -z "${E026_PAYLOAD_BASE:-}" ]]; then
      echo "Definir E026_PAYLOAD_BASE avant de modifier les charges GPU." >&2
      exit 1
    fi
    save_runtime_state
    if ! (
      create_secret &&
      pause_gpu_competitors &&
      kubectl scale deployment "$api_deployment" -n "$namespace" --replicas=1 &&
      bash scripts/deploy-app-image.sh &&
      validate_deployment_and_plan &&
      configure_week_api &&
      launch_job
    ); then
      echo "Deploiement E026W incomplet : restauration de l'etat precedent." >&2
      kubectl delete job "$job_name" -n "$namespace" --ignore-not-found \
        --wait=true --timeout=180s || true
      restore_runtime || true
      exit 1
    fi
    ;;
  start)
    require_repository
    if kubectl get job "$job_name" -n "$namespace" >/dev/null 2>&1; then
      echo "Le Job existe déjà. Utiliser status, stop ou resume." >&2
      exit 1
    fi
    if kubectl get configmap "$runtime_config" -n "$namespace" >/dev/null 2>&1; then
      echo "Un etat E026W existe deja. Utiliser resume ou stop, sans l'ecraser." >&2
      exit 1
    fi
    if [[ -z "${E026_PAYLOAD_BASE:-}" ]]; then
      echo "Definir E026_PAYLOAD_BASE avant de modifier les charges GPU." >&2
      exit 1
    fi
    validate_deployment_and_plan
    save_runtime_state
    if ! (create_secret && prepare_gpu_and_api && launch_job); then
      echo "Demarrage E026W incomplet : restauration de l'etat precedent." >&2
      kubectl delete job "$job_name" -n "$namespace" --ignore-not-found \
        --wait=true --timeout=180s || true
      restore_runtime || true
      exit 1
    fi
    ;;
  resume)
    require_repository
    if ! kubectl get configmap "$runtime_config" -n "$namespace" >/dev/null 2>&1; then
      echo "Aucun etat E026W a reprendre. Utiliser start." >&2
      exit 1
    fi
    if [[ -z "${E026_PAYLOAD_BASE:-}" ]] && \
      ! kubectl get secret "$secret_name" -n "$namespace" >/dev/null 2>&1; then
      echo "Definir E026_PAYLOAD_BASE pour recreer le Secret manquant." >&2
      exit 1
    fi
    if ! kubectl get secret "$secret_name" -n "$namespace" >/dev/null 2>&1; then
      create_secret
    fi
    kubectl delete job "$job_name" -n "$namespace" --ignore-not-found \
      --wait=true --timeout=180s
    prepare_gpu_and_api
    launch_job
    ;;
  status)
    show_status
    ;;
  logs)
    kubectl logs -n "$namespace" job/"$job_name" -f
    ;;
  stop)
    kubectl delete job "$job_name" -n "$namespace" --ignore-not-found \
      --wait=true --timeout=180s
    restore_runtime
    ;;
  *)
    echo "Usage: $0 {deploy-start|start|resume|status|logs|stop}" >&2
    exit 2
    ;;
esac
