#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-status}"
expected_notebook="${2:-21_e026_prompt_parameter_advisor.ipynb}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
state_dir="${XDG_STATE_HOME:-${HOME}/.local/state}/prooftag-qr"
state_file="${PROOFTAG_NOTEBOOK_STATE_FILE:-${state_dir}/notebook-previous-state}"

if [[ ! "$expected_notebook" =~ ^[A-Za-z0-9_.-]+\.ipynb$ ]]; then
  echo "Nom de notebook invalide : $expected_notebook" >&2
  exit 2
fi
expected_notebook_path="/workspace/notebooks/${expected_notebook}"
advisor_mode=0
offline_mode=0
case "$expected_notebook" in
  21_e026_prompt_parameter_advisor.ipynb|22_e027_srmpgd_policy_holdout.ipynb|23_e028_hierarchical_prompt_advisor.ipynb|24_e029_srmpgd_exact_raster_recovery.ipynb|26_e031_prospective_stage2_holdout.ipynb|27_e032_srmpgd_paper_reconstruction.ipynb|28_e033_srmpgd_microdiagnostic.ipynb|29_e034_srmpgd_four_iteration_gate.ipynb)
    advisor_mode=1
    ;;
  25_e030_reliable_qrverify_cascade.ipynb)
    advisor_mode=1
    offline_mode=1
    ;;
esac

replicas_or_zero() {
  kubectl get deployment "$1" -n "$2" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0'
}

prepare_state_directory() {
  install -d -m 700 "$(dirname "$state_file")"
}

write_previous_state() {
  local api_replicas="$1"
  local vllm_replicas="$2"
  local temporary
  if [[ ! "$api_replicas" =~ ^[0-9]+$ || ! "$vllm_replicas" =~ ^[0-9]+$ ]]; then
    echo "État de réplica invalide : api=$api_replicas vllm=$vllm_replicas" >&2
    return 1
  fi
  prepare_state_directory
  temporary="$(mktemp "$(dirname "$state_file")/.notebook-previous-state.XXXXXX")"
  chmod 600 "$temporary"
  printf 'api_replicas=%s\nvllm_replicas=%s\n' \
    "$api_replicas" "$vllm_replicas" > "$temporary"
  mv -f -- "$temporary" "$state_file"
}

read_previous_state() {
  api_replicas=1
  vllm_replicas=0
  [[ -f "$state_file" ]] || return 0
  local key value
  while IFS='=' read -r key value; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
      echo "État notebook invalide dans $state_file : $key=$value" >&2
      return 1
    }
    case "$key" in
      api_replicas) api_replicas="$value" ;;
      vllm_replicas) vllm_replicas="$value" ;;
      *)
        echo "Clé d'état notebook inconnue dans $state_file : $key" >&2
        return 1
        ;;
    esac
  done < "$state_file"
}

wait_for_pods_to_stop() {
  kubectl wait --for=delete pod -n "$1" -l "app=$2" --timeout=300s || true
}

configure_notebook_runtime() {
  local patch
  if [[ "$offline_mode" -eq 1 ]]; then
    patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"offline-cpu"}},"spec":{"runtimeClassName":null,"containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
  elif [[ "$advisor_mode" -eq 1 ]]; then
    patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"advisor-cpu"}},"spec":{"runtimeClassName":null,"containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"1","memory":"2Gi"},"limits":{"cpu":"4","memory":"8Gi"}}}]}}}}'
  else
    patch='{"spec":{"template":{"metadata":{"annotations":{"prooftag.io/notebook-mode":"generation-gpu"}},"spec":{"runtimeClassName":"nvidia","containers":[{"name":"notebook","resources":{"$patch":"replace","requests":{"cpu":"2","memory":"8Gi","nvidia.com/gpu":"1"},"limits":{"cpu":"12","memory":"32Gi","nvidia.com/gpu":"1"}}}]}}}}'
  fi
  kubectl patch deployment "$notebook_deployment" -n "$namespace" \
    --type=strategic -p "$patch" >/dev/null
}

prepare_runtime() {
  if [[ "$offline_mode" -eq 1 ]]; then
    configure_notebook_runtime
    return
  fi
  kubectl scale deployment/vllm -n vllm --replicas=0 >/dev/null
  wait_for_pods_to_stop vllm vllm
  configure_notebook_runtime
  if [[ "$advisor_mode" -eq 1 ]]; then
    kubectl scale "deployment/${api_deployment}" -n "$namespace" --replicas=1 >/dev/null
    kubectl rollout status "deployment/${api_deployment}" -n "$namespace" --timeout=1200s
  else
    kubectl scale "deployment/${api_deployment}" -n "$namespace" --replicas=0 >/dev/null
    wait_for_pods_to_stop "$namespace" prooftag-qr
  fi
}

ensure_token() {
  if ! kubectl get secret prooftag-qr-notebook -n "$namespace" >/dev/null 2>&1; then
    local token
    token="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
    kubectl create secret generic prooftag-qr-notebook -n "$namespace" \
      --from-literal="token=${token}" >/dev/null
  fi
}

print_token() {
  local encoded
  local service_ip
  local service_port
  encoded="$(kubectl get secret prooftag-qr-notebook -n "$namespace" -o jsonpath='{.data.token}')"
  service_ip="$(kubectl get service "$notebook_deployment" -n "$namespace" \
    -o jsonpath='{.spec.clusterIP}')"
  service_port="$(kubectl get service "$notebook_deployment" -n "$namespace" \
    -o jsonpath='{.spec.ports[0].port}')"
  printf 'JUPYTER_TOKEN=%s\n' "$(printf '%s' "$encoded" | base64 --decode)"
  printf 'JUPYTER_TARGET=%s:%s\n' "$service_ip" "$service_port"
}

ready_notebook_pod_for_image() {
  local expected_image="$1"
  local deadline=$((SECONDS + 180))
  local pod running_image ready deleting
  while ((SECONDS < deadline)); do
    while IFS= read -r pod; do
      [[ -n "$pod" ]] || continue
      deleting="$(
        kubectl get pod "$pod" -n "$namespace" \
          -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null || true
      )"
      [[ -z "$deleting" ]] || continue
      running_image="$(
        kubectl get pod "$pod" -n "$namespace" \
          -o 'jsonpath={.spec.containers[?(@.name=="notebook")].image}' \
          2>/dev/null || true
      )"
      ready="$(
        kubectl get pod "$pod" -n "$namespace" \
          -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}' \
          2>/dev/null || true
      )"
      if [[ "$running_image" == "$expected_image" && "$ready" == "True" ]]; then
        printf '%s\n' "$pod"
        return 0
      fi
    done < <(
      kubectl get pods -n "$namespace" \
        -l app=prooftag-qr-notebook \
        --field-selector=status.phase=Running \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
    )
    sleep 2
  done
  echo "Aucun pod notebook prêt n'exécute l'image attendue $expected_image." >&2
  kubectl get pods -n "$namespace" -l app=prooftag-qr-notebook -o wide >&2
  return 1
}

verify_running_notebook() {
  local pod desired_image running_image desired_mode runtime_mode
  desired_image="$(
    kubectl get deployment "$notebook_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="notebook")].image}'
  )"
  pod="$(ready_notebook_pod_for_image "$desired_image")"
  running_image="$(
    kubectl get pod "$pod" -n "$namespace" \
      -o jsonpath='{.spec.containers[?(@.name=="notebook")].image}'
  )"
  desired_mode="generation-gpu"
  if [[ "$offline_mode" -eq 1 ]]; then
    desired_mode="offline-cpu"
  elif [[ "$advisor_mode" -eq 1 ]]; then
    desired_mode="advisor-cpu"
  fi
  runtime_mode="$(
    kubectl get deployment "$notebook_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.metadata.annotations.prooftag\.io/notebook-mode}'
  )"
  if [[ "$runtime_mode" != "$desired_mode" ]]; then
    echo "Mode notebook incorrect : actif=$runtime_mode demande=$desired_mode" >&2
    echo "Utiliser -Reset pour recréer le pod dans le bon mode." >&2
    return 1
  fi
  if [[ "$running_image" != "$desired_image" ]]; then
    echo "Image notebook obsolete : pod=$running_image deployment=$desired_image" >&2
    return 1
  fi
  if ! kubectl exec -n "$namespace" "$pod" -- test -f "$expected_notebook_path"; then
    echo "Notebook absent du pod $pod ($running_image) : $expected_notebook_path" >&2
    echo "Redeployer avec scripts/deploy-notebook-image.sh avant de relancer." >&2
    return 1
  fi
  echo "Notebook vérifié dans le pod prêt : $pod:$expected_notebook_path ($running_image)" >&2
}

restore_previous_state() {
  local api_replicas
  local vllm_replicas
  read_previous_state
  kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=0 >/dev/null
  wait_for_pods_to_stop "$namespace" prooftag-qr-notebook
  kubectl scale "deployment/${api_deployment}" -n "$namespace" \
    --replicas="${api_replicas}" >/dev/null
  kubectl scale deployment/vllm -n vllm --replicas="${vllm_replicas}" >/dev/null
  if [[ "$api_replicas" -gt 0 ]]; then
    kubectl rollout status "deployment/${api_deployment}" -n "$namespace" --timeout=900s
  fi
  if [[ "$vllm_replicas" -gt 0 ]]; then
    kubectl rollout status deployment/vllm -n vllm --timeout=900s
  fi
}

case "$command_name" in
  deploy-prepare)
    notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
    if [[ "$notebook_replicas" -gt 0 ]]; then
      echo "Un notebook est deja actif. L'arreter avant de preparer un deploiement." >&2
      exit 1
    fi
    api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
    vllm_replicas="$(replicas_or_zero vllm vllm)"
    write_previous_state "$api_replicas" "$vllm_replicas"
    rollback() {
      echo "Echec de la preparation, restauration de l'etat GPU precedent" >&2
      restore_previous_state
    }
    trap rollback ERR
    prepare_runtime
    trap - ERR
    echo "Etat GPU memorise avant deploiement : API=${api_replicas}, vLLM=${vllm_replicas}." >&2
    ;;
  deploy-start)
    if [[ ! -f "$state_file" ]]; then
      echo "Etat GPU precedent absent. Utiliser deploy-prepare avant deploy-start." >&2
      exit 1
    fi
    # Validate the prepared state before touching any workload. Unlike start,
    # deploy-start must never replace it with the temporary API=1/vLLM=0 state.
    read_previous_state
    ensure_token
    notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
    if [[ "$notebook_replicas" -gt 0 ]]; then
      verify_running_notebook
      print_token
      exit 0
    fi
    rollback() {
      echo "Echec du notebook, restauration de l'etat GPU precedent" >&2
      restore_previous_state
    }
    trap rollback ERR
    prepare_runtime
    kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=1 >/dev/null
    kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
    verify_running_notebook
    trap - ERR
    print_token
    ;;
  start)
    ensure_token
    notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
    if [[ "$notebook_replicas" -gt 0 ]]; then
      verify_running_notebook
      print_token
      exit 0
    fi
    api_replicas="$(replicas_or_zero "$api_deployment" "$namespace")"
    vllm_replicas="$(replicas_or_zero vllm vllm)"
    write_previous_state "$api_replicas" "$vllm_replicas"
    rollback() {
      echo "Échec du notebook, restauration de l'état GPU précédent" >&2
      restore_previous_state
    }
    trap rollback ERR
    prepare_runtime
    kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=1 >/dev/null
    kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
    verify_running_notebook
    trap - ERR
    print_token
    ;;
  reset)
    ensure_token
    notebook_replicas="$(replicas_or_zero "$notebook_deployment" "$namespace")"
    if [[ "$notebook_replicas" -lt 1 ]]; then
      echo "Le notebook n'est pas actif. Utiliser start pour mémoriser et arrêter les charges GPU." >&2
      exit 1
    fi
    kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=0 >/dev/null
    wait_for_pods_to_stop "$namespace" prooftag-qr-notebook
    prepare_runtime
    kubectl scale "deployment/${notebook_deployment}" -n "$namespace" --replicas=1 >/dev/null
    kubectl rollout status "deployment/${notebook_deployment}" -n "$namespace" --timeout=1200s
    verify_running_notebook
    print_token
    ;;
  stop)
    restore_previous_state
    ;;
  status)
    kubectl get deployment,pod,service -n "$namespace" -l app=prooftag-qr-notebook
    ;;
  *)
    echo "Usage: $0 {deploy-prepare|deploy-start|start|reset|stop|status}" >&2
    exit 2
    ;;
esac
