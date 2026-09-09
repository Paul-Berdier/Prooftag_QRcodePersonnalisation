#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e046-large-dataset.sh [plan|run|resume|status|progress|logs|verify|pause-after-current|cancel-current|restore-runtime]" >&2
  return 2
fi
set -Eeuo pipefail

action="${1:-plan}"
log_job="${2:-}"
profile="${PROOFTAG_E046_LARGE_PROFILE:-smoke}"
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
output_root="${PROOFTAG_E046_LARGE_OUTPUT_ROOT:-/data/e046-large-advisor-dataset-v1}"
historical_e045_output_root="/data/e045-foundation-v1"
historical_e046_output_root="/data/e046-controlled-best-generator-v1"
experiment="e046-large-advisor-dataset-v1"
kubectl_bin="${KUBECTL:-kubectl}"
parent_timeout="${PROOFTAG_E046_LARGE_PARENT_TIMEOUT_SECONDS:-10800}"
refinement_timeout="${PROOFTAG_E046_LARGE_REFINEMENT_TIMEOUT_SECONDS:-14400}"
retry_failed_jobs="${PROOFTAG_E046_LARGE_RETRY_FAILED:-0}"
override_srl_terminal="${PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL:-0}"
pause_file="${PROOFTAG_E046_LARGE_PAUSE_FILE:-/tmp/prooftag-e046-large.pause-after-current}"
runner_lock_file="${PROOFTAG_E046_LARGE_LOCK_FILE:-/tmp/prooftag-e046-large.runner.lock}"
terminal_srl_pattern="local upstream SRL port diverged from the pinned official class"

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

require_repo() {
  local file
  for file in \
    deploy/k8s/e046-large-parent-job.yaml \
    deploy/k8s/e046-large-refinement-job.yaml \
    prooftag_qr/e046_large_catalog.py \
    prooftag_qr/e046_large_doe.py \
    prooftag_qr/e046_large_campaign.py; do
    [[ -f "$file" ]] || {
      echo "Fichier E046 large absent : $file" >&2
      exit 1
    }
  done
}

validate_profile() {
  case "$profile" in
    smoke|pilot|full)
      ;;
    *)
      echo "PROOFTAG_E046_LARGE_PROFILE doit être smoke, pilot ou full." >&2
      exit 1
      ;;
  esac
}

running_large_jobs() {
  "$kubectl_bin" get jobs -n "$namespace" \
    -l "prooftag.io/experiment=$experiment" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' \
    | awk '($2 + 0) > 0 { print $1 }'
}

running_large_pods() {
  "$kubectl_bin" get pods -n "$namespace" \
    -l "prooftag.io/experiment=$experiment" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

nonterminal_large_pods() {
  "$kubectl_bin" get pods -n "$namespace" \
    -l "prooftag.io/experiment=$experiment" -o json \
    | "$host_python" -c '
import json, sys
payload = json.load(sys.stdin)
for pod in payload.get("items", []):
    phase = pod.get("status", {}).get("phase")
    deleting = pod.get("metadata", {}).get("deletionTimestamp")
    if phase in {"Pending", "Running", "Unknown"} or (
        deleting and phase not in {"Succeeded", "Failed"}
    ):
        print(pod.get("metadata", {}).get("name"))
'
}

assert_at_most_one_large_job() {
  local jobs count
  jobs="$(running_large_jobs)"
  count="$(printf '%s\n' "$jobs" | awk 'NF { count += 1 } END { print count + 0 }')"
  if [[ "$count" -gt 1 ]]; then
    echo "INVARIANT RTX VIOLÉ : $count Jobs E046 large sont actifs." >&2
    printf '%s\n' "$jobs" >&2
    exit 1
  fi
}

wait_for_deployment_pods_to_stop() {
  local target_namespace="$1"
  local app_label="$2"
  local deadline=$((SECONDS + 300))
  local pods
  while ((SECONDS < deadline)); do
    pods="$(
      "$kubectl_bin" get pods -n "$target_namespace" \
        -l "app=$app_label" \
        --field-selector=status.phase=Running \
        -o name
    )"
    [[ -z "$pods" ]] && return 0
    sleep 2
  done
  echo "Des pods $target_namespace/$app_label utilisent encore la RTX." >&2
  printf '%s\n' "$pods" >&2
  return 1
}

ensure_api() {
  local jobs pods foreign
  jobs="$(running_large_jobs)"
  pods="$(nonterminal_large_pods)"
  if [[ -n "$jobs" || -n "$pods" ]]; then
    echo "Un Job E046 large est actif : l'API GPU reste arrêtée." >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null
  "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null
  wait_for_deployment_pods_to_stop "$namespace" prooftag-qr-notebook
  wait_for_deployment_pods_to_stop vllm vllm
  foreign="$(foreign_gpu_pods_except_api)"
  if [[ -n "$foreign" ]]; then
    echo "L'API ne redémarre pas : une autre charge GPU est active :" >&2
    printf '%s\n' "$foreign" >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" \
    -n "$namespace" --timeout=1200s >/dev/null
}

stop_gpu_services() {
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null
  "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=0 >/dev/null
  wait_for_deployment_pods_to_stop "$namespace" prooftag-qr-notebook
  wait_for_deployment_pods_to_stop vllm vllm
  wait_for_deployment_pods_to_stop "$namespace" prooftag-qr
}

foreign_gpu_pods() {
  "$kubectl_bin" get pods -A -o json \
    | "$host_python" -c '
import json, sys
payload = json.load(sys.stdin)
for pod in payload.get("items", []):
    if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
        continue
    labels = pod.get("metadata", {}).get("labels", {})
    if labels.get("prooftag.io/experiment") == "e046-large-advisor-dataset-v1":
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
        metadata = pod.get("metadata", {})
        pod_namespace = metadata.get("namespace")
        pod_name = metadata.get("name")
        print(f"{pod_namespace}/{pod_name}")
'
}

foreign_gpu_pods_except_api() {
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
        metadata.get("namespace") == allowed_namespace
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
' "$namespace" "prooftag-qr"
}

restore_runtime_if_idle() {
  local jobs pods api_ready foreign
  jobs="$(running_large_jobs)"
  pods="$(nonterminal_large_pods)"
  if [[ -n "$jobs" || -n "$pods" ]]; then
    echo "Un Job E046 large est encore actif : API laissée à 0 pour la RTX." >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$notebook_deployment" \
    -n "$namespace" --replicas=0 >/dev/null
  "$kubectl_bin" scale deployment vllm -n vllm --replicas=0 \
    >/dev/null
  wait_for_deployment_pods_to_stop "$namespace" prooftag-qr-notebook
  wait_for_deployment_pods_to_stop vllm vllm
  foreign="$(foreign_gpu_pods_except_api)"
  if [[ -n "$foreign" ]]; then
    echo "Runtime non restauré : une autre charge GPU est active :" >&2
    printf '%s\n' "$foreign" >&2
    return 1
  fi
  "$kubectl_bin" scale deployment "$api_deployment" \
    -n "$namespace" --replicas=1 >/dev/null
  "$kubectl_bin" rollout status deployment/"$api_deployment" \
    -n "$namespace" --timeout=1200s >/dev/null
  api_ready="$(
    "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
      -o jsonpath='{.status.availableReplicas}'
  )"
  [[ "${api_ready:-0}" -ge 1 ]] || {
    echo "L'API n'a aucune réplique disponible après restauration." >&2
    return 1
  }
  echo "Runtime recherche restauré : API=1, notebook=0, vLLM=0."
}

active_exec_target() {
  local pod
  pod="$(running_large_pods | head -n 1)"
  [[ -n "$pod" ]] || return 1
  printf 'pod/%s\n' "$pod"
}

ready_api_target() {
  local pod
  pod="$(
    "$kubectl_bin" get pods -n "$namespace" \
      -l app=prooftag-qr --field-selector=status.phase=Running \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
      | awk '$2 == "True" { print $1; exit }'
  )"
  [[ -n "$pod" ]] || return 1
  printf 'pod/%s\n' "$pod"
}

campaign_read_command() {
  local command="$1"
  shift
  local target jobs api_target
  if target="$(active_exec_target)"; then
    "$kubectl_bin" exec -i -n "$namespace" "$target" -c e046-large -- \
      python -m prooftag_qr.e046_large_campaign "$command" \
        --output-root "$output_root" "$@"
  else
    jobs="$(running_large_jobs)"
    if [[ -n "$jobs" ]]; then
      echo "Job actif mais pod pas encore exécutable; état Kubernetes affiché, état persistant momentanément indisponible." >&2
      return 0
    fi
    if ! api_target="$(ready_api_target)"; then
      echo "État persistant indisponible sans runtime actif; utiliser restore-runtime si cette consultation est nécessaire." >&2
      return 0
    fi
    "$kubectl_bin" exec -i -n "$namespace" "$api_target" -c api -- \
      python -m prooftag_qr.e046_large_campaign "$command" \
        --output-root "$output_root" "$@"
  fi
}

show_jobs() {
  echo "===== JOBS E046 LARGE ====="
  "$kubectl_bin" get jobs -n "$namespace" \
    -l "prooftag.io/experiment=$experiment" -o wide 2>/dev/null || true
  echo
  echo "===== PODS E046 LARGE ====="
  "$kubectl_bin" get pods -n "$namespace" \
    -l "prooftag.io/experiment=$experiment" -o wide 2>/dev/null || true
}

show_status() {
  show_jobs
  echo
  echo "===== ÉTAT PERSISTANT ====="
  campaign_read_command status
  echo
  echo "===== DISQUE ====="
  local target jobs api_target
  if target="$(active_exec_target)"; then
    "$kubectl_bin" exec -n "$namespace" "$target" -c e046-large -- \
      du -sh "$output_root" 2>/dev/null || true
  else
    jobs="$(running_large_jobs)"
    if [[ -n "$jobs" ]]; then
      echo "Mesure disque momentanément indisponible tant que le pod démarre."
    elif api_target="$(ready_api_target)"; then
      "$kubectl_bin" exec -n "$namespace" "$api_target" -c api -- \
        du -sh "$output_root" 2>/dev/null || true
    else
      echo "Mesure disque indisponible sans runtime actif."
    fi
  fi
}

show_progress() {
  campaign_read_command progress
}

show_logs() {
  local jobs job
  if [[ -n "$log_job" ]]; then
    "$kubectl_bin" get job "$log_job" -n "$namespace" >/dev/null
    jobs="$log_job"
  else
    jobs="$(
      "$kubectl_bin" get jobs -n "$namespace" \
        -l "prooftag.io/experiment=$experiment" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
        2>/dev/null | tail -n 5 || true
    )"
  fi
  if [[ -z "$jobs" ]]; then
    echo "Aucun Job E046 large."
    return 0
  fi
  while IFS= read -r job; do
    [[ -n "$job" ]] || continue
    echo
    echo "===== $job ====="
    "$kubectl_bin" logs -n "$namespace" job/"$job" \
      --all-containers=true --tail=500 2>/dev/null || true
  done <<<"$jobs"
}

write_pause_request() {
  local jobs
  jobs="$(running_large_jobs)"
  : >"$pause_file"
  # Le témoin /data est auditable et survit à une reconnexion SSH. Le témoin
  # /tmp permet au shell ordonnanceur courant de réagir sans réveiller l'API.
  if [[ -n "$jobs" ]]; then
    campaign_read_command pause-after-current || \
      echo "Témoin persistant indisponible; le témoin opérateur local reste actif." >&2
  else
    ensure_api
    campaign_read_command pause-after-current
  fi
  if [[ -n "$jobs" ]]; then
    echo "Pause demandée. Le Job courant finit; aucun nouveau Job ne sera lancé."
    printf 'Job courant : %s\n' "$jobs"
  else
    echo "Pause enregistrée avant le prochain Job. Aucun nouveau Job ne sera lancé."
  fi
}

cancel_current_job() {
  local jobs count job
  jobs="$(running_large_jobs)"
  count="$(printf '%s\n' "$jobs" | awk 'NF { count += 1 } END { print count + 0 }')"
  if [[ "$count" -eq 0 ]]; then
    echo "Aucun Job E046 large actif à arrêter." >&2
    return 1
  fi
  if [[ "$count" -ne 1 ]]; then
    echo "Refus : $count Jobs E046 large actifs; intervention manuelle requise." >&2
    printf '%s\n' "$jobs" >&2
    return 1
  fi
  job="$jobs"
  : >"$pause_file"
  campaign_read_command pause-after-current || true
  echo "Arrêt immédiat demandé pour $job. Les fichiers /data sont conservés."
  "$kubectl_bin" delete job "$job" -n "$namespace" \
    --cascade=foreground --wait=true --timeout=180s
}

load_runtime_identity() {
  local api_pod api_image_id pod_spec_image embedded_commit
  source_commit="$(git rev-parse HEAD)"
  ensure_api
  image_tag="$(
    "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
  )"
  deployed_commit="$(
    "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_GIT_COMMIT")].value}'
  )"
  build_image_digest="$(
    "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
      -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="PROOFTAG_RUNTIME_IMAGE_DIGEST")].value}'
  )"
  if [[ ! "$build_image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "Digest de build déclaré par l'API invalide : $build_image_digest" >&2
    exit 1
  fi
  if [[ "$deployed_commit" != "$source_commit" ]]; then
    echo "Image API et dépôt désynchronisés : API=$deployed_commit Git=$source_commit" >&2
    echo "Lancer : bash scripts/deploy-e046-large-dataset.sh deploy" >&2
    exit 1
  fi
  api_pod="$(
    "$kubectl_bin" get pods -n "$namespace" \
      -l app=prooftag-qr --field-selector=status.phase=Running \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' \
      | awk '$2 == "True" { print $1; exit }'
  )"
  [[ -n "$api_pod" ]] || {
    echo "Aucun pod API Ready pour établir l'identité immuable de l'image." >&2
    exit 1
  }
  pod_spec_image="$(
    "$kubectl_bin" get pod "$api_pod" -n "$namespace" \
      -o jsonpath='{.spec.containers[?(@.name=="api")].image}'
  )"
  if [[ "$pod_spec_image" != "$image_tag" ]]; then
    echo "Le pod Ready n'exécute pas le tag du Deployment : $pod_spec_image != $image_tag" >&2
    exit 1
  fi
  api_image_id="$(
    "$kubectl_bin" get pod "$api_pod" -n "$namespace" \
      -o jsonpath='{.status.containerStatuses[?(@.name=="api")].imageID}'
  )"
  embedded_commit="$(
    "$kubectl_bin" exec -i -n "$namespace" "$api_pod" -c api -- \
      python -c "from pathlib import Path; print(Path('/app/prooftag-build-commit.txt').read_text(encoding='ascii').strip())"
  )"
  if [[ ! "$embedded_commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Commit embarqué dans l'image invalide : $embedded_commit" >&2
    exit 1
  fi
  if [[ "$embedded_commit" != "$deployed_commit" || "$embedded_commit" != "$source_commit" ]]; then
    echo "Attestation commit incohérente : image=$embedded_commit env=$deployed_commit Git=$source_commit" >&2
    exit 1
  fi
  image="${api_image_id#docker-pullable://}"
  if [[ ! "$image" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "ImageID API non épinglable : $api_image_id" >&2
    exit 1
  fi
  image_digest="${image##*@}"
  # PROOFTAG_RUNTIME_IMAGE_DIGEST est le digest de configuration construit par
  # Docker. image_digest est ici le digest du manifeste réellement exécuté,
  # obtenu via status.containerStatuses.imageID et utilisable dans image@sha256.
}

latest_plan_metadata() {
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python - "$output_root/LATEST.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = json.loads((Path(payload["plan_dir"]) / "plan.json").read_text(encoding="utf-8"))
fields = (
    plan["plan_id"],
    plan["profile"],
    plan["source_commit"],
    plan["runtime_image"],
    plan["runtime_image_digest"],
)
print("\t".join(fields))
PY
}

plan_is_complete() {
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python -c \
      'import sys; from pathlib import Path; raise SystemExit(0 if Path(sys.argv[1]).is_file() else 1)' \
      "$output_root/$plan_id/COMPLETE.json"
}

create_plan() {
  echo "===== E046 LARGE — PLAN $profile ====="
  "$kubectl_bin" exec -i -n "$namespace" \
    deployment/"$api_deployment" -c api -- \
    python -m prooftag_qr.e046_large_campaign plan \
      --output-root "$output_root" \
      --profile "$profile" \
      --source-commit "$source_commit" \
      --runtime-image "$image" \
      --runtime-image-digest "$image_digest"
}

wait_job() {
  local job="$1"
  local timeout="$2"
  local started elapsed exists active succeeded failed state
  started="$(date +%s)"
  while true; do
    state="$(job_state "$job")"
    IFS=$'\t' read -r exists active succeeded failed <<<"$state"
    if [[ "$exists" -eq 0 ]]; then
      echo "Le Job $job n'existe plus. Les données partielles /data restent intactes." >&2
      return 1
    fi
    elapsed="$(( $(date +%s) - started ))"
    printf '[E046-large:%s] elapsed=%ss active=%s succeeded=%s failed=%s\n' \
      "$job" "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
    if [[ "${succeeded:-0}" -ge 1 ]]; then
      return 0
    fi
    if [[ "${failed:-0}" -ge 1 ]]; then
      echo "Job $job en échec. Tentatives et checkpoints conservés dans /data." >&2
      "$kubectl_bin" logs -n "$namespace" job/"$job" \
        --all-containers=true --tail=1200 || true
      return 1
    fi
    if [[ "$elapsed" -ge "$timeout" ]]; then
      echo "Timeout opérateur $job après ${elapsed}s. Le Job n'est pas supprimé." >&2
      return 1
    fi
    sleep 30
  done
}

job_state() {
  local job="$1"
  local payload
  payload="$(
    "$kubectl_bin" get job "$job" -n "$namespace" \
      --ignore-not-found -o json
  )"
  if [[ -z "$payload" ]]; then
    printf '0\t0\t0\t0\n'
    return 0
  fi
  printf '%s' "$payload" | "$host_python" -c '
import json, sys
job = json.load(sys.stdin)
status = job.get("status", {})
print("\t".join(str(int(status.get(key) or 0)) for key in ("active", "succeeded", "failed")))
' | awk '{ print "1\t" $0 }'
}

job_name_for() {
  local kind="$1"
  local task_id="$2"
  local recipe_id="${3:-}"
  local digest prefix
  digest="$(printf '%s' "${kind}:${task_id}:${recipe_id}:${plan_id}" | sha256sum | cut -c1-12)"
  prefix="p"
  [[ "$kind" == "refinement" ]] && prefix="r"
  printf 'prooftag-qr-e046-large-%s-%s\n' "$prefix" "$digest"
}

job_has_terminal_srl_failure() {
  local job="$1"
  "$kubectl_bin" logs -n "$namespace" job/"$job" \
    --all-containers=true 2>/dev/null | grep -Fq "$terminal_srl_pattern"
}

start_or_attach_job() {
  local kind="$1"
  local task_id="$2"
  local recipe_id="${3:-}"
  local job template tmp active succeeded failed exists foreign timeout state
  job="$(job_name_for "$kind" "$task_id" "$recipe_id")"
  template="deploy/k8s/e046-large-parent-job.yaml"
  [[ "$kind" == "refinement" ]] && \
    template="deploy/k8s/e046-large-refinement-job.yaml"

  state="$(job_state "$job")"
  IFS=$'\t' read -r exists active succeeded failed <<<"$state"

  if [[ "${active:-0}" -ge 1 ]]; then
    echo "Rattachement au Job actif : $job"
  elif [[ "${failed:-0}" -ge 1 ]]; then
    echo "Le Job $job a déjà échoué avec cette spécification." >&2
    if [[ "$kind" == "refinement" ]] && job_has_terminal_srl_failure "$job"; then
      if [[ "$override_srl_terminal" != "1" ]]; then
        echo "Échec scientifique SRL terminal : la relance générique est interdite." >&2
        echo "Après correction et audit de fidélité seulement :" >&2
        echo "  PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL=1 bash scripts/run-e046-large-dataset.sh resume" >&2
        return 1
      fi
      echo "Reprise manuelle de l'échec SRL terminal explicitement autorisée."
    elif [[ "$retry_failed_jobs" != "1" ]]; then
      echo "Aucune relance identique automatique." >&2
      echo "Après diagnostic d'un incident transitoire seulement :" >&2
      echo "  PROOFTAG_E046_LARGE_RETRY_FAILED=1 bash scripts/run-e046-large-dataset.sh resume" >&2
      return 1
    fi
    echo "Relance explicitement autorisée; aucun fichier /data n'est supprimé."
    "$kubectl_bin" delete job "$job" -n "$namespace" \
      --ignore-not-found --wait=true >/dev/null
    exists=0
  elif [[ "${succeeded:-0}" -ge 1 ]]; then
    if [[ "$kind" == "refinement" && "$override_srl_terminal" == "1" ]] && \
        job_has_terminal_srl_failure "$job"; then
      echo "Job terminal sans promotion : reprise SRL manuelle explicitement autorisée."
      "$kubectl_bin" delete job "$job" -n "$namespace" \
        --ignore-not-found --wait=true >/dev/null
      exists=0
    else
      echo "Job $job réussi mais marqueur scientifique absent." >&2
      echo "État incohérent : relance automatique refusée." >&2
      return 1
    fi
  elif [[ "$exists" -eq 1 ]]; then
    echo "Job existant en attente de planification : $job"
  fi

  if [[ "$exists" -eq 0 ]]; then
    if [[ -e "$pause_file" ]]; then
      echo "Pause opérateur active : aucun nouveau Job lancé."
      return 75
    fi
    assert_at_most_one_large_job
    stop_gpu_services
    foreign="$(foreign_gpu_pods)"
    if [[ -n "$foreign" ]]; then
      echo "Une autre charge GPU est active; E046 large ne démarre pas :" >&2
      printf '%s\n' "$foreign" >&2
      return 1
    fi
    tmp="$(mktemp)"
    sed \
      -e "s|__JOB_NAME__|$job|g" \
      -e "s|__NAMESPACE__|$namespace|g" \
      -e "s|__IMAGE__|$image|g" \
      -e "s|__OUTPUT_ROOT__|$output_root|g" \
      -e "s|__PLAN_ID__|$plan_id|g" \
      -e "s|__CANDIDATE_ID__|$task_id|g" \
      -e "s|__RECIPE_ID__|$recipe_id|g" \
      -e "s|__SOURCE_COMMIT__|$source_commit|g" \
      -e "s|__IMAGE_DIGEST__|$image_digest|g" \
      -e "s|__OVERRIDE_SRL_TERMINAL__|$override_srl_terminal|g" \
      "$template" >"$tmp"
    "$kubectl_bin" apply -f "$tmp" >/dev/null
    rm -f "$tmp"
  else
    # Un Job existant actif ou Pending doit lui aussi rester seul sur la RTX.
    stop_gpu_services
  fi

  timeout="$parent_timeout"
  [[ "$kind" == "refinement" ]] && timeout="$refinement_timeout"
  wait_job "$job" "$timeout"
  "$kubectl_bin" logs -n "$namespace" job/"$job" \
    --all-containers=true --tail=250 || true
}

session_progress() {
  local label="$1"
  local completed="$2"
  local total="$3"
  local started="$4"
  local elapsed average remaining eta percent
  elapsed="$(( $(date +%s) - started ))"
  average="$(( completed > 0 ? elapsed / completed : 0 ))"
  remaining="$(( total - completed ))"
  eta="$(( average * remaining ))"
  percent="$(awk -v done="$completed" -v all="$total" 'BEGIN { printf "%.1f", all ? (100 * done / all) : 100 }')"
  printf '[%s] %s/%s (%s%%), moyenne=%ss/tâche, ETA session=%ss\n' \
    "$label" "$completed" "$total" "$percent" "$average" "$eta"
}

pause_if_requested() {
  if [[ -e "$pause_file" ]]; then
    echo "Pause opérateur appliquée après le Job courant."
    echo "Reprendre : bash scripts/run-e046-large-dataset.sh resume"
    return 0
  fi
  return 1
}

on_signal() {
  : >"$pause_file"
  echo >&2
  echo "Interruption reçue : le Job Kubernetes courant continue, aucun suivant ne sera lancé." >&2
  exit 130
}

require_repo
validate_profile

for historical_output_root in \
  "$historical_e045_output_root" "$historical_e046_output_root"; do
  if [[ "$output_root" == "$historical_output_root" || \
        "$output_root" == "$historical_output_root"/* ]]; then
    echo "Refus : les dossiers historiques E045/E046 et leurs enfants sont intouchables." >&2
    exit 1
  fi
done

if [[ ! "$parent_timeout" =~ ^[0-9]+$ || "$parent_timeout" -lt 1 || "$parent_timeout" -gt 10800 ]]; then
  echo "Timeout parent invalide : entier compris entre 1 et 10800 secondes." >&2
  exit 1
fi
if [[ ! "$refinement_timeout" =~ ^[0-9]+$ || "$refinement_timeout" -lt 1 || "$refinement_timeout" -gt 14400 ]]; then
  echo "Timeout refinement invalide : entier compris entre 1 et 14400 secondes." >&2
  exit 1
fi
if [[ "$override_srl_terminal" != "0" && "$override_srl_terminal" != "1" ]]; then
  echo "PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL doit valoir 0 ou 1." >&2
  exit 1
fi

case "$action" in
  status)
    show_status
    exit 0
    ;;
  progress)
    show_progress
    exit 0
    ;;
  logs)
    show_logs
    exit 0
    ;;
  verify)
    assert_at_most_one_large_job
    verify_jobs="$(running_large_jobs)"
    if [[ -n "$verify_jobs" ]]; then
      echo "Un Job E046 large est actif : vérification différée pour ne pas démarrer l'API." >&2
      exit 1
    fi
    ensure_api
    campaign_read_command verify
    exit 0
    ;;
  pause-after-current)
    write_pause_request
    exit $?
    ;;
  cancel-current)
    cancel_current_job
    exit $?
    ;;
  restore-runtime)
    assert_at_most_one_large_job
    restore_runtime_if_idle
    exit $?
    ;;
  plan|run|resume)
    ;;
  *)
    echo "Action inconnue : $action" >&2
    echo "Actions : plan, run, resume, status, progress, logs [job], verify, pause-after-current, cancel-current, restore-runtime" >&2
    exit 2
    ;;
esac

assert_at_most_one_large_job

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant E046 large." >&2
  exit 1
fi

trap on_signal INT TERM HUP

if [[ "$action" == "plan" || "$action" == "run" || "$action" == "resume" ]]; then
  command -v flock >/dev/null 2>&1 || {
    echo "flock est requis pour garantir un seul ordonnanceur E046 large." >&2
    exit 1
  }
  exec 9>"$runner_lock_file"
  if ! flock -n 9; then
    echo "Un autre ordonnanceur E046 large est déjà actif." >&2
    echo "Utiliser status/progress, pas un second run/resume." >&2
    exit 1
  fi
fi

resume_active_jobs=""
if [[ "$action" == "resume" ]]; then
  resume_active_jobs="$(running_large_jobs)"
fi
if [[ -n "$resume_active_jobs" ]]; then
  current_job="$resume_active_jobs"
  current_kind="$(
    "$kubectl_bin" get job "$current_job" -n "$namespace" \
      -o jsonpath='{.metadata.labels.prooftag\.io/task-kind}'
  )"
  current_timeout="$parent_timeout"
  [[ "$current_kind" == "refinement" ]] && current_timeout="$refinement_timeout"
  echo "Reprise : rattachement au Job actif $current_job."
  stop_gpu_services
  resume_wait_exit() {
    local code="$?"
    local jobs pods
    [[ "$code" -eq 0 ]] && return 0
    if ! jobs="$(running_large_jobs)" || ! pods="$(nonterminal_large_pods)"; then
      echo "État Kubernetes inconnu après l'échec de rattachement; API laissée à 0." >&2
      return "$code"
    fi
    if [[ -z "$jobs" && -z "$pods" ]]; then
      restore_runtime_if_idle || true
    else
      echo "Le Job est potentiellement encore actif; API, notebook et vLLM restent à 0." >&2
    fi
    return "$code"
  }
  trap resume_wait_exit EXIT
  wait_job "$current_job" "$current_timeout"
  trap - EXIT
fi

load_runtime_identity

if [[ "$action" == "plan" ]]; then
  create_plan
fi
IFS=$'\t' read -r plan_id planned_profile planned_commit planned_image planned_digest \
  < <(latest_plan_metadata)
if [[ "$action" == "resume" ]]; then
  profile="$planned_profile"
fi

if [[ "$action" == "run" && "$planned_profile" != "$profile" ]]; then
  echo "Le dernier plan est '$planned_profile', mais le profil demandé est '$profile'." >&2
  echo "Créer/revoir d'abord le bon plan :" >&2
  echo "  PROOFTAG_E046_LARGE_PROFILE=$profile bash scripts/run-e046-large-dataset.sh plan" >&2
  exit 1
fi

if [[ "$action" != "plan" && (
      "$planned_commit" != "$source_commit" ||
      "$planned_image" != "$image" ||
      "$planned_digest" != "$image_digest"
    ) ]]; then
  echo "Le plan gelé et le runtime courant ne correspondent pas." >&2
  echo "Plan    : commit=$planned_commit image=$planned_image digest=$planned_digest" >&2
  echo "Runtime : commit=$source_commit image=$image digest=$image_digest" >&2
  echo "Ne pas migrer silencieusement : créer un nouveau plan avec l'image courante." >&2
  exit 1
fi

if [[ "$action" != "plan" ]] && plan_is_complete; then
  echo "Plan déjà COMPLETE : vérification en lecture seule, aucun scoring ni agrégat réécrit."
  campaign_read_command verify --plan-id "$plan_id"
  exit 0
fi

if [[ "$action" != "plan" && "$planned_profile" == "full" && \
      "${PROOFTAG_E046_LARGE_CONFIRM_FULL:-}" != "RUN-3072" ]]; then
  echo "Ce plan full contient 3 072 parents et n'est jamais lancé implicitement." >&2
  echo "Après lecture du plan, confirmer explicitement :" >&2
  echo "  PROOFTAG_E046_LARGE_PROFILE=full PROOFTAG_E046_LARGE_CONFIRM_FULL=RUN-3072 bash scripts/run-e046-large-dataset.sh $action" >&2
  exit 1
fi

echo "PLAN_ID=$plan_id"
echo "PROFILE=$profile"
echo "IMAGE=$image"
echo "IMAGE_DIGEST=$image_digest"
echo "BUILD_IMAGE_DIGEST=$build_image_digest"
echo "COMMIT=$source_commit"
echo "OUTPUT_ROOT=$output_root"

if [[ "$action" == "plan" ]]; then
  echo "Aucune génération lancée."
  echo "Démarrer ce profil : PROOFTAG_E046_LARGE_PROFILE=$profile bash scripts/run-e046-large-dataset.sh run"
  exit 0
fi

# Une commande run/resume explicite acquitte les témoins de pause local et
# persistant avant de demander les tâches encore réellement Pending.
rm -f "$pause_file"
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign clear-pause \
    --output-root "$output_root" --plan-id "$plan_id" >/dev/null

parent_list="$(mktemp)"
refinement_list="$(mktemp)"
cleanup() {
  rm -f "$parent_list" "$refinement_list"
}
on_exit() {
  local exit_code="$?"
  local active_jobs active_pods
  cleanup
  if ! active_jobs="$(running_large_jobs)"; then
    echo "État Kubernetes inconnu : restauration GPU refusée par sécurité." >&2
    return "$exit_code"
  fi
  if ! active_pods="$(nonterminal_large_pods)"; then
    echo "État des pods inconnu : restauration GPU refusée par sécurité." >&2
    return "$exit_code"
  fi
  if [[ -z "$active_jobs" && -z "$active_pods" ]]; then
    restore_runtime_if_idle >/dev/null 2>&1 || true
  else
    echo "Un Job E046 large continue; API, notebook et vLLM restent à 0." >&2
    echo "Suivre : bash scripts/run-e046-large-dataset.sh status" >&2
    echo "Puis reprendre : bash scripts/run-e046-large-dataset.sh resume" >&2
  fi
  return "$exit_code"
}
trap on_exit EXIT

# Le moteur ne renvoie ici que les tâches sans promotion atomique valide :
# GENERATION_COMPLETE + empreintes pour le GPU, SCORING_COMPLETE + empreintes
# pour le scoring. Une tâche promue n'est donc jamais recalculée au resume.
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign list-parents \
    --output-root "$output_root" \
    --plan-id "$plan_id" \
    --pending-only >"$parent_list"

parent_total="$(awk 'NF { count += 1 } END { print count + 0 }' "$parent_list")"
parent_done=0
parent_started="$(date +%s)"
echo "===== PHASE A — $parent_total PARENTS GPU EN ATTENTE ====="
while IFS= read -r candidate_id; do
  [[ -n "$candidate_id" ]] || continue
  echo
  echo "----- parent $candidate_id -----"
  start_or_attach_job parent "$candidate_id"
  parent_done="$((parent_done + 1))"
  session_progress parents "$parent_done" "$parent_total" "$parent_started"
  pause_if_requested && exit 0
done <"$parent_list"

echo "===== PHASE A — SCORING STAGE2 RAW ====="
ensure_api
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign score-parents \
    --output-root "$output_root" --plan-id "$plan_id"

echo "===== PHASE B — SÉLECTION INFORMATIVE SR-MPGD ====="
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign select-refinements \
    --output-root "$output_root" --plan-id "$plan_id"
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  env \
    "PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL=$override_srl_terminal" \
  python -m prooftag_qr.e046_large_campaign list-refinements \
    --output-root "$output_root" \
    --plan-id "$plan_id" \
    --pending-only >"$refinement_list"

refinement_total="$(awk 'NF { count += 1 } END { print count + 0 }' "$refinement_list")"
refinement_done=0
refinement_started="$(date +%s)"
echo "===== PHASE B — $refinement_total RAFFINEMENTS GPU EN ATTENTE ====="
while IFS=$'\t' read -r candidate_id recipe_id; do
  [[ -n "$candidate_id" && -n "$recipe_id" ]] || continue
  echo
  echo "----- SR-MPGD $candidate_id / $recipe_id -----"
  start_or_attach_job refinement "$candidate_id" "$recipe_id"
  refinement_done="$((refinement_done + 1))"
  session_progress refinements "$refinement_done" "$refinement_total" "$refinement_started"
  pause_if_requested && exit 0
done <"$refinement_list"

echo "===== PHASE B — SCORING ET DATASET ADVISOR ====="
ensure_api
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign score-refinements \
    --output-root "$output_root" --plan-id "$plan_id"
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign aggregate \
    --output-root "$output_root" --plan-id "$plan_id"
"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- \
  python -m prooftag_qr.e046_large_campaign verify \
    --output-root "$output_root" --plan-id "$plan_id"

trap - EXIT
cleanup
restore_runtime_if_idle >/dev/null

echo "===== E046 LARGE TERMINÉ ====="
echo "Plan     : $plan_id"
echo "Résultat : $output_root/$plan_id"
echo "Ancien E045 resté intact : /data/e045-foundation-v1"
echo "Ancien E046 resté intact : /data/e046-controlled-best-generator-v1"
