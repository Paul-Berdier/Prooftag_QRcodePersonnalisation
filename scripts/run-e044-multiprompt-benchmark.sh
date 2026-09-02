#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then echo "Ne pas sourcer ce script." >&2; return 2; fi
set -Eeuo pipefail
ns="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
nb="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
k="${KUBECTL:-kubectl}"
root="${PROOFTAG_E044_RESULTS_ROOT:-/data/e044-multi-prompt-best-pipeline-v1}"
[[ -z "$(git status --porcelain)" ]] || { echo "Dépôt non propre." >&2; exit 1; }
$k scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
$k rollout status deployment/"$api" -n "$ns" --timeout=1200s
image="$($k get deployment "$api" -n "$ns" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
commit="$(git rev-parse HEAD)"
prompts=(p01_greenhouse p02_blue_vase p03_lighthouse p04_brutalist_grid p05_sleeping_cat p06_mycelium p07_winter_cabin)
echo "===== E044 MULTI-PROMPT ====="
echo "Image: $image"
echo "Root : $root"
echo "Commit: $commit"
echo "7 prompts × 2 gamma × 9 checkpoints = 126 checkpoints SR-MPGD"
$k scale deployment "$nb" -n "$ns" --replicas=0 >/dev/null || true
$k scale deployment vllm -n vllm --replicas=0 >/dev/null || true
$k scale deployment "$api" -n "$ns" --replicas=0 >/dev/null
for pid in "${prompts[@]}"; do
  # Check completion through a temporary lightweight API scale-up only when needed.
  $k scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
  $k rollout status deployment/"$api" -n "$ns" --timeout=1200s >/dev/null
  if $k exec -n "$ns" deployment/"$api" -c api -- test -f "$root/prompts/$pid/COMPLETE.json"; then
    echo "[E044] $pid déjà complet -> skip"
    $k scale deployment "$api" -n "$ns" --replicas=0 >/dev/null
    continue
  fi
  $k scale deployment "$api" -n "$ns" --replicas=0 >/dev/null
  job="prooftag-qr-e044-${pid//_/-}"
  tmp="$(mktemp)"
  sed -e "s|__JOB_NAME__|$job|g" -e "s|__NAMESPACE__|$ns|g" -e "s|__IMAGE__|$image|g" -e "s|__RESULTS_ROOT__|$root|g" -e "s|__PROMPT_ID__|$pid|g" -e "s|__SOURCE_COMMIT__|$commit|g" deploy/k8s/e044-prompt-job.yaml > "$tmp"
  $k delete job "$job" -n "$ns" --ignore-not-found >/dev/null
  $k apply -f "$tmp" >/dev/null
  rm -f "$tmp"
  started=$(date +%s); timeout=${PROOFTAG_E044_PROMPT_TIMEOUT_SECONDS:-21600}
  while true; do
    s="$($k get job "$job" -n "$ns" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"; f="$($k get job "$job" -n "$ns" -o jsonpath='{.status.failed}' 2>/dev/null || true)"; a="$($k get job "$job" -n "$ns" -o jsonpath='{.status.active}' 2>/dev/null || true)"; e=$(( $(date +%s)-started ))
    printf '[E044:%s] elapsed=%ss active=%s succeeded=%s failed=%s\n' "$pid" "$e" "${a:-0}" "${s:-0}" "${f:-0}"
    [[ "${s:-0}" -ge 1 ]] && break
    if [[ "${f:-0}" -ge 1 ]]; then $k logs -n "$ns" job/"$job" --all-containers=true --tail=1000 || true; exit 1; fi
    if [[ "$e" -ge "$timeout" ]]; then echo "Timeout $pid" >&2; exit 1; fi
    sleep 30
  done
  $k logs -n "$ns" job/"$job" --all-containers=true --tail=300 || true
done
$k scale deployment "$api" -n "$ns" --replicas=1 >/dev/null
$k rollout status deployment/"$api" -n "$ns" --timeout=1200s
$k exec -i -n "$ns" deployment/"$api" -c api -- python -m prooftag_qr.e044_aggregate --root "$root" --source-commit "$commit"
$k exec -n "$ns" deployment/"$api" -c api -- test -f "$root/verdict.json"
echo "===== E044 TERMINÉ ====="
echo "Résultats: $root"
echo "Windows: .\\scripts\\e044-remote.ps1"
echo "Atlas  : .\\scripts\\e044-remote.ps1 -Atlas"
