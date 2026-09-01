#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/run-e042-decoder-diagnostic.sh" >&2
  return 2
fi
set -Eeuo pipefail
namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
decode_job="${PROOFTAG_E042_DECODE_JOB_NAME:-prooftag-qr-e042-decode}"
diagnose_job="${PROOFTAG_E042_DIAGNOSE_JOB_NAME:-prooftag-qr-e042-diagnose}"
e041_results_dir="${PROOFTAG_E042_E041_RESULTS_DIR:-/data/e041-gamma-functional-pattern-frontier-v1}"
results_dir="${PROOFTAG_E042_RESULTS_DIR:-/data/e042-decoder-failure-localization-v1}"
kubectl_bin="${KUBECTL:-kubectl}"
[[ -f deploy/k8s/e042-decode-selected-latents-job.yaml ]] || { echo "Lancer depuis la racine du dépôt." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Le dépôt contient des modifications non commitées." >&2; exit 1; }

previous_api_replicas="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.replicas}')"
"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1 >/dev/null
"$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
image="$($kubectl_bin get deployment "$api_deployment" -n "$namespace" -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}')"
source_commit="$(git rev-parse HEAD)"

"$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$e041_results_dir/verdict.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); assert p.is_file(), p
v=json.loads(p.read_text(encoding='utf-8'))
assert v['experiment']=='e041-gamma-functional-pattern-frontier-v1'
assert v['phase_a_checkpoint_count']==54
assert v['generalization_authorized'] is False
assert v['production_ready'] is False
print('E041 vérifié:', 'PhaseA=', v['phase_a_checkpoint_count'], 'winnerSSR=', f"{v['winner_ssr_exact_presets']}/37")
PY

verdict_state="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c "test -f '$results_dir/verdict.json' && echo COMPLETE || echo INCOMPLETE")"
if [[ "$verdict_state" == "COMPLETE" ]]; then
  echo "E042 est déjà terminé : $results_dir/verdict.json"
  "$kubectl_bin" exec -n "$namespace" deployment/"$api_deployment" -c api -- cat "$results_dir/verdict.json"
  exit 0
fi

restored=0
cleanup() {
  code="$?"
  if [[ "$restored" -eq 0 ]]; then
    "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}" >/dev/null || true
    if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
      "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s || true
    fi
  fi
  exit "$code"
}
trap cleanup EXIT

wait_job() {
  local job="$1" label="$2" timeout="$3"
  local started elapsed succeeded failed active
  started="$(date +%s)"
  while true; do
    succeeded="$($kubectl_bin get job "$job" -n "$namespace" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$($kubectl_bin get job "$job" -n "$namespace" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    active="$($kubectl_bin get job "$job" -n "$namespace" -o jsonpath='{.status.active}' 2>/dev/null || true)"
    elapsed="$(( $(date +%s) - started ))"
    printf '[%s] elapsed=%ss active=%s succeeded=%s failed=%s\n' "$label" "$elapsed" "${active:-0}" "${succeeded:-0}" "${failed:-0}"
    if [[ "${succeeded:-0}" -ge 1 ]]; then return 0; fi
    if [[ "${failed:-0}" -ge 1 ]]; then
      echo "Job $job en échec." >&2
      "$kubectl_bin" logs -n "$namespace" job/"$job" --all-containers=true --tail=700 || true
      "$kubectl_bin" describe job "$job" -n "$namespace" || true
      return 1
    fi
    if [[ "$elapsed" -ge "$timeout" ]]; then
      echo "Timeout $job après ${elapsed}s." >&2
      "$kubectl_bin" get job,pod -n "$namespace" -l "prooftag.io/experiment=e042-decoder-failure-localization-v1" -o wide || true
      return 1
    fi
    sleep 15
  done
}

render_job() {
  local template="$1" job="$2" target="$3"
  sed \
    -e "s|__NAMESPACE__|$namespace|g" \
    -e "s|__JOB_NAME__|$job|g" \
    -e "s|__IMAGE__|$image|g" \
    -e "s|__E041_RESULTS_DIR__|$e041_results_dir|g" \
    -e "s|__RESULTS_DIR__|$results_dir|g" \
    -e "s|__SOURCE_COMMIT__|$source_commit|g" \
    "$template" > "$target"
}

echo "===== E042 — localisation de l'échec décodeur ====="
echo "Image       : $image"
echo "E041        : $e041_results_dir"
echo "Sortie      : $results_dir"
echo "États       : parent + gamma500/1000/2000 sélectionnés (9)"
echo "But         : détection -> quiet zone -> binarisation -> grille -> bits/ECC"
echo "Optimisation: AUCUNE"

"$kubectl_bin" scale deployment "$notebook_deployment" -n "$namespace" --replicas=0 >/dev/null || true
"$kubectl_bin" scale deployment vllm -n vllm --replicas=0 >/dev/null || true

# Phase GPU courte: seulement si les neuf raw VAE n'ont pas déjà été re-décodés.
decode_complete="$($kubectl_bin exec -n "$namespace" deployment/"$api_deployment" -c api -- sh -c "test -f '$results_dir/decode/complete.json' && echo YES || echo NO")"
if [[ "$decode_complete" != "YES" ]]; then
  tmp_decode="$(mktemp)"
  render_job deploy/k8s/e042-decode-selected-latents-job.yaml "$decode_job" "$tmp_decode"
  echo "===== PHASE 1/2 — VAE re-decode GPU (pas de SR-MPGD) ====="
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
  "$kubectl_bin" wait --for=delete pod -n "$namespace" -l app=prooftag-qr --timeout=600s || true
  "$kubectl_bin" delete job "$decode_job" -n "$namespace" --ignore-not-found
  "$kubectl_bin" apply -f "$tmp_decode"
  rm -f "$tmp_decode"
  wait_job "$decode_job" E042-DECODE "${PROOFTAG_E042_DECODE_TIMEOUT_SECONDS:-3600}"
  echo "===== LOGS DECODE E042 ====="
  "$kubectl_bin" logs -n "$namespace" job/"$decode_job" --all-containers=true --tail=400 || true
else
  echo "Phase decode déjà complète : réutilisation des raw VAE E042."
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0 >/dev/null
fi

# Phase CPU: QR-Verify et diagnostics. Aucun GPU demandé.
tmp_diag="$(mktemp)"
render_job deploy/k8s/e042-decoder-diagnostic-job.yaml "$diagnose_job" "$tmp_diag"
echo "===== PHASE 2/2 — diagnostic scanners CPU ====="
"$kubectl_bin" delete job "$diagnose_job" -n "$namespace" --ignore-not-found
"$kubectl_bin" apply -f "$tmp_diag"
rm -f "$tmp_diag"
wait_job "$diagnose_job" E042-DIAG "${PROOFTAG_E042_DIAGNOSE_TIMEOUT_SECONDS:-7200}"
echo "===== LOGS DIAGNOSTIC E042 ====="
"$kubectl_bin" logs -n "$namespace" job/"$diagnose_job" --all-containers=true --tail=700 || true

"$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas="${previous_api_replicas:-1}"
if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" rollout status deployment/"$api_deployment" -n "$namespace" --timeout=1200s
fi
restored=1

if [[ "${previous_api_replicas:-1}" -gt 0 ]]; then
  "$kubectl_bin" exec -i -n "$namespace" deployment/"$api_deployment" -c api -- python - "$results_dir/verdict.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); assert p.is_file(), p
v=json.loads(p.read_text(encoding='utf-8'))
assert v['experiment']=='e042-decoder-failure-localization-v1'
assert v['diagnostic_only'] is True
assert v['production_ready'] is False
assert v['generalization_authorized'] is False
print(json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True))
PY
else
  echo "API était initialement à 0 replica ; verdict disponible dans le PVC : $results_dir/verdict.json"
fi
trap - EXIT

echo "===== E042 TERMINÉ ====="
echo "Résultats : $results_dir"
echo "PowerShell : .\\scripts\\e042-remote.ps1"
