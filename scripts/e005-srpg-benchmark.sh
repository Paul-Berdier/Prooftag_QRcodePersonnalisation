#!/usr/bin/env bash
set -euo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
timeout="${PROOFTAG_QR_ROLLOUT_TIMEOUT:-1200s}"
baseline_log="$(mktemp)"
srpg_log="$(mktemp)"

restore_defaults() {
  kubectl set env "deployment/${deployment}" -n "$namespace" \
    PROOFTAG_QR_GUIDED_REDIFFUSION_ENABLED- \
    PROOFTAG_QR_SRPG_ENABLED- \
    PROOFTAG_QR_LATENT_REFINEMENT_ENABLED- >/dev/null
  kubectl rollout status "deployment/${deployment}" -n "$namespace" --timeout="$timeout"
}

set_mode() {
  local srpg="$1"
  kubectl set env "deployment/${deployment}" -n "$namespace" \
    "PROOFTAG_QR_GUIDED_REDIFFUSION_ENABLED=false" \
    "PROOFTAG_QR_SRPG_ENABLED=${srpg}" \
    "PROOFTAG_QR_LATENT_REFINEMENT_ENABLED=false"
  kubectl rollout status "deployment/${deployment}" -n "$namespace" --timeout="$timeout"
}

archive_from_output() {
  sed -n 's/^BENCHMARK_ARCHIVE=//p' | tail -n 1
}

cleanup() {
  local status="$?"
  local restore_status=0
  trap - EXIT
  set +e
  restore_defaults
  restore_status="$?"
  rm -f -- "$baseline_log" "$srpg_log"
  if [[ "$status" -eq 0 && "$restore_status" -ne 0 ]]; then
    status="$restore_status"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "E005 1/2 - baseline du même commit, tous les raffinements désactivés"
set_mode false
PROOFTAG_QR_BENCHMARK_MAX_ATTEMPTS=1 make benchmark | tee "$baseline_log"
baseline_archive="$(archive_from_output < "$baseline_log")"
if [[ -z "$baseline_archive" ]]; then
  echo "Impossible de trouver BENCHMARK_ARCHIVE dans la sortie baseline" >&2
  exit 1
fi

echo "E005 2/2 - véritable boucle DDIM avec SRPG, sans autre raffinement"
set_mode true
PROOFTAG_QR_BENCHMARK_MAX_ATTEMPTS=1 make benchmark | tee "$srpg_log"
srpg_archive="$(archive_from_output < "$srpg_log")"
if [[ -z "$srpg_archive" ]]; then
  echo "Impossible de trouver BENCHMARK_ARCHIVE dans la sortie E005" >&2
  exit 1
fi

echo "E005_BASELINE_ARCHIVE=${baseline_archive}"
echo "E005_SRPG_ARCHIVE=${srpg_archive}"
