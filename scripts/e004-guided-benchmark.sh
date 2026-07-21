#!/usr/bin/env bash
set -euo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
timeout="${PROOFTAG_QR_ROLLOUT_TIMEOUT:-900s}"
baseline_log="$(mktemp)"
guided_log="$(mktemp)"

restore_defaults() {
  kubectl set env "deployment/${deployment}" -n "$namespace" \
    PROOFTAG_QR_GUIDED_REDIFFUSION_ENABLED- \
    PROOFTAG_QR_LATENT_REFINEMENT_ENABLED- >/dev/null
  kubectl rollout status "deployment/${deployment}" -n "$namespace" --timeout="$timeout"
}

set_mode() {
  local guided="$1"
  local latent="$2"
  kubectl set env "deployment/${deployment}" -n "$namespace" \
    "PROOFTAG_QR_GUIDED_REDIFFUSION_ENABLED=${guided}" \
    "PROOFTAG_QR_LATENT_REFINEMENT_ENABLED=${latent}"
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
  rm -f -- "$baseline_log" "$guided_log"
  if [[ "$status" -eq 0 && "$restore_status" -ne 0 ]]; then
    status="$restore_status"
  fi
  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "E004 1/2 - baseline du même commit, deux raffinements désactivés"
set_mode false false
make benchmark | tee "$baseline_log"
baseline_archive="$(archive_from_output < "$baseline_log")"
if [[ -z "$baseline_archive" ]]; then
  echo "Impossible de trouver BENCHMARK_ARCHIVE dans la sortie baseline" >&2
  exit 1
fi

echo "E004 2/2 - seconde diffusion guidée puis SR-MPGD"
set_mode true true
make benchmark | tee "$guided_log"
guided_archive="$(archive_from_output < "$guided_log")"
if [[ -z "$guided_archive" ]]; then
  echo "Impossible de trouver BENCHMARK_ARCHIVE dans la sortie E004" >&2
  exit 1
fi

echo "E004_BASELINE_ARCHIVE=${baseline_archive}"
echo "E004_GUIDED_ARCHIVE=${guided_archive}"
