#!/usr/bin/env bash

# E035 orchestration: CPU notebook + two mutually exclusive one-shot GPU Jobs.
# 1) capture-parent runs Stage 2 once from the hash-verified E034 Stage-1 PNG;
# 2) run loads the frozen latent and compares paper SRL vs official upstream SRL.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Ne pas sourcer ce script. Utiliser : bash scripts/deploy-e035-notebook.sh <commande>" >&2
  return 2
fi
set -Eeuo pipefail

command_name="${1:-help}"
namespace="${PROOFTAG_QR_NAMESPACE:-${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}}"
api_deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
notebook_deployment="${PROOFTAG_QR_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
job_name="${PROOFTAG_E035_JOB_NAME:-prooftag-qr-e035}"
capture_job_name="${PROOFTAG_E035_CAPTURE_JOB_NAME:-prooftag-qr-e035-parent-capture}"
notebook="30_e035_srmpgd_loss_fidelity_gate.ipynb"
parent_dir="${PROOFTAG_E035_PARENT_DIR:-/data/e035-parent-v1}"
capture_audit_dir="${PROOFTAG_E035_CAPTURE_AUDIT_DIR:-/data/e035-parent-capture-audit-v2}"
results_dir="${PROOFTAG_E035_RESULTS_DIR:-/data/e035-loss-fidelity-gate-v1}"
expected_parent_commit="${PROOFTAG_E035_PARENT_COMMIT:-}"
local_stage1_asset="${PROOFTAG_E035_STAGE1_ASSET:-docs/e035-assets/e034-observed-stage1.png}"
remote_stage1_asset="${PROOFTAG_E035_STAGE1_REMOTE:-/data/e035-input/e034-observed-stage1.png}"
stage1_image_sha256="ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
stage1_file_sha256="be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71"
kubectl_bin="${KUBECTL:-kubectl}"
verified_parent_commit_value=""

usage() {
  cat <<EOF
Usage: bash scripts/deploy-e035-notebook.sh <commande>

Commandes :
  prepare         Reconstruit le notebook, déploie les images et charge le Stage 1 E034 sur le PVC.
  verify-input    Vérifie localement puis sur le PVC le PNG Stage 1 exact d'E034.
  capture-parent  Exécute uniquement le Stage 2 une fois et fige PNG + latent safetensors.
  verify-parent   Vérifie contrat, hashes, latent et provenance du parent immuable.
  run             Lance les deux branches GPU E035 depuis le même latent vérifié.
  all             Exécute prepare, capture-parent si nécessaire, puis run.
  status          Affiche Jobs, pods, parent et verdict.
  logs            Suit les journaux du Job de comparaison E035.
  logs-capture    Suit les journaux du Job de capture parent.
  download        Télécharge parent, audit et résultats dans ./artifacts/e035-download/.
  restore         Supprime les Jobs et remet l'API à une réplique.
  help            Affiche cette aide.

Variables principales :
  PROOFTAG_E035_PARENT_DIR          défaut: $parent_dir
  PROOFTAG_E035_CAPTURE_AUDIT_DIR   défaut: $capture_audit_dir
  PROOFTAG_E035_RESULTS_DIR         défaut: $results_dir
  PROOFTAG_E035_PARENT_COMMIT       SHA parent attendu facultatif
  PROOFTAG_E035_STAGE1_ASSET        défaut: $local_stage1_asset
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Commande absente: $1" >&2; exit 1; }
}

require_repo() {
  local required=(
    "notebooks/${notebook}"
    "scripts/build_e035_srmpgd_loss_fidelity_gate_notebook.py"
    "deploy/k8s/e035-loss-fidelity-job.yaml"
    "deploy/k8s/e035-parent-capture-job.yaml"
    "docs/e035-assets/e034-observed-stage1.png"
    "prooftag_qr/e035_losses.py"
    "prooftag_qr/e035_loss_fidelity.py"
    "prooftag_qr/e035_parent_artifact.py"
    "prooftag_qr/e035_parent_capture.py"
  )
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "Fichier E035 absent: $path" >&2; exit 1; }
  done
}

require_clean_repo() {
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Le dépôt contient des modifications non commitées." >&2
    echo "Commit/push, puis pull sur le serveur avant le déploiement E035." >&2
    exit 1
  fi
}

api_image() {
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].image}'
}

api_replicas() {
  "$kubectl_bin" get deployment "$api_deployment" -n "$namespace" \
    -o jsonpath='{.spec.replicas}'
}

api_pod() {
  "$kubectl_bin" get pods -n "$namespace" -l app=prooftag-qr \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

ensure_api_running() {
  local replicas
  replicas="$(api_replicas)"
  if [[ "${replicas:-0}" -lt 1 ]]; then
    "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  fi
  "$kubectl_bin" rollout status deployment/$api_deployment -n "$namespace" --timeout=1200s
}

verify_local_stage1() {
  [[ -f "$local_stage1_asset" ]] || {
    echo "Stage 1 E034 absent: $local_stage1_asset" >&2
    exit 1
  }
  python - "$local_stage1_asset" "$stage1_file_sha256" "$stage1_image_sha256" <<'PY'
from pathlib import Path
from PIL import Image
import hashlib
import numpy as np
import sys

path = Path(sys.argv[1])
expected_file = sys.argv[2]
expected_image = sys.argv[3]
file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
with Image.open(path) as opened:
    image = opened.convert("RGB")
digest = hashlib.sha256()
digest.update(f"RGB:{image.width}x{image.height}:".encode())
digest.update(np.asarray(image, dtype=np.uint8).tobytes())
image_hash = digest.hexdigest()
assert file_hash == expected_file, (file_hash, expected_file)
assert image_hash == expected_image, (image_hash, expected_image)
assert image.size == (736, 736), image.size
print(f"Stage 1 local vérifié: file={file_hash} pixels={image_hash}")
PY
}

upload_stage1_to_pvc() {
  ensure_api_running
  verify_local_stage1
  local pod remote_dir
  pod="$(api_pod)"
  [[ -n "$pod" ]] || { echo "Pod API introuvable" >&2; exit 1; }
  remote_dir="$(dirname "$remote_stage1_asset")"
  "$kubectl_bin" exec -i -n "$namespace" "$pod" -c api -- mkdir -p "$remote_dir"
  "$kubectl_bin" cp "$local_stage1_asset" \
    "$namespace/$pod:$remote_stage1_asset" -c api
  verify_remote_stage1
}

verify_remote_stage1() {
  ensure_api_running
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    python - "$remote_stage1_asset" "$stage1_file_sha256" "$stage1_image_sha256" <<'PY'
from pathlib import Path
from PIL import Image
import hashlib
import numpy as np
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Stage 1 absent du PVC: {path}")
file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
with Image.open(path) as opened:
    image = opened.convert("RGB")
digest = hashlib.sha256()
digest.update(f"RGB:{image.width}x{image.height}:".encode())
digest.update(np.asarray(image, dtype=np.uint8).tobytes())
image_hash = digest.hexdigest()
assert file_hash == sys.argv[2], (file_hash, sys.argv[2])
assert image_hash == sys.argv[3], (image_hash, sys.argv[3])
assert image.size == (736, 736), image.size
print(f"Stage 1 PVC vérifié: file={file_hash} pixels={image_hash}")
PY
}

parent_exists_in_api() {
  ensure_api_running
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    sh -c "test -f '$parent_dir/parent-stage2-metadata.json'"
}

verify_parent_in_api() {
  ensure_api_running
  local metadata_file contract
  metadata_file="$(mktemp)"
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    python -m prooftag_qr.e035_parent_artifact "$parent_dir" >"$metadata_file"
  verified_parent_commit_value="$(python - "$metadata_file" "$expected_parent_commit" <<'PY'
import json
from pathlib import Path
import sys

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
explicit = sys.argv[2]
source = metadata["source"]
if explicit and source["source_commit"] != explicit:
    raise SystemExit(
        f"commit parent inattendu: {source['source_commit']} != {explicit}"
    )
assert source["qr_version"] == 3
assert source["qr_mask_pattern"] == 4
assert source["qr_module_size"] == 20
assert source["qr_padding_px"] == 78
assert source["diffqrcoder_revision"] == "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"
method = source.get("source_method_id")
assert source.get("stage1_image_sha256") == (
    "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
)
assert source.get("stage1_file_sha256") == (
    "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71"
)
assert source.get("generation", {}).get("stage1_regenerated") is False
if method == "e033_public_demo_srpg_from_fixed_e034_stage1":
    assert source.get("parent_origin") == "stage2_replayed_from_exact_e034_stage1"
elif method == "e033_public_demo_srpg_exact_e034_export":
    assert source.get("parent_origin") == "exact_e034_stage2_export"
else:
    raise SystemExit(f"source_method_id parent non autorisé: {method!r}")
print(source["source_commit"])
PY
)"
  contract="$(python - "$metadata_file" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["contract_sha256"])
PY
)"
  rm -f "$metadata_file"
  echo "Parent E035 vérifié: contrat=$contract source_commit=$verified_parent_commit_value"
}

prepare() {
  require_command "$kubectl_bin"
  require_repo
  require_clean_repo
  python scripts/build_e035_srmpgd_loss_fidelity_gate_notebook.py
  git diff --exit-code -- notebooks/$notebook

  local git_sha current_notebook_replicas
  git_sha="$(git rev-parse HEAD)"
  echo "===== E035 PREPARE — commit $git_sha ====="
  current_notebook_replicas="$($kubectl_bin get deployment "$notebook_deployment" \
    -n "$namespace" -o jsonpath='{.spec.replicas}' 2>/dev/null || printf '0')"
  if [[ "${current_notebook_replicas:-0}" -gt 0 ]]; then
    bash scripts/notebook-server.sh stop
  fi
  bash scripts/notebook-server.sh deploy-prepare "$notebook"
  bash scripts/deploy-app-image.sh
  [[ "$(git rev-parse HEAD)" == "$git_sha" ]] || {
    echo "Le commit a changé pendant la construction API." >&2; exit 1;
  }
  bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"
  [[ "$(git rev-parse HEAD)" == "$git_sha" ]] || {
    echo "Le commit a changé pendant la construction notebook." >&2; exit 1;
  }
  bash scripts/notebook-server.sh deploy-start "$notebook"
  "$kubectl_bin" rollout status deployment/$api_deployment -n "$namespace" --timeout=1200s
  "$kubectl_bin" rollout status deployment/$notebook_deployment -n "$namespace" --timeout=1200s
  upload_stage1_to_pvc

  if parent_exists_in_api; then
    verify_parent_in_api
  else
    echo "Aucun parent immuable dans $parent_dir."
    echo "Étape suivante: bash scripts/deploy-e035-notebook.sh capture-parent"
  fi
  echo "===== E035 PRÉPARÉ ====="
  echo "Notebook: $notebook"
}

render_loss_job() {
  local image="$1" parent_commit="$2"
  sed \
    -e "s|__NAMESPACE__|$namespace|g" \
    -e "s|__JOB_NAME__|$job_name|g" \
    -e "s|__IMAGE__|$image|g" \
    -e "s|__PARENT_DIR__|$parent_dir|g" \
    -e "s|__RESULTS_DIR__|$results_dir|g" \
    -e "s|__EXPECTED_PARENT_COMMIT__|$parent_commit|g" \
    deploy/k8s/e035-loss-fidelity-job.yaml
}

render_capture_job() {
  local image="$1" source_commit="$2"
  sed \
    -e "s|__NAMESPACE__|$namespace|g" \
    -e "s|__JOB_NAME__|$capture_job_name|g" \
    -e "s|__IMAGE__|$image|g" \
    -e "s|__STAGE1_IMAGE__|$remote_stage1_asset|g" \
    -e "s|__STAGE1_IMAGE_SHA256__|$stage1_image_sha256|g" \
    -e "s|__STAGE1_FILE_SHA256__|$stage1_file_sha256|g" \
    -e "s|__PARENT_DIR__|$parent_dir|g" \
    -e "s|__CAPTURE_AUDIT_DIR__|$capture_audit_dir|g" \
    -e "s|__SOURCE_COMMIT__|$source_commit|g" \
    deploy/k8s/e035-parent-capture-job.yaml
}

run_exclusive_gpu_job() {
  local selected_job="$1" rendered_file="$2" timeout="$3"
  local previous_replicas restored
  previous_replicas="$(api_replicas)"
  restored=0

  cleanup_gpu_job() {
    local code="$?"
    rm -f "$rendered_file"
    if [[ "$restored" -eq 0 ]]; then
      "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" \
        --replicas="${previous_replicas:-1}" >/dev/null || true
      if [[ "${previous_replicas:-1}" -gt 0 ]]; then
        "$kubectl_bin" rollout status deployment/$api_deployment -n "$namespace" \
          --timeout=1200s || true
      fi
    fi
    return "$code"
  }
  trap cleanup_gpu_job RETURN

  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=0
  "$kubectl_bin" wait --for=delete pod -l app=prooftag-qr -n "$namespace" \
    --timeout=600s || true
  "$kubectl_bin" delete job "$selected_job" -n "$namespace" --ignore-not-found
  "$kubectl_bin" apply -f "$rendered_file"
  if ! "$kubectl_bin" wait --for=condition=complete job/$selected_job \
    -n "$namespace" --timeout="$timeout"; then
    "$kubectl_bin" logs -n "$namespace" job/$selected_job \
      --all-containers=true --tail=-1 || true
    "$kubectl_bin" describe job "$selected_job" -n "$namespace" || true
    return 1
  fi
  "$kubectl_bin" logs -n "$namespace" job/$selected_job \
    --all-containers=true --tail=-1

  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" \
    --replicas="${previous_replicas:-1}"
  if [[ "${previous_replicas:-1}" -gt 0 ]]; then
    "$kubectl_bin" rollout status deployment/$api_deployment -n "$namespace" --timeout=1200s
  fi
  restored=1
  rm -f "$rendered_file"
  trap - RETURN
}

capture_parent() {
  require_repo
  require_clean_repo
  upload_stage1_to_pvc
  if parent_exists_in_api; then
    echo "Un parent existe déjà dans $parent_dir; aucun écrasement automatique." >&2
    verify_parent_in_api
    return 1
  fi
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    python - "$parent_dir" "$capture_audit_dir" <<'PY'
from pathlib import Path
import sys
for raw in sys.argv[1:]:
    path = Path(raw)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"répertoire non vide: {path}")
print("Répertoires de capture disponibles")
PY

  local image git_sha job_file
  image="$(api_image)"
  git_sha="$(git rev-parse HEAD)"
  job_file="$(mktemp)"
  render_capture_job "$image" "$git_sha" >"$job_file"
  echo "Capture Stage 2 depuis le Stage 1 E034 vérifié avec $image, commit $git_sha"
  run_exclusive_gpu_job "$capture_job_name" "$job_file" \
    "${PROOFTAG_E035_CAPTURE_TIMEOUT:-10800s}"
  verify_parent_in_api
  echo "===== PARENT E035 FIGÉ ====="
  echo "Parent: $parent_dir"
  echo "Audit de capture: $capture_audit_dir"
}

run_job() {
  require_repo
  require_clean_repo
  verify_parent_in_api

  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    python - "$results_dir" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
if path.exists() and any(path.iterdir()):
    raise SystemExit(f"Répertoire E035 non vide: {path}")
print("Sortie E035 disponible:", path)
PY

  local image job_file
  image="$(api_image)"
  job_file="$(mktemp)"
  render_loss_job "$image" "$verified_parent_commit_value" >"$job_file"
  echo "Lancement du Job E035 apparié avec $image"
  run_exclusive_gpu_job "$job_name" "$job_file" \
    "${PROOFTAG_E035_JOB_TIMEOUT:-10800s}"

  ensure_api_running
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    python - "$results_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
verdict = json.loads((root / "verdict.json").read_text(encoding="utf-8"))
assert verdict["production_ready"] is False
assert verdict["advisor_training_authorized"] is False
print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
PY
  echo "===== E035 TERMINÉ ====="
  echo "Résultats: $results_dir"
  echo "Dans Jupyter: Run > Run All Cells sur $notebook"
}

run_all() {
  prepare
  if parent_exists_in_api; then
    verify_parent_in_api
  else
    capture_parent
  fi
  run_job
}

status_job() {
  "$kubectl_bin" get job "$capture_job_name" "$job_name" -n "$namespace" -o wide \
    2>/dev/null || true
  "$kubectl_bin" get pods -n "$namespace" \
    -l 'app in (prooftag-qr-e035-parent-capture,prooftag-qr-e035)' -o wide \
    2>/dev/null || true
  ensure_api_running
  verify_remote_stage1 || true
  if parent_exists_in_api; then
    verify_parent_in_api || true
  else
    echo "Parent absent: $parent_dir"
  fi
  "$kubectl_bin" exec -i -n "$namespace" deployment/$api_deployment -c api -- \
    sh -c "test -f '$results_dir/verdict.json' && cat '$results_dir/verdict.json' || echo 'verdict.json absent'"
}

logs_job() {
  "$kubectl_bin" logs -n "$namespace" job/$job_name --all-containers=true -f
}

logs_capture() {
  "$kubectl_bin" logs -n "$namespace" job/$capture_job_name --all-containers=true -f
}

download_artifacts() {
  ensure_api_running
  local destination pod archive_remote archive_local
  destination="${PROOFTAG_E035_DOWNLOAD_DIR:-artifacts/e035-download}"
  mkdir -p "$destination"
  pod="$(api_pod)"
  archive_remote="/tmp/e035-download.tar.gz"
  archive_local="$destination/e035-download.tar.gz"
  "$kubectl_bin" exec -i -n "$namespace" "$pod" -c api -- \
    tar -czf "$archive_remote" \
      "$parent_dir" "$capture_audit_dir" "$results_dir"
  "$kubectl_bin" cp "$namespace/$pod:$archive_remote" "$archive_local" -c api
  "$kubectl_bin" exec -i -n "$namespace" "$pod" -c api -- rm -f "$archive_remote"
  sha256sum "$archive_local" | tee "$archive_local.sha256"
  echo "Artefacts téléchargés: $archive_local"
}

restore() {
  "$kubectl_bin" delete job "$job_name" "$capture_job_name" -n "$namespace" \
    --ignore-not-found
  "$kubectl_bin" scale deployment "$api_deployment" -n "$namespace" --replicas=1
  "$kubectl_bin" rollout status deployment/$api_deployment -n "$namespace" --timeout=1200s
  echo "API restaurée à une réplique."
}

case "$command_name" in
  prepare) prepare ;;
  verify-input) verify_local_stage1; verify_remote_stage1 ;;
  capture-parent) capture_parent ;;
  verify-parent) verify_parent_in_api ;;
  run) run_job ;;
  all) run_all ;;
  status) status_job ;;
  logs) logs_job ;;
  logs-capture) logs_capture ;;
  download) download_artifacts ;;
  restore) restore ;;
  help|-h|--help) usage ;;
  *) echo "Commande inconnue: $command_name" >&2; usage >&2; exit 2 ;;
esac
