#!/usr/bin/env bash
set -euo pipefail

namespace="${PROOFTAG_QR_NAMESPACE:-qr-core}"
deployment="${PROOFTAG_QR_DEPLOYMENT:-prooftag-qr}"
container="${PROOFTAG_QR_CONTAINER:-api}"
init_container="${PROOFTAG_QR_INIT_CONTAINER:-database-migrations}"
image_repository="${PROOFTAG_QR_IMAGE:-prooftag-qr}"

if [[ ! -f Dockerfile || ! -d prooftag_qr ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Le dépôt contient des modifications non commitées." >&2
  echo "Commit/push/pull avant de construire une image traçable." >&2
  exit 1
fi

ready_pod_for_image() {
  local expected_image="$1"
  local deadline=$((SECONDS + 180))
  local pod running_image ready deleting
  while ((SECONDS < deadline)); do
    while IFS= read -r pod; do
      [[ -n "$pod" ]] || continue
      deleting="$(
        kubectl -n "$namespace" get pod "$pod" \
          -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null || true
      )"
      [[ -z "$deleting" ]] || continue
      running_image="$(
        kubectl -n "$namespace" get pod "$pod" \
          -o "jsonpath={.spec.containers[?(@.name=='${container}')].image}" \
          2>/dev/null || true
      )"
      ready="$(
        kubectl -n "$namespace" get pod "$pod" \
          -o 'jsonpath={.status.conditions[?(@.type=="Ready")].status}' \
          2>/dev/null || true
      )"
      if [[ "$running_image" == "$expected_image" && "$ready" == "True" ]]; then
        printf '%s\n' "$pod"
        return 0
      fi
    done < <(
      kubectl -n "$namespace" get pods \
        -l app=prooftag-qr \
        --field-selector=status.phase=Running \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
    )
    sleep 2
  done
  echo "Aucun pod prêt n'exécute l'image attendue $expected_image." >&2
  kubectl -n "$namespace" get pods -l app=prooftag-qr \
    -o custom-columns='POD:.metadata.name,DELETING:.metadata.deletionTimestamp,READY:.status.conditions[?(@.type=="Ready")].status,IMAGE:.spec.containers[?(@.name=="api")].image' \
    >&2
  return 1
}

git_sha="$(git rev-parse HEAD)"
git_tag="$(git rev-parse --short=12 HEAD)"
image="${image_repository}:${git_tag}"

echo "Construction de $image depuis $git_sha"
docker build \
  --label "org.opencontainers.image.revision=${git_sha}" \
  -t "$image" .

echo "Image Docker construite : $image"
docker image inspect "$image" \
  --format 'id={{.Id}} révision={{index .Config.Labels "org.opencontainers.image.revision"}}'

docker save "$image" | sudo k3s ctr images import -

kubectl apply -f deploy/k8s/app-config.yaml

image_patch="$(
  printf \
    '{"spec":{"template":{"spec":{"containers":[{"name":"%s","image":"%s"}],"initContainers":[{"name":"%s","image":"%s"}]}}}}' \
    "$container" "$image" "$init_container" "$image"
)"
kubectl -n "$namespace" patch "deployment/${deployment}" \
  --type=strategic \
  -p "$image_patch"
kubectl -n "$namespace" annotate "deployment/${deployment}" \
  "prooftag.io/git-revision=${git_sha}" --overwrite

deployed_api="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].image}"
)"
deployed_init="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.initContainers[?(@.name=='${init_container}')].image}"
)"
if [[ "$deployed_api" != "$image" || "$deployed_init" != "$image" ]]; then
  echo "Images inattendues : api=$deployed_api init=$deployed_init attendu=$image" >&2
  exit 1
fi

kubectl -n "$namespace" rollout status \
  "deployment/${deployment}" --timeout=1200s

pod="$(ready_pod_for_image "$image")"
running_image="$(
  kubectl -n "$namespace" get pod "$pod" \
    -o "jsonpath={.spec.containers[?(@.name=='${container}')].image}"
)"
echo "Pod courant vérifié : pod=$pod image=$running_image"

kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from prooftag_qr.lab import laboratory_profiles; p = laboratory_profiles(); ids = [x['id'] for x in p]; expected = ['qr_reference', 'diffqrcoder_stage1', 'diffqrcoder_srpg', 'diffqrcoder_paper_srpg', 'diffqrcoder_srmpgd', 'diffqrcoder_srmpgd_robust', 'diffqrcoder_auto', 'diffqrcoder_srpg_s035', 'diffqrcoder_srpg_s050', 'diffqrcoder_srpg_s080', 'diffqrcoder_qart_srpg']; assert ids == expected, (ids, expected); generated = [x for x in p if x['backend'] == 'controlnet']; assert all(x['model'].get('diffqrcoder_upstream_enabled') for x in generated); print([(x['id'], x['output_variant'], x['enabled']) for x in p])"
kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from pathlib import Path; import diffqrcoder, hpsv2, prooftag_qr; from prooftag_qr.config import get_settings; from prooftag_qr.qr import generate_diffqrcoder_qr; from prooftag_qr.validation import QRVerifyDecoder; settings = get_settings(); assert settings.lab_clip_scoring_enabled and settings.lab_hps_scoring_enabled; root = Path(prooftag_qr.__file__).with_name('lab_static'); html = (root / 'index.html').read_text(encoding='utf-8'); js = (root / 'app.js').read_text(encoding='utf-8'); assert '20260805-e025-quality-scores-1' in html; assert all(label in js for label in ('Score QR-Verify', 'CLIP-AES', 'CLIPScore', 'HPS v2.1')); payload = 'https://ptag.io/t/deploy-check'; decoder = QRVerifyDecoder(); attempts = decoder.decode_presets(generate_diffqrcoder_qr(payload).image); decoder.close(); assert len(attempts) == 37 and all(item['text'] == payload for item in attempts); print('DiffQRCoder', Path(diffqrcoder.__file__).resolve()); print('E025 confirmé: QR-Verify 37/37 + CLIP/HPS activés')"

echo "Image déployée et vérifiée : $image"
echo "Pod : $pod"
