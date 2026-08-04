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

pod="$(
  kubectl -n "$namespace" get pod \
    -l app=prooftag-qr \
    -o jsonpath='{.items[0].metadata.name}'
)"
running_image="$(
  kubectl -n "$namespace" get pod "$pod" \
    -o "jsonpath={.spec.containers[?(@.name=='${container}')].image}"
)"
if [[ "$running_image" != "$image" ]]; then
  echo "Le pod exécute $running_image au lieu de $image." >&2
  exit 1
fi

kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from prooftag_qr.lab import laboratory_profiles; p = laboratory_profiles(); ids = [x['id'] for x in p]; expected = ['qr_reference', 'diffqrcoder_stage1', 'diffqrcoder_srpg', 'diffqrcoder_srmpgd', 'diffqrcoder_srmpgd_robust', 'diffqrcoder_auto', 'diffqrcoder_srpg_s035', 'diffqrcoder_srpg_s050', 'diffqrcoder_srpg_s080', 'diffqrcoder_qart_srpg']; assert ids == expected, (ids, expected); generated = [x for x in p if x['backend'] == 'controlnet']; assert all(x['model'].get('diffqrcoder_upstream_enabled') for x in generated); print([(x['id'], x['output_variant'], x['enabled']) for x in p])"
kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from pathlib import Path; import diffqrcoder, prooftag_qr; from prooftag_qr.lab import laboratory_profiles; path = Path(prooftag_qr.__file__).with_name('lab_static') / 'index.html'; text = path.read_text(encoding='utf-8'); assert '20260804-e020-trace-robust-1' in text; profiles = laboratory_profiles(); srpg = next(x for x in profiles if x['id'] == 'diffqrcoder_srpg'); srmpgd = next(x for x in profiles if x['id'] == 'diffqrcoder_srmpgd'); robust = next(x for x in profiles if x['id'] == 'diffqrcoder_srmpgd_robust'); auto = next(x for x in profiles if x['id'] == 'diffqrcoder_auto'); qart = next(x for x in profiles if x['id'] == 'diffqrcoder_qart_srpg'); assert srpg['tools']['settings']['diffqrcoder_stage2_initialization'] == 'paper_stage1_noise'; assert srpg['tools']['settings']['diffqrcoder_stage2_target_mode'] == 'binary_exact'; assert srpg['tools']['settings']['diffqrcoder_stage2_strength'] == 0.65; assert auto['output_variant'] == 'auto' and auto['enabled'] is False; assert qart['enabled'] is False; assert srmpgd['enabled'] is True and robust['enabled'] is True; s = srmpgd['tools']['settings']; r = robust['tools']['settings']; assert (s['srmpgd_max_iterations'], s['srmpgd_step_size'], s['srmpgd_lpips_weight']) == (4, 100.0, 0.10); assert (s['srmpgd_max_step_rms'], s['srmpgd_max_total_delta_rms']) == (0.02, 0.06); assert (r['srmpgd_robust_blur_weight'], r['srmpgd_robust_downscale_weight'], r['srmpgd_robust_contrast_weight']) == (1.0, 1.0, 1.0); print('DiffQRCoder importé depuis', Path(diffqrcoder.__file__).resolve()); print('E020 trace et loss robuste appariée confirmées')"

echo "Image déployée et vérifiée : $image"
echo "Pod : $pod"
