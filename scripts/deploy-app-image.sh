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
  "from prooftag_qr.lab import laboratory_profiles; p = next(p for p in laboratory_profiles() if p['id'] == 'srpg_full_restart_srmpgd'); assert p['enabled'] is True and p['output_variant'] == 'srmpgd' and p['tools']['srmpgd_enabled'] is True; print(p['id'], p['output_variant'], p['tools']['srmpgd_enabled'])"
kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from pathlib import Path; import prooftag_qr; path = Path(prooftag_qr.__file__).with_name('lab_static') / 'index.html'; assert '20260729-quiet-zone-1' in path.read_text(encoding='utf-8'); print('Assets Web quiet-zone confirmés')"

echo "Image déployée et vérifiée : $image"
echo "Pod : $pod"
