#!/usr/bin/env bash
set -euo pipefail

namespace="${PROOFTAG_NOTEBOOK_NAMESPACE:-qr-core}"
deployment="${PROOFTAG_NOTEBOOK_DEPLOYMENT:-prooftag-qr-notebook}"
container="${PROOFTAG_NOTEBOOK_CONTAINER:-notebook}"
image_repository="${PROOFTAG_NOTEBOOK_IMAGE:-prooftag-qr-notebook}"
expected_notebook="${1:-notebooks/21_e026_prompt_parameter_advisor.ipynb}"

if [[ ! -f Dockerfile.notebook || ! -d notebooks ]]; then
  echo "Lancer ce script depuis la racine du dépôt." >&2
  exit 1
fi
if [[ ! -f "$expected_notebook" ]]; then
  echo "Notebook attendu absent du dépôt : $expected_notebook" >&2
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
image_notebook="/workspace/${expected_notebook}"

echo "Construction de $image depuis $git_sha"
docker build -f Dockerfile.notebook \
  --build-arg "EXPECTED_NOTEBOOK=${expected_notebook}" \
  --label "org.opencontainers.image.revision=${git_sha}" \
  -t "$image" .

image_revision="$(
  docker image inspect "$image" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
if [[ "$image_revision" != "$git_sha" ]]; then
  echo "Révision de l'image inattendue : $image_revision != $git_sha" >&2
  exit 1
fi
echo "Notebook vérifié pendant le build : $image_notebook"

docker save "$image" | sudo k3s ctr images import -

kubectl -n "$namespace" set image "deployment/${deployment}" \
  "${container}=${image}"
kubectl -n "$namespace" annotate "deployment/${deployment}" \
  "prooftag.io/git-revision=${git_sha}" --overwrite

deployed_image="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].image}"
)"
if [[ "$deployed_image" != "$image" ]]; then
  echo "Image du Deployment inattendue : $deployed_image != $image" >&2
  exit 1
fi

replicas="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o jsonpath='{.spec.replicas}'
)"
if [[ "${replicas:-0}" -gt 0 ]]; then
  kubectl -n "$namespace" rollout status \
    "deployment/${deployment}" --timeout=1200s
  pod="$(
    kubectl -n "$namespace" get pod \
      -l app=prooftag-qr-notebook \
      -o jsonpath='{.items[0].metadata.name}'
  )"
  kubectl -n "$namespace" exec "$pod" -- test -f "$image_notebook"
  echo "Notebook vérifié dans le pod : $pod:$image_notebook"
fi

echo "Image déployée : $image"
echo "Ouvrir depuis le PC :"
if [[ "${replicas:-0}" -gt 0 ]]; then
  echo ".\\scripts\\notebook-remote.ps1 -Reset -Notebook $(basename "$expected_notebook")"
else
  echo ".\\scripts\\notebook-remote.ps1 -Notebook $(basename "$expected_notebook")"
fi
