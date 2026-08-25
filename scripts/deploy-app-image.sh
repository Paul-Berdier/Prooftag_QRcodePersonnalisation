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

image_revision="$(
  docker image inspect "$image" \
    --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
)"
image_digest="$(
  docker image inspect "$image" --format '{{.Id}}'
)"
if [[ "$image_revision" != "$git_sha" ]]; then
  echo "Révision de l'image inattendue : $image_revision != $git_sha" >&2
  exit 1
fi
if [[ ! "$image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Digest de l'image inattendu : $image_digest" >&2
  exit 1
fi
echo "Image Docker construite : $image ($image_digest, révision $image_revision)"

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
kubectl -n "$namespace" set env "deployment/${deployment}" \
  --containers="$container" \
  "PROOFTAG_GIT_COMMIT=${git_sha}" \
  "PROOFTAG_RUNTIME_IMAGE=${image}" \
  "PROOFTAG_RUNTIME_IMAGE_DIGEST=${image_digest}"

deployed_api="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].image}"
)"
deployed_init="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.initContainers[?(@.name=='${init_container}')].image}"
)"
deployed_commit="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].env[?(@.name=='PROOFTAG_GIT_COMMIT')].value}"
)"
deployed_runtime_image="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].env[?(@.name=='PROOFTAG_RUNTIME_IMAGE')].value}"
)"
deployed_runtime_digest="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o "jsonpath={.spec.template.spec.containers[?(@.name=='${container}')].env[?(@.name=='PROOFTAG_RUNTIME_IMAGE_DIGEST')].value}"
)"
deployed_annotation="$(
  kubectl -n "$namespace" get "deployment/${deployment}" \
    -o 'jsonpath={.metadata.annotations.prooftag\.io/git-revision}'
)"
if [[ "$deployed_api" != "$image" || "$deployed_init" != "$image" ]]; then
  echo "Images inattendues : api=$deployed_api init=$deployed_init attendu=$image" >&2
  exit 1
fi
if [[ "$deployed_commit" != "$git_sha" || "$deployed_annotation" != "$git_sha" ]]; then
  echo "Commit du Deployment inattendu : env=$deployed_commit annotation=$deployed_annotation attendu=$git_sha" >&2
  exit 1
fi
if [[ "$deployed_runtime_image" != "$image" ]]; then
  echo "Image runtime du Deployment inattendue : $deployed_runtime_image != $image" >&2
  exit 1
fi
if [[ "$deployed_runtime_digest" != "$image_digest" ]]; then
  echo "Digest runtime du Deployment inattendu : $deployed_runtime_digest != $image_digest" >&2
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
  "import os; expected = {'PROOFTAG_GIT_COMMIT': '$git_sha', 'PROOFTAG_RUNTIME_IMAGE': '$image', 'PROOFTAG_RUNTIME_IMAGE_DIGEST': '$image_digest'}; actual = {key: os.environ.get(key) for key in expected}; assert actual == expected, (actual, expected); print('Identité runtime API vérifiée:', actual)"

kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from prooftag_qr.lab import laboratory_profiles; p = laboratory_profiles(); ids = [x['id'] for x in p]; required = {'qr_reference', 'diffqrcoder_stage1', 'diffqrcoder_srpg', 'diffqrcoder_srmpgd'}; assert len(ids) == len(set(ids)), ids; assert required <= set(ids), (sorted(required - set(ids)), ids); generated = [x for x in p if x['backend'] == 'controlnet']; assert generated and all(x['model'].get('diffqrcoder_upstream_enabled') for x in generated); print([(x['id'], x['output_variant'], x['enabled']) for x in p])"
kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from pathlib import Path; import diffqrcoder, hpsv2, prooftag_qr; from prooftag_qr.config import get_settings; from prooftag_qr.qr import generate_diffqrcoder_qr; from prooftag_qr.validation import QRVerifyDecoder; settings = get_settings(); assert settings.lab_clip_scoring_enabled and settings.lab_hps_scoring_enabled; root = Path(prooftag_qr.__file__).with_name('lab_static'); html = (root / 'index.html').read_text(encoding='utf-8'); js = (root / 'app.js').read_text(encoding='utf-8'); assert '/lab-assets/app.js?v=' in html; assert all(label in js for label in ('Score QR-Verify', 'CLIP-AES', 'CLIPScore', 'HPS v2.1')); payload = 'https://ptag.io/t/deploy-check'; decoder = QRVerifyDecoder(); attempts = decoder.decode_presets(generate_diffqrcoder_qr(payload).image); decoder.close(); assert len(attempts) == 37 and all(item['text'] == payload for item in attempts); print('DiffQRCoder', Path(diffqrcoder.__file__).resolve()); print('QR-Verify 37/37 + CLIP/HPS activés')"
kubectl -n "$namespace" exec "$pod" -c "$container" -- \
  python -c \
  "from prooftag_qr.config import get_settings; s=get_settings(); expected={'base_model_revision':'f914b3679760c1c3baea6bb1815867bf1c9c92a4','base_model_config_revision':'451f4fe16113bff5a5d2269ed5ad43b0592e9a14','controlnet_model_revision':'560fb7b15d0badb409f8cd578a2bfe63bd4b8046','diffqrcoder_revision':'e24ea73ee2e13c7e6e87cb422e8b11784e70ae00'}; actual={key:getattr(s,key) for key in expected}; assert actual==expected,(actual,expected); assert '/resolve/f914b3679760c1c3baea6bb1815867bf1c9c92a4/' in s.base_model_id; print('Révisions des modèles vérifiées:', actual)"

echo "Image déployée et vérifiée : $image ($image_digest)"
echo "Pod : $pod"
