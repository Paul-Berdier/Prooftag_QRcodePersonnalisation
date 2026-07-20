#!/usr/bin/env bash
set -euo pipefail

namespace="qr-core"
secret_name="prooftag-qr-database"

kubectl apply -f deploy/k8s/namespace.yaml

if kubectl get secret "$secret_name" -n "$namespace" >/dev/null 2>&1; then
  echo "Secret $namespace/$secret_name already exists; keeping the current password."
  exit 0
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate the database password." >&2
  exit 1
fi

database_password="$(openssl rand -base64 36 | tr -d '\n')"
kubectl create secret generic "$secret_name" \
  -n "$namespace" \
  --from-literal="password=$database_password"
unset database_password

echo "Created $namespace/$secret_name. The password was not printed."
