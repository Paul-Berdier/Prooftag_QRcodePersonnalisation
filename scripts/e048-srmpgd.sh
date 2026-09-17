#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/nightops${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -B -m qrnight.e048_host "$@"
