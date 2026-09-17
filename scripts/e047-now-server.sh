#!/usr/bin/env bash
# Execute on pcIA only. A subshell failure never closes the interactive SSH session.
(
set -Eeuo pipefail
cd /home/paul/apps/Prooftag_QRcodePersonnalisation

if [[ "$(git branch --show-current)" != main ]]; then
  echo "ARRET : le clone serveur n'est pas sur main." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git status --short
  echo "ARRET : modifications locales suivies. Ne pas utiliser reset --hard." >&2
  exit 1
fi

# HTTPS avoids the outgoing SSH connection to GitHub that was blocked on pcIA.
GIT_TERMINAL_PROMPT=0 timeout --kill-after=5s 180s \
  git -c http.version=HTTP/1.1 fetch --progress origin main
GIT_TERMINAL_PROMPT=0 timeout --kill-after=5s 180s \
  git -c http.version=HTTP/1.1 pull --ff-only origin main

test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
python3 -B - <<'PY'
import ast
import hashlib
import json
from pathlib import Path
root = Path.cwd()
m = json.loads((root / "MANIFEST_NUIT_E047.json").read_text(encoding="utf-8"))
if m.get("release_version") != "1.1.0-immediate":
    raise SystemExit("Le correctif start-now n'est pas dans le commit recupere")
for name, expected in m["files"].items():
    p = root / name
    if p.is_symlink() or not p.resolve().is_relative_to(root):
        raise SystemExit(f"Chemin refuse : {name}")
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Fichier different du ZIP valide : {name}")
    if p.suffix == ".py":
        ast.parse(p.read_text(encoding="utf-8"), filename=name)
print("VERSION ET FICHIERS DU CORRECTIF : OK")
PY

PYTHONPATH="$PWD/nightops" python3 -B -m unittest discover \
  -s nightops/tests -p test_immediate.py -v

# No prepare + schedule and no future start time: the service starts after validation.
sudo bash scripts/qr-night.sh start-now \
  --reuse-prepared-run qrn-20260916-154541 \
  --max-hours 12 \
  --confirm COUPURE-VLLM-AUTORISEE
)
