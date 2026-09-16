from __future__ import annotations
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
LABEL = "prooftag.io/night-run"


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: str | Path, obj: Any) -> None:
    """Atomic replacement inside our own run only; never an upstream output."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


def git_blob(path: str | Path) -> str:
    b = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp(text: str) -> float:
    d = datetime.fromisoformat(text)
    if d.tzinfo is None:
        # Ambiguous/nonexistent DST wall times are rejected, not silently chosen.
        a, b = d.replace(tzinfo=PARIS, fold=0), d.replace(tzinfo=PARIS, fold=1)
        if a.utcoffset() != b.utcoffset():
            raise ValueError("Heure ambiguë ou inexistante : préciser +01:00 ou +02:00")
        d = a
    return d.timestamp()


def schedule(start: str, ready: str, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    a, z = stamp(start), stamp(ready)
    stop = z - 3600  # GPU + CPU all stopped; full hour reserved to recover production
    if a < now + 120:
        raise ValueError("Le départ doit être au moins deux minutes dans le futur.")
    if not 4 * 3600 <= stop - a <= 16 * 3600:
        raise ValueError("Fenêtre calcul attendue : entre 4 et 16 h, hors heure de restauration.")
    span = stop - a
    return {"start_epoch": a, "score_until": a + span * 0.46,
            "train_until": a + span * 0.56, "generate_until": stop - 1200,
            "stop_epoch": stop, "ready_epoch": z,
            "start_paris": datetime.fromtimestamp(a, PARIS).isoformat(),
            "stop_paris": datetime.fromtimestamp(stop, PARIS).isoformat(),
            "ready_paris": datetime.fromtimestamp(z, PARIS).isoformat()}


def gpu_requested(p: dict) -> bool:
    return any(float(c.get("resources", {}).get(t, {}).get("nvidia.com/gpu", 0)) > 0
               for c in p.get("spec", {}).get("containers", []) +
                        p.get("spec", {}).get("initContainers", [])
               for t in ("requests", "limits"))


def alive(p: dict) -> bool:
    return p.get("status", {}).get("phase") not in ("Succeeded", "Failed")


def eligibility(row: dict, minimum: int = 34) -> bool:
    return (row.get("raw") is True and row.get("visual_guard_pass") is True and
            row.get("wechat_original_exact") is True and
            int(row.get("wechat_exact_presets") or 0) >= minimum and
            not row.get("uniform_quiet_zone_replacement", False))


def normalized_prompt(x: Any) -> str:
    return " ".join(str(x).casefold().split())


def within(path: str | Path, root: str | Path) -> Path:
    p, r = Path(path).resolve(), Path(root).resolve()
    if p != r and r not in p.parents:
        raise ValueError(f"Chemin hors racine : {p}")
    return p


def check_id(x: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,45}", x):
        raise ValueError("Identifiant de run invalide")
    return x
