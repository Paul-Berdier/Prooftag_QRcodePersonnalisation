"""Immediate launch with a fresh run, reusing only the validated worker image.

No future start timer. No change to the old run, historical data or live vLLM
container image. This module runs on the host; workers retain their audited code.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import time
from datetime import datetime, timezone

from . import BASE_COMMIT, host
from .common import PARIS, LABEL, digest, read, write, sha, check_id

VERSION = "1.1.0-immediate"
CONFIRMATION = "COUPURE-VLLM-AUTORISEE"
WORKER_FILES = (
    "Dockerfile", "requirements-media.txt", "config.json", "prompts.json",
    "qrnight/__init__.py", "qrnight/common.py", "qrnight/worker.py",
    "qrnight/learning.py", "qrnight/report.py",
)


def immediate_window(now: float, hours: float) -> dict:
    if not math.isfinite(hours) or not 5 <= hours <= 17:
        raise ValueError("--max-hours doit être entre 5 et 17, dernière heure réservée à la restauration")
    end = now + hours * 3600
    stop = end - 3600
    span = stop - now
    return {
        "launch_mode": "immediate", "start_epoch": now,
        "score_until": now + span * 0.46, "train_until": now + span * 0.56,
        "generate_until": stop - 1200, "stop_epoch": stop, "ready_epoch": end,
        "start_paris": datetime.fromtimestamp(now, PARIS).isoformat(),
        "stop_paris": datetime.fromtimestamp(stop, PARIS).isoformat(),
        "ready_paris": datetime.fromtimestamp(end, PARIS).isoformat(),
    }


def verify_release(repo: Path) -> dict:
    """Validate the complete new overlay; no pip, checkout or fetch here."""
    manifest = read(repo / "MANIFEST_NUIT_E047.json")
    if manifest.get("release_version") != VERSION:
        raise RuntimeError("Manifeste du correctif immédiat absent ou mauvais ZIP")
    for name, expected in manifest["files"].items():
        p = repo / name
        if p.is_symlink() or not p.resolve().is_relative_to(repo):
            raise RuntimeError("Chemin de release invalide : " + name)
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError("Fichier différent du ZIP vérifié : " + name)
    git = ["git", "-c", "safe.directory=" + str(repo), "-C", repo]
    head = host.command([*git, "rev-parse", "HEAD"], quiet=True).stdout.strip()
    host.command([*git, "merge-base", "--is-ancestor", BASE_COMMIT, head], quiet=True)
    host.command([*git, "diff", "--quiet", BASE_COMMIT, "--", "prooftag_qr", "data", "qr_verify_bridge"])
    dirty = host.command([*git, "status", "--porcelain", "--untracked-files=no"], quiet=True).stdout.strip()
    if dirty:
        raise RuntimeError("Modifications suivies non commitées : utiliser le flux commit/push puis fetch/merge")
    return {"head": head, "manifest": manifest}


def verify_worker_image(old_cfg: dict, repo: Path) -> dict:
    old_install = Path(old_cfg["install"]).resolve()
    if not old_install.is_relative_to(host.INSTALL_ROOT.resolve()):
        raise RuntimeError("Installation worker d'origine hors de la racine attendue")
    verified = {}
    for name in WORKER_FILES:
        old = old_install / "nightops" / name
        new = repo / "nightops" / name
        if old.is_symlink() or not old.is_file() or sha(old) != sha(new):
            raise RuntimeError("Worker scientifique modifié; réutilisation de l'image refusée : " + name)
        verified[name] = sha(new)
    # CRI inspect reads local metadata and never pulls an image.
    p = host.command(["/usr/local/bin/k3s", "crictl", "inspecti", old_cfg["image"]], quiet=True, timeout=45)
    status = json.loads(p.stdout).get("status", {})
    actual = str(status.get("id", ""))
    if actual.removeprefix("sha256:") != str(old_cfg["image_id"]).removeprefix("sha256:"):
        raise RuntimeError("Digest de l'image worker locale différent du préflight; aucun tag réécrit")
    return {"image": old_cfg["image"], "image_id": old_cfg["image_id"], "worker_files": verified}


def ensure_no_other_run(source: Path) -> None:
    for r in (host.STATE_ROOT / "runs").iterdir():
        if not r.is_dir():
            continue
        if r != source and ((r / "SCHEDULED").exists() or (r / "STARTING.json").exists()):
            if not (r / "RESTORED.json").is_file():
                raise RuntimeError("Autre campagne active ou armée : " + r.name)
    cfg = read(source / "config.json")
    state = host.command(["systemctl", "is-active", cfg["run_id"] + ".service"],
                         check=False, quiet=True).stdout.strip()
    if state in ("active", "activating", "deactivating", "reloading"):
        raise RuntimeError("Le run de référence est encore actif")
    pending = host.obj("pods", "-n", cfg["namespace"], "-l", LABEL + "=" + cfg["run_id"]).get("items", [])
    if any(host.alive(p) for p in pending):
        raise RuntimeError("Un pod du run de référence est encore actif")


def install_host(repo: Path, manifest: dict) -> tuple[Path, str]:
    """Copy precisely the manifested nightops files, never a .bak or local cache."""
    files = {name: h for name, h in manifest["files"].items() if name.startswith("nightops/")}
    ident = digest(files)
    target = host.INSTALL_ROOT / ident[:16]
    target.mkdir(parents=True, exist_ok=True, mode=0o755)
    if target.is_symlink():
        raise RuntimeError("Installation hôte symbolique refusée")
    os.chown(target, 0, 0)
    os.chmod(target, 0o755)
    for name, expected in files.items():
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            raise RuntimeError("Lien symbolique dans l'installation hôte")
        if dest.exists() and sha(dest) != expected:
            raise RuntimeError("Installation hôte existante modifiée : " + str(dest))
        if not dest.exists():
            dest.write_bytes((repo / name).read_bytes())
        os.chown(dest, 0, 0)
        os.chmod(dest, 0o644)
    for d in target.rglob("*"):
        if d.is_symlink():
            raise RuntimeError("Lien symbolique dans l'installation hôte")
        if d.is_dir():
            os.chown(d, 0, 0)
            os.chmod(d, 0o755)
    return target, ident


def immediate_units(r: Path, cfg: dict) -> dict:
    units = host.unit_files(r, cfg)
    # Do not create the future launch timer. Start this .service directly.
    del units[cfg["run_id"] + ".timer"]
    units[cfg["run_id"] + ".service"] = units[cfg["run_id"] + ".service"].replace(
        "Type=simple", "Type=exec"
    )
    return units


def arm_now(r: Path, cfg: dict, unit_root: Path = Path("/etc/systemd/system")) -> None:
    units = immediate_units(r, cfg)
    for name, text in units.items():
        p = unit_root / name
        if p.exists():
            raise RuntimeError("Collision de nom systemd, aucun remplacement : " + name)
        p.write_text(text, encoding="utf-8")
    host.command(["systemd-analyze", "verify", *[unit_root / name for name in units]], quiet=True)
    host.command(["systemctl", "daemon-reload"], quiet=True)
    write(r / "SCHEDULED", {"at": host.utc(), "mode": "immediate"})
    write(r / "IMMEDIATE_AUTHORIZED.json", {
        "at": host.utc(), "epoch": time.time(), "run_id": cfg["run_id"],
        "confirmation": CONFIRMATION, "scope": "CPU now then temporary daytime vLLM outage for GPU",
    })
    write(r / "STARTING.json", {"at": host.utc(), "epoch": time.time()})
    host.heartbeat(r)
    host.command(["systemctl", "enable", "--now", cfg["run_id"] + "-guard.timer"], quiet=True)
    active = host.command(["systemctl", "is-active", cfg["run_id"] + "-guard.timer"], check=False, quiet=True)
    if active.returncode:
        raise RuntimeError("Gardien non actif : service principal NON démarré")
    host.command(["systemctl", "start", cfg["run_id"] + ".service"], quiet=True)


def observe_start(r: Path, cfg: dict, seconds: int = 150) -> str:
    """Do not mistake `systemctl start` success for application success."""
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        for marker in ("FAILED.json", "RESTORED.json", "CANCEL"):
            if (r / marker).is_file():
                error = read(r / "FAILED.json") if (r / "FAILED.json").exists() else {"reason": marker}
                raise RuntimeError("La campagne s'est arrêtée au démarrage : " + json.dumps(error, ensure_ascii=False))
        state = host.command(["systemctl", "is-active", cfg["run_id"] + ".service"],
                             check=False, quiet=True).stdout.strip()
        if state in ("failed", "inactive", "unknown"):
            raise RuntimeError("Service de campagne " + state + "; consulter journalctl, pas relancer")
        job = r / "jobs" / (cfg["run_id"] + "-inventory.json")
        if (r / "STARTED").exists() and job.exists():
            pods = host.obj("pods", "-n", cfg["namespace"], "-l",
                            "job-name=" + cfg["run_id"] + "-inventory").get("items", [])
            for p in pods:
                if p.get("status", {}).get("phase") == "Failed":
                    raise RuntimeError("Le pod d'inventaire a échoué; consulter son journal")
                if p.get("status", {}).get("phase") in ("Running", "Succeeded"):
                    write(r / "LAUNCH_ACK.json", {"at": host.utc(), "inventory_pod": p["metadata"]["name"],
                          "phase": p["status"]["phase"], "not_a_scientific_completion": True})
                    return "STARTED_WITH_INVENTORY_POD"
        time.sleep(2)
    return "START_REQUESTED_NOT_YET_CONFIRMED"


def start_now(args) -> None:
    if args.confirm != CONFIRMATION:
        raise ValueError("Autorisation de coupure exigée : --confirm COUPURE-VLLM-AUTORISEE")
    # Serialize creation without waiting for another user's ongoing launch.
    lock = host.STATE_ROOT / "immediate-launch.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _start_now(args)
    finally:
        os.close(fd)


def validate_output_root(output_root: Path) -> None:
    if output_root == Path("/") or not str(output_root).startswith("/home/"):
        raise RuntimeError("Choisir une sortie dédiée sous /home, pas /tmp ni une racine système")


def _start_now(args) -> None:
    repo = Path(args.repo).resolve()
    validated = verify_release(repo)
    source = host.location(check_id(args.reuse_prepared_run))
    if read(source / "PREPARED.json").get("preflight") != "PASS":
        raise RuntimeError("Le run de référence n'a pas de préflight validé")
    if not (source / "RESTORED.json").is_file():
        raise RuntimeError("Le run de référence n'est pas terminé/restauré; pas de reprise implicite")
    old = read(source / "config.json")
    original_preflight = read(Path(old["artifacts"]) / "preflight.json")
    if original_preflight.get("status") != "PASS":
        raise RuntimeError("Rapport de préflight worker de référence invalide")
    ensure_no_other_run(source)
    host.clock_ok()
    print("[1/5] Vérification de l'image worker existante, sans rebuild ni téléchargement...", flush=True)
    reuse = verify_worker_image(old, repo)
    print("[2/5] Contrôle vLLM réel (/usr/bin/python3, HTTP et petite inférence)...", flush=True)
    host.idle_check(old)
    checks = host.health(old, True)
    print("VLLM_TEST_COMPLET_OK : " + checks["model"], flush=True)
    install, release_hash = install_host(repo, validated["manifest"])
    cfg = read(repo / "nightops/config.json")
    cfg.update(immediate_window(time.time(), args.max_hours))
    if args.cpu_workers is not None:
        if args.cpu_workers not in (1, 2):
            raise ValueError("1 ou 2 workers CPU autorisés")
        cfg["cpu_workers"] = args.cpu_workers
    ident = "qrn-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg.update({"run_id": ident, "configmap": ident + "-config", "install": str(install),
                "release_sha256": release_hash, "host_release_version": VERSION,
                "repo_head": validated["head"], "image": reuse["image"], "image_id": reuse["image_id"],
                "operator_uid": repo.stat().st_uid, "operator_gid": repo.stat().st_gid,
                "reused_preflight_run": source.name, "demo_prompts": read(repo / "nightops/prompts.json")})
    output_root = Path(args.output_root).resolve()
    validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for claim in (cfg["data_claim"], cfg["cache_claim"]):
        pvc = host.obj("pvc", claim, "-n", cfg["namespace"])
        if pvc.get("status", {}).get("phase") != "Bound":
            raise RuntimeError("PVC non Bound : " + claim)
        pv = host.obj("pv", pvc["spec"]["volumeName"])
        local = pv["spec"].get("local", {}).get("path") or pv["spec"].get("hostPath", {}).get("path")
        if local and output_root.is_relative_to(Path(local).resolve()):
            raise RuntimeError("Les résultats ne doivent pas être écrits dans les PVC historiques")
    if shutil.disk_usage(output_root).free < cfg["min_free_gib"] * 2**30:
        raise RuntimeError("Espace disque insuffisant; aucun nettoyage automatique")
    cfg["tolerations"] = host.obj("deployment", cfg["api_deployment"], "-n", cfg["namespace"])["spec"]["template"]["spec"].get("tolerations", [])
    r = host.location(ident)
    r.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = output_root / ident
    out.mkdir(mode=0o755)
    # Root's umask must not hide the run from UID 10001 in Kubernetes.
    os.chmod(out, 0o755)
    artifacts = out / "artifacts"
    artifacts.mkdir(mode=0o750)
    os.chmod(artifacts, 0o750)
    os.chown(artifacts, 10001, 10001)
    cfg.update(artifacts=str(artifacts), output_dir=str(out))
    write(r / "config.json", cfg)
    write(r / "IMAGE_REUSE.json", reuse)
    write(r / "preflight-vllm.json", checks)
    write(host.STATE_ROOT / "latest.json", {"run_id": ident})
    print("RUN_ID=" + ident, flush=True)
    try:
        cm = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": cfg["configmap"], "namespace": cfg["namespace"], "labels": {LABEL: ident}},
              "data": {"config.json": json.dumps(cfg)}}
        host.k("create", "-f", "-", data=json.dumps(cm))
        print("[3/5] Préflight CPU sur l'image réutilisée; vLLM reste en service...", flush=True)
        end = time.time() + 1200
        job = host.launch(r, cfg, "preflight", end)
        results = host.wait(r, cfg, [job], end)
        if results[job] != "success" or not (artifacts / "preflight.json").is_file():
            raise RuntimeError("Préflight CPU échoué : " + str(r / "logs"))
        pf = read(artifacts / "preflight.json")
        if pf.get("status") != "PASS" or pf.get("plan_sha256") != original_preflight.get("plan_sha256"):
            raise RuntimeError("Préflight incomplet ou plan historique modifié")
        host.idle_check(cfg)
        write(r / "PREPARED.json", {"at": host.utc(), "preflight": "PASS", "vllm_checks": checks,
              "worker_image_reused": True, "startup_mode": "immediate"})
        print("[4/5] Activation du gardien, puis démarrage du service MAINTENANT...", flush=True)
        arm_now(r, cfg)
        host.log(r, "DÉMARRAGE IMMÉDIAT DEMANDÉ ; arrêt calculs " + cfg["stop_paris"])
    except BaseException as exc:
        write(r / "FAILED.json", {"at": host.utc(), "phase": "PREPARATION_OR_STARTUP",
              "type": type(exc).__name__, "error": str(exc)[:5000], "scientific_success": False})
        try:
            host.recover(r)
        except Exception as recovery:
            write(r / "RECOVERY_ERROR.json", {"at": host.utc(), "error": str(recovery)[:5000]})
        raise
    # If operator disconnects during this observation, the systemd process and guard remain independent.
    print("[5/5] Vérification du démarrage réel du service et du pod d'inventaire...", flush=True)
    result = observe_start(r, cfg)
    if result == "STARTED_WITH_INVENTORY_POD":
        print("CAMPAGNE_DEMARREE : service actif et pod d'inventaire démarré.", flush=True)
    else:
        print("INITIALISATION_EN_COURS : service lancé mais pod non encore confirmé; consulter status/logs.", flush=True)
    print("RUN_ID=" + ident)
    print("Arrêt calcul : " + cfg["stop_paris"] + " ; objectif vLLM : " + cfg["ready_paris"])
    print("ARTIFACTS=" + cfg["artifacts"])
    print("Les phases s'enchaînent; cette confirmation ne signifie pas que l'entraînement est terminé.")
