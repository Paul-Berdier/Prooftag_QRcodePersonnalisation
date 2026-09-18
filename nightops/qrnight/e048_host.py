"""Superviseur E048 root-side : SR-MPGD nocturne avec restauration vLLM garantie.

E048 ne modifie jamais les artefacts E047 : ils sont montés read-only dans les Jobs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

from . import host
from .common import LABEL, read, write, sha, digest, schedule, alive, gpu_requested, utc, check_id

VERSION = "1.2.0-e048-e040-canary"
EXPECTED_BASE_COMMIT = "7de284647bc526d1b206e2be21614d724f51e29e"
STATE_ROOT = Path("/var/lib/prooftag-qr-e048")
INSTALL_ROOT = Path("/opt/prooftag-qr-e048")
DEFAULT_OUTPUT = Path("/home/paul/qr-night-runs")
CONFIRM = "COUPURE-VLLM-AUTORISEE"


def root_only() -> None:
    if os.geteuid() != 0:
        raise SystemExit("Exécuter avec sudo.")


def location(run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = read(STATE_ROOT / "latest.json")["run_id"]
    return STATE_ROOT / "runs" / check_id(run_id)


def log(r: Path, message: str) -> None:
    print(f"{utc()} {message}", flush=True)
    with (r / "events.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{utc()} {message}\n")


def verify_release(repo: Path) -> dict:
    manifest_path = repo / "MANIFEST_E048_SRMPGD.json"
    manifest = read(manifest_path)
    if manifest.get("release_version") != VERSION:
        raise RuntimeError("MANIFEST_E048_SRMPGD.json absent ou mauvaise version")
    for name, expected in manifest.get("files", {}).items():
        path = repo / name
        if path.is_symlink() or not path.resolve().is_relative_to(repo):
            raise RuntimeError("chemin release E048 invalide : " + name)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("fichier E048 différent du ZIP validé : " + name)
    git = ["git", "-c", "safe.directory=" + str(repo), "-C", repo]
    head = host.command([*git, "rev-parse", "HEAD"], quiet=True).stdout.strip()
    host.command([*git, "merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT, head], quiet=True)
    # E048 est additive : les briques E047 utilisées pour orchestration/scoring ne doivent pas
    # avoir changé depuis le commit validé.
    protected = [
        "nightops/qrnight/__init__.py", "nightops/qrnight/common.py",
        "nightops/qrnight/host.py", "nightops/qrnight/worker.py",
        "nightops/Dockerfile", "nightops/config.json", "nightops/prompts.json",
    ]
    host.command([*git, "diff", "--quiet", EXPECTED_BASE_COMMIT, "--", *protected], quiet=True)
    dirty = host.command([*git, "status", "--porcelain", "--untracked-files=no"], quiet=True).stdout.strip()
    if dirty:
        raise RuntimeError("modifications suivies non commitées : commit/push avant prepare")
    return {"manifest": manifest, "repo_head": head}


def source_state(source_run_id: str) -> tuple[Path, dict]:
    source = host.STATE_ROOT / "runs" / check_id(source_run_id)
    if not source.is_dir():
        raise FileNotFoundError("run E047 source introuvable : " + source_run_id)
    for marker in ("PREPARED.json", "STARTED", "FINISHED.json", "RESTORED.json"):
        if not (source / marker).exists():
            raise RuntimeError(f"run E047 source incomplet : {marker} absent")
    if (source / "FAILED.json").exists():
        raise RuntimeError("run E047 source marqué FAILED")
    finished = read(source / "FINISHED.json")
    if (
        int(finished.get("completed_generations", -1)) != 36
        or finished.get("report_exists") is not True
        or finished.get("advisor_trained") is not True
    ):
        raise RuntimeError("run E047 source non complet : 36 générations + advisor + rapport exigés")
    cfg = read(source / "config.json")
    if cfg.get("repo_head") != EXPECTED_BASE_COMMIT:
        raise RuntimeError(
            "le run E047 source ne provient pas du commit validé attendu : "
            + str(cfg.get("repo_head"))
        )
    artifacts = Path(cfg["artifacts"])
    if not artifacts.is_dir():
        raise FileNotFoundError("artefacts E047 source absents")
    return source, cfg


def no_other_active_run() -> None:
    root = STATE_ROOT / "runs"
    if not root.exists():
        return
    for run in root.iterdir():
        if not run.is_dir():
            continue
        if (run / "SCHEDULED").exists() and not (run / "RESTORED.json").exists():
            raise RuntimeError("une campagne E048 est déjà armée/active : " + run.name)


def build_image(repo: Path, output: Path, base_image: str, release_hash: str) -> tuple[str, str]:
    local = host.command(["docker", "image", "inspect", base_image], check=False, quiet=True)
    docker_root = Path(host.command(["docker", "info", "--format", "{{.DockerRootDir}}"], quiet=True).stdout.strip())
    needed = 18 if local.returncode else 5
    if shutil.disk_usage(docker_root).free < needed * 2**30:
        raise RuntimeError(f"stockage Docker insuffisant : {needed} Gio libres requis")
    if local.returncode:
        canonical = base_image if "/" in base_image else "docker.io/library/" + base_image
        archive = output / "e048-base.tar"
        host.command(["/usr/local/bin/k3s", "ctr", "images", "export", str(archive), canonical], timeout=1800)
        host.command(["docker", "load", "-i", archive], timeout=1800)
        archive.unlink()
    image = "prooftag-qr-e048:" + release_hash[:16]
    host.command([
        "docker", "build", "--pull=false", "--build-arg", "BASE_IMAGE=" + base_image,
        "-t", image, "-f", repo / "nightops/Dockerfile.e048", repo / "nightops",
    ], timeout=1800)
    inspected = json.loads(host.command(["docker", "image", "inspect", image], quiet=True).stdout)[0]
    iid = inspected["Id"]
    archive = output / "e048-image.tar"
    host.command(["docker", "save", "-o", archive, image], timeout=1800)
    host.command(["/usr/local/bin/k3s", "ctr", "images", "import", archive], timeout=1800)
    archive.unlink()
    return image, iid


def manifest(cfg: dict, action: str, name: str, deadline: float, gpu: bool) -> dict:
    seconds = max(1, int(deadline - time.time()))
    resources = {
        "requests": {"cpu": "2", "memory": "4Gi"},
        "limits": {"cpu": "4", "memory": "24Gi"},
    }
    if gpu:
        resources["requests"].update({"nvidia.com/gpu": "1", "memory": "8Gi"})
        resources["limits"].update({"nvidia.com/gpu": "1", "memory": "40Gi"})
    container = {
        "name": "worker",
        "image": cfg["image"],
        "imagePullPolicy": "Never",
        "command": ["python", "-B", "-m", "qrnight.e048_worker", action, "--deadline", str(deadline)],
        "resources": resources,
        "envFrom": [{"configMapRef": {"name": "prooftag-qr-config"}}],
        "env": [
            {"name": "QRNIGHT_IMAGE", "value": cfg["image"]},
            {"name": "QRNIGHT_IMAGE_DIGEST", "value": cfg["image_id"]},
            {"name": "HF_HOME", "value": "/cache/huggingface"},
            {"name": "TORCH_HOME", "value": "/cache/torch"},
            {"name": "XDG_CACHE_HOME", "value": "/cache/xdg"},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            {"name": "PROOFTAG_QR_MODEL_CACHE_DIR", "value": "/cache"},
            {"name": "PROOFTAG_QR_DATA_DIR", "value": "/night/scratch"},
            {"name": "PROOFTAG_QR_DATABASE_BACKEND", "value": "sqlite"},
            {"name": "PROOFTAG_QR_DATABASE_URL_OVERRIDE", "value": "sqlite:////night/scratch/unused.sqlite3"},
            {"name": "MPLCONFIGDIR", "value": "/tmp/matplotlib"},
            {"name": "PYTHONHASHSEED", "value": "0"},
        ] + ([] if gpu else [
            {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
        ]),
        "volumeMounts": [
            {"name": "history", "mountPath": "/data", "readOnly": True},
            {"name": "source", "mountPath": "/source", "readOnly": True},
            {"name": "out", "mountPath": "/night"},
            {"name": "cache", "mountPath": "/cache"},
            {"name": "config", "mountPath": "/config", "readOnly": True},
            {"name": "shm", "mountPath": "/dev/shm"},
        ],
        "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
    }
    pod = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "nodeSelector": {"kubernetes.io/hostname": cfg["node"]},
        "securityContext": {"runAsUser": 10001, "runAsGroup": 10001},
        "terminationGracePeriodSeconds": 60,
        "containers": [container],
        "volumes": [
            {"name": "history", "persistentVolumeClaim": {"claimName": cfg["data_claim"], "readOnly": True}},
            {"name": "source", "hostPath": {"path": cfg["source_artifacts"], "type": "Directory"}},
            {"name": "out", "hostPath": {"path": cfg["artifacts"], "type": "Directory"}},
            {"name": "cache", "persistentVolumeClaim": {"claimName": cfg["cache_claim"]}},
            {"name": "config", "configMap": {"name": cfg["configmap"]}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}},
        ],
    }
    if gpu:
        pod["runtimeClassName"] = "nvidia"
    if cfg.get("tolerations"):
        pod["tolerations"] = cfg["tolerations"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": cfg["namespace"], "labels": {LABEL: cfg["run_id"], "prooftag.io/e048": "true"}},
        "spec": {"backoffLimit": 0, "activeDeadlineSeconds": seconds,
                 "template": {"metadata": {"labels": {LABEL: cfg["run_id"], "prooftag.io/e048": "true"}}, "spec": pod}},
    }


def launch(r: Path, cfg: dict, action: str, deadline: float, gpu: bool = False) -> str:
    with host.locked(host.STATE_ROOT / "mutation.lock"):
        if (r / "CANCEL").exists():
            raise RuntimeError("run E048 annulé")
        name = f"{cfg['run_id']}-{action}"
        m = manifest(cfg, action, name, deadline, gpu)
        write(r / "jobs" / f"{name}.json", m)
        host.k("create", "-f", "-", data=json.dumps(m))
        log(r, "Job créé : " + name)
        return name


def job_state(cfg: dict, name: str) -> str:
    d = host.obj("job", name, "-n", cfg["namespace"])
    if d["metadata"].get("labels", {}).get(LABEL) != cfg["run_id"]:
        raise RuntimeError("job non détenu par E048")
    status = d.get("status", {})
    if status.get("succeeded", 0):
        return "success"
    if status.get("failed", 0) or any(c.get("type") == "Failed" and c.get("status") == "True" for c in status.get("conditions", [])):
        return "failed"
    return "running"


def collect_logs(r: Path, cfg: dict, name: str) -> None:
    p = host.k("logs", "-n", cfg["namespace"], "job/" + name, "--tail=4000", check=False)
    (r / "logs").mkdir(exist_ok=True)
    (r / "logs" / f"{name}.txt").write_text(p.stdout + p.stderr, encoding="utf-8")


def wait_job(r: Path, cfg: dict, name: str, deadline: float) -> str:
    wall = time.time(); mono = time.monotonic()
    while time.time() < deadline and time.monotonic() - mono < max(0, deadline - wall):
        if (r / "CANCEL").exists():
            raise RuntimeError("arrêt demandé par le gardien")
        if abs((time.time() - wall) - (time.monotonic() - mono)) > 30:
            raise RuntimeError("saut d'horloge détecté")
        host.heartbeat(r)
        state = job_state(cfg, name)
        if state != "running":
            collect_logs(r, cfg, name)
            return state
        time.sleep(5)
    collect_logs(r, cfg, name)
    # Ne jamais lancer le rapport pendant qu'un Job précédent écrit encore dans /night.
    # Supprime uniquement le Job portant le label de CE run puis attend la disparition du pod.
    try:
        if job_state(cfg, name) == "running":
            host.delete_owned(cfg, name)
            host.wait_no_owned_pods(cfg, timeout=300)
    except Exception as exc:
        raise RuntimeError("impossible d'arrêter proprement le Job arrivé à deadline") from exc
    return "time_budget_exhausted"


def unit_files(cfg: dict) -> dict[str, str]:
    py = "/usr/bin/python3"
    runner = cfg["install"] + "/nightops"
    env = f"Environment=PYTHONPATH={runner}\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
    name = cfg["run_id"]
    main = f"""[Unit]\nDescription=Prooftag QR E048 SR-MPGD {name}\nAfter=k3s.service network-online.target\n[Service]\nType=simple\n{env}ExecStart={py} -B -m qrnight.e048_host run --run-id {name}\nExecStopPost={py} -B -m qrnight.e048_host recover --run-id {name}\nTimeoutStopSec=1500\nKillMode=control-group\nNice=10\nUMask=0077\n"""
    guard = f"""[Unit]\nDescription=Prooftag QR E048 recovery {name}\nAfter=k3s.service\n[Service]\nType=oneshot\n{env}ExecStart={py} -B -m qrnight.e048_host guard --run-id {name}\nTimeoutStartSec=1500\nUMask=0077\n"""
    fmt = lambda value: datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    timer = f"""[Unit]\nDescription=Prooftag QR E048 start {name}\n[Timer]\nOnCalendar={fmt(cfg['start_epoch'])}\nAccuracySec=1s\nPersistent=false\nUnit={name}.service\n[Install]\nWantedBy=timers.target\n"""
    watchdog = f"""[Unit]\nDescription=Prooftag QR E048 deadline guard {name}\n[Timer]\nOnCalendar={fmt(cfg['stop_epoch'])}\nOnBootSec=30s\nOnUnitInactiveSec=60s\nAccuracySec=1s\nPersistent=true\nUnit={name}-guard.service\n[Install]\nWantedBy=timers.target\n"""
    return {name + ".service": main, name + "-guard.service": guard,
            name + ".timer": timer, name + "-guard.timer": watchdog}


def prepare(args) -> None:
    repo = Path(args.repo).resolve()
    host.clock_ok()
    verified = verify_release(repo)
    no_other_active_run()
    source_dir, source_cfg = source_state(args.source_run)
    # Aucun autre job E047 actif ne doit être masqué.
    source_pods = host.obj("pods", "-n", source_cfg["namespace"], "-l", LABEL + "=" + args.source_run).get("items", [])
    if any(alive(p) for p in source_pods):
        raise RuntimeError("un pod E047 source est encore actif")
    base_cfg = read(repo / "nightops/e048_config.json")
    cfg = {**base_cfg}
    for key in ("namespace", "vllm_namespace", "vllm_deployment", "api_deployment", "notebook_deployment",
                "data_claim", "cache_claim", "node", "source_plan_id", "source_root", "payload"):
        cfg[key] = source_cfg[key]
    cfg.update(schedule(args.start, args.ready_by))
    cfg["source_run_id"] = args.source_run
    cfg["source_artifacts"] = str(Path(source_cfg["artifacts"]).resolve())
    cfg["operator_uid"] = repo.stat().st_uid
    cfg["operator_gid"] = repo.stat().st_gid
    cfg["repo_head"] = verified["repo_head"]
    cfg["run_id"] = "e048-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg["configmap"] = cfg["run_id"] + "-config"
    cfg["tolerations"] = host.obj("deployment", cfg["api_deployment"], "-n", cfg["namespace"])["spec"]["template"]["spec"].get("tolerations", [])
    # vLLM doit être sain au moment de la préparation ; aucune coupure ici.
    host.idle_check(cfg)
    vllm_check = host.health(cfg, True)
    output_root = Path(args.output_root).resolve()
    if output_root == Path("/") or not str(output_root).startswith("/home/"):
        raise RuntimeError("output-root doit être un dossier dédié sous /home")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < int(cfg["min_free_gib"]) * 2**30:
        raise RuntimeError("espace disque insuffisant pour E048")
    r = location(cfg["run_id"])
    r.mkdir(parents=True, exist_ok=False, mode=0o700)
    out = output_root / cfg["run_id"]
    out.mkdir(mode=0o755)
    artifacts = out / "artifacts"
    artifacts.mkdir(mode=0o750)
    os.chown(artifacts, 10001, 10001)
    cfg["output_dir"] = str(out)
    cfg["artifacts"] = str(artifacts)
    # Installer une copie root-owned de TOUT le package nightops. e048_host importe
    # host/common existants ; figer seulement les nouveaux fichiers rendrait le service non autonome.
    source_nightops = repo / "nightops"
    release_files = {
        str(path.relative_to(repo)): sha(path)
        for path in source_nightops.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    }
    release_hash = digest(release_files)
    install = INSTALL_ROOT / release_hash[:16]
    if not install.exists():
        (install / "nightops").mkdir(parents=True)
        shutil.copytree(source_nightops, install / "nightops", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in install.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("symlink interdit dans installation E048")
        os.chown(path, 0, 0)
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    cfg["install"] = str(install)
    cfg["release_sha256"] = release_hash
    print("Construction de l'image E048 additive depuis l'image E047 validée...", flush=True)
    cfg["image"], cfg["image_id"] = build_image(repo, out, source_cfg["image"], release_hash)
    write(r / "config.json", cfg)
    cm = {
        "apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
        "metadata": {"name": cfg["configmap"], "namespace": cfg["namespace"], "labels": {LABEL: cfg["run_id"]}},
        "data": {"config.json": json.dumps(cfg)},
    }
    host.k("create", "-f", "-", data=json.dumps(cm))
    # Préflight réel dans l'image, source montée read-only, GPU masqué.
    name = launch(r, cfg, "preflight", time.time() + 1200, gpu=False)
    state = wait_job(r, cfg, name, time.time() + 1200)
    if state != "success" or not (artifacts / "preflight.json").is_file():
        raise RuntimeError("préflight E048 échoué; vLLM n'a pas été coupé")
    if cfg["start_epoch"] <= time.time() + 120:
        raise RuntimeError("préparation trop tardive pour cette heure de départ")
    write(r / "PREPARED.json", {"at": utc(), "preflight": "PASS", "vllm_checks": vllm_check,
                                 "source_run_id": args.source_run, "source_read_only": True})
    write(STATE_ROOT / "latest.json", {"run_id": cfg["run_id"]})
    print("\nE048 PRÊT, NON PROGRAMMÉ. vLLM reste en service.")
    print("RUN_ID=" + cfg["run_id"])
    print("Départ=" + cfg["start_paris"] + " ; arrêt calcul=" + cfg["stop_paris"] + " ; objectif vLLM=" + cfg["ready_paris"])
    print("ARTIFACTS=" + cfg["artifacts"])


def arm(r: Path, confirmation: str) -> None:
    if confirmation != CONFIRM:
        raise ValueError("autorisation explicite requise : --confirm " + CONFIRM)
    cfg = read(r / "config.json")
    if not (r / "PREPARED.json").exists():
        raise RuntimeError("E048 non préparé")
    if cfg["start_epoch"] < time.time() + 120:
        raise RuntimeError("heure de départ dépassée ou trop proche")
    if (r / "CANCEL").exists():
        raise RuntimeError("run annulé")
    no_other_active_run_except = [x for x in (STATE_ROOT / "runs").iterdir() if x != r and (x / "SCHEDULED").exists() and not (x / "RESTORED.json").exists()]
    if no_other_active_run_except:
        raise RuntimeError("autre run E048 actif")
    host.clock_ok(); host.idle_check(cfg); host.health(cfg, False)
    units = unit_files(cfg)
    for name, text in units.items():
        path = Path("/etc/systemd/system") / name
        if path.exists():
            raise RuntimeError("unité systemd existe déjà : " + name)
        path.write_text(text, encoding="utf-8")
    host.command(["systemd-analyze", "verify", *["/etc/systemd/system/" + n for n in units]], quiet=True)
    host.command(["systemctl", "daemon-reload"], quiet=True)
    write(r / "SCHEDULED", {"at": utc()})
    host.command(["systemctl", "enable", "--now", cfg["run_id"] + "-guard.timer"], quiet=True)
    host.command(["systemctl", "enable", "--now", cfg["run_id"] + ".timer"], quiet=True)
    log(r, "E048 PROGRAMMÉ : " + cfg["start_paris"] + " ; arrêt calcul " + cfg["stop_paris"])


def recover(r: Path) -> None:
    # Le récupérateur E047 est volontairement réutilisé : mêmes labels, même politique de snapshot,
    # même contrôle de template vLLM et petite inférence après restauration.
    host.recover(r, True)


def guard(r: Path) -> None:
    if not (r / "SCHEDULED").exists() or (r / "RESTORED.json").exists():
        return
    cfg = read(r / "config.json")
    should = time.time() >= cfg["stop_epoch"] or (r / "CANCEL").exists()
    if (r / "STARTED").exists() and (r / "heartbeat.json").exists():
        heartbeat = read(r / "heartbeat.json")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        should |= heartbeat["boot"] != boot or time.monotonic() - heartbeat["monotonic"] > 180
    if should:
        try:
            recover(r)
        except Exception as exc:
            write(r / "RECOVERY_ERROR.json", {"at": utc(), "error": str(exc)[:6000]})
            log(r, "ALERTE RESTAURATION E048 : " + str(exc))
            raise


def run(r: Path) -> None:
    try:
        _run(r)
    except Exception as exc:
        write(r / "FAILED.json", {"at": utc(), "type": type(exc).__name__, "error": str(exc)[:6000],
                                   "phase": read(r / "PHASE.json") if (r / "PHASE.json").exists() else "STARTUP",
                                   "scientific_success": False})
        log(r, "ECHEC E048 : " + str(exc))
        raise


def _run(r: Path) -> None:
    cfg = read(r / "config.json")
    work = Path(cfg["artifacts"])
    if not (r / "PREPARED.json").exists() or not (r / "SCHEDULED").exists():
        raise RuntimeError("E048 non préparé/programmé")
    if host.command(["systemctl", "is-active", cfg["run_id"] + "-guard.timer"], check=False, quiet=True).returncode:
        raise RuntimeError("gardien E048 non actif")
    if (r / "STARTED").exists() or (r / "CANCEL").exists():
        raise RuntimeError("run E048 déjà démarré/annulé")
    if not cfg["start_epoch"] <= time.time() < cfg["start_epoch"] + 900:
        raise RuntimeError("fenêtre de démarrage E048 manquée")
    host.clock_ok()
    lockpath = "/tmp/prooftag-e046-large.runner.lock"
    try:
        fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
        os.fchown(fd, cfg["operator_uid"], cfg["operator_gid"])
    except FileExistsError:
        fd = os.open(lockpath, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(r / "snapshot.json", host.idle_check(cfg))
        write(r / "baseline-health.json", host.health(cfg, False))
        write(r / "STARTED", {"at": utc()})
        host.heartbeat(r)
        try:
            write(r / "PHASE.json", {"phase": "VLLM_CUT", "at": utc()})
            host.cut(r, cfg)
            optimize_deadline = cfg["stop_epoch"] - int(cfg.get("report_reserve_seconds", 1800))
            # Canary GPU réel : une itération complète latent -> SR-MPGD -> QR-Verify.
            # Un échec ici stoppe immédiatement la campagne et déclenche recover() dans finally.
            canary_timeout = int(cfg.get("gpu_canary_timeout_seconds", 1200))
            canary_deadline = min(time.time() + canary_timeout, optimize_deadline - 600)
            if canary_deadline <= time.time() + 120:
                raise RuntimeError("fenêtre insuffisante pour le canary GPU E048")
            write(r / "PHASE.json", {"phase": "GPU_CANARY", "at": utc(), "deadline": canary_deadline})
            canary_job = launch(r, cfg, "canary", canary_deadline, gpu=True)
            canary_state = wait_job(r, cfg, canary_job, canary_deadline)
            if canary_state != "success" or not (work / "GPU_CANARY_PASS.json").is_file():
                raise RuntimeError("canary GPU E048 échoué : " + canary_state)
            host.wait_no_owned_pods(cfg)
            write(r / "PHASE.json", {"phase": "SRMPGD_GPU", "at": utc(), "deadline": optimize_deadline})
            job = launch(r, cfg, "optimize", optimize_deadline, gpu=True)
            state = wait_job(r, cfg, job, optimize_deadline)
            if state == "failed":
                raise RuntimeError("Job GPU E048 échoué")
            if not any((work / "optimized").glob("*/result.json")):
                raise RuntimeError("E048 n'a produit aucun candidat optimisé")
            # Rapport CPU borné avant la restauration. Même si la fenêtre GPU a été consommée,
            # les résultats déjà atomiquement écrits restent exploitables.
            report_deadline = cfg["stop_epoch"] - 120
            if time.time() < report_deadline - 120:
                write(r / "PHASE.json", {"phase": "REPORT_CPU", "at": utc()})
                report_job = launch(r, cfg, "report", report_deadline, gpu=False)
                report_state = wait_job(r, cfg, report_job, report_deadline)
            else:
                report_state = "skipped_for_recovery_margin"
            optimization = read(work / "OPTIMIZATION_COMPLETE.json") if (work / "OPTIMIZATION_COMPLETE.json").exists() else {
                "partial_by_deadline": True,
                "done": len(list((work / "optimized").glob("*/result.json"))),
            }
            report_summary = read(work / "report/summary.json") if (work / "report/summary.json").exists() else None
            complete = bool(
                state == "success"
                and int(optimization.get("done", 0)) == int(optimization.get("total", 36))
                and int(optimization.get("failures", 0)) == 0
                and not bool(optimization.get("partial_by_deadline", True))
            )
            write(r / "FINISHED.json", {"at": utc(), "optimization_state": state,
                                         "report_state": report_state, "optimization": optimization,
                                         "report": report_summary, "gpu_canary": read(work / "GPU_CANARY_PASS.json"),
                                         "scientific_success": complete})
        finally:
            recover(r)
    finally:
        os.close(fd)


def status(r: Path) -> None:
    cfg = read(r / "config.json")
    print(json.dumps({k: cfg[k] for k in ("run_id", "source_run_id", "start_paris", "stop_paris", "ready_paris", "artifacts")}, indent=2))
    if (r / "FAILED.json").exists():
        print("CAMPAGNE=ECHEC")
    elif (r / "FINISHED.json").exists():
        print("CAMPAGNE=PIPELINE_TERMINE")
    elif (r / "STARTED").exists():
        print("CAMPAGNE=EN_COURS")
    elif (r / "SCHEDULED").exists():
        print("CAMPAGNE=PROGRAMMEE")
    else:
        print("CAMPAGNE=PREPAREE")
    for name in ("PREPARED.json", "SCHEDULED", "STARTED", "CUT.json", "PHASE.json", "FINISHED.json", "FAILED.json", "RESTORED.json", "RECOVERY_ERROR.json"):
        path = r / name
        if path.exists():
            print(name, path.read_text(encoding="utf-8")[:5000] if path.suffix == ".json" else "OUI")
    canary = Path(cfg["artifacts"]) / "GPU_CANARY_PASS.json"
    if canary.exists():
        print("GPU_CANARY_PASS.json", canary.read_text(encoding="utf-8"))
    canary_failed = Path(cfg["artifacts"]) / "GPU_CANARY_FAILED.json"
    if canary_failed.exists():
        print("GPU_CANARY_FAILED.json", canary_failed.read_text(encoding="utf-8")[:12000])
    progress = Path(cfg["artifacts"]) / "progress.json"
    if progress.exists():
        print("progress.json", progress.read_text(encoding="utf-8"))
    host.command(["systemctl", "list-timers", "--all", cfg["run_id"] + "*", "--no-pager"], check=False)
    print("STATE=" + str(r))


def export(r: Path) -> None:
    cfg = read(r / "config.json")
    base = Path(cfg["artifacts"])
    dest = Path(cfg["output_dir"]) / (cfg["run_id"] + "-livrables.tar.gz")
    with tarfile.open(dest, "w:gz") as archive:
        for path in base.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(base)
            # L'export garde les preuves, meilleurs PNG et GIF, pas les latents ni les centaines de previews.
            if (path.suffix == ".safetensors" or "previews" in rel.parts or
                    "trajectory" in rel.parts or "qr-cache" in rel.parts or
                    "qr-evidence" in rel.parts):
                continue
            archive.add(path, arcname=str(rel), recursive=False)
        for name in ("PREPARED.json", "FINISHED.json", "FAILED.json", "RESTORED.json", "RECOVERY_ERROR.json", "events.log"):
            if (r / name).exists():
                archive.add(r / name, arcname="operations/" + name)
    if os.environ.get("SUDO_UID"):
        os.chown(dest, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    print("ARCHIVE=" + str(dest))


def cleanup(r: Path) -> None:
    if not (r / "RESTORED.json").exists():
        raise RuntimeError("restauration vLLM non validée")
    cfg = read(r / "config.json")
    name = cfg["run_id"]
    host.command(["systemctl", "disable", "--now", name + ".timer", name + "-guard.timer"], check=False, quiet=True)
    cm = host.obj("configmap", cfg["configmap"], "-n", cfg["namespace"])
    if cm["metadata"].get("labels", {}).get(LABEL) == name:
        host.k("delete", "configmap", cfg["configmap"], "-n", cfg["namespace"], "--wait=false", check=False)
    print("Timers E048 désactivés; résultats conservés.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "schedule", "run", "guard", "recover", "status", "logs", "export", "cleanup", "cancel"])
    parser.add_argument("--repo", default=str(Path.cwd()))
    parser.add_argument("--run-id")
    parser.add_argument("--source-run")
    parser.add_argument("--start")
    parser.add_argument("--ready-by")
    parser.add_argument("--confirm")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    root_only()
    STATE_ROOT.mkdir(parents=True, exist_ok=True); os.chmod(STATE_ROOT, 0o700)
    if args.action == "prepare":
        if not args.source_run or not args.start or not args.ready_by:
            parser.error("prepare exige --source-run, --start et --ready-by")
        prepare(args); return
    r = location(args.run_id)
    if args.action == "schedule": arm(r, args.confirm)
    elif args.action == "run": run(r)
    elif args.action == "guard": guard(r)
    elif args.action == "recover": recover(r)
    elif args.action == "cancel":
        cfg = read(r / "config.json")
        host.command(["systemctl", "disable", "--now", cfg["run_id"] + ".timer"], check=False, quiet=True)
        recover(r)
    elif args.action == "status": status(r)
    elif args.action == "logs":
        name = read(r / "config.json")["run_id"]
        os.execvp("journalctl", ["journalctl", "-u", name + ".service", "-u", name + "-guard.service", "-f", "-n", "120"])
    elif args.action == "export": export(r)
    elif args.action == "cleanup": cleanup(r)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERREUR E048 : " + str(exc), file=sys.stderr)
        raise SystemExit(1)
