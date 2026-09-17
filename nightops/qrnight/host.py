"""Root-side bounded scheduler. Only vLLM replicas and this run's Jobs are mutable.

No kubeconfig is copied, no data PVC is writable in our Jobs, no server clock is changed.
Scientific code runs as UID 10001 without Kubernetes credentials; recovery runs separately.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from . import BASE_COMMIT
from .common import LABEL, read, write, sha, digest, schedule, gpu_requested, alive, check_id, utc

STATE_ROOT = Path("/var/lib/prooftag-qr-nightly")
INSTALL_ROOT = Path("/opt/prooftag-qr-nightly")
KUBE = ["/usr/local/bin/kubectl", "--kubeconfig=/etc/rancher/k3s/k3s.yaml", "--request-timeout=20s"]


def command(args, *, data=None, timeout=60, check=True, quiet=False):
    p = subprocess.run([str(x) for x in args], input=data, text=True, capture_output=True,
                       timeout=timeout, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if check and p.returncode:
        # Don't echo arbitrary pod env, Secrets, config contents or auth material.
        raise RuntimeError(f"Commande {Path(str(args[0])).name} échouée ({p.returncode}): {p.stderr[-1800:]}")
    if not quiet and p.stdout.strip():
        print(p.stdout[-12000:], flush=True)
    return p


def k(*args, data=None, check=True, timeout=60):
    return command([*KUBE, *args], data=data, timeout=timeout, check=check, quiet=True)


def obj(*args):
    return json.loads(k("get", *args, "-o", "json").stdout)


def root_only():
    if os.geteuid() != 0:
        raise SystemExit("Exécuter cette commande avec sudo (aucune installation Python sur l'hôte).")


def location(run_id=None):
    run_id = run_id or read(STATE_ROOT / "latest.json")["run_id"]
    return STATE_ROOT / "runs" / check_id(run_id)


@contextmanager
def locked(path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def log(r, msg):
    print(f"{utc()} {msg}", flush=True)
    with (r / "events.log").open("a", encoding="utf-8") as f:
        f.write(f"{utc()} {msg}\n")


def clock_ok():
    sync = command(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], quiet=True).stdout.strip()
    if sync != "yes":
        raise RuntimeError("NTP non synchronisé : planification refusée, aucune correction automatique de l'heure")


def controllers(cfg):
    result = {}
    for key, ns, name in (("vllm", cfg["vllm_namespace"], cfg["vllm_deployment"]),
                          ("api", cfg["namespace"], cfg["api_deployment"]),
                          ("notebook", cfg["namespace"], cfg["notebook_deployment"])):
        d = obj("deployment", name, "-n", ns)
        result[key] = {"namespace": ns, "name": name, "uid": d["metadata"]["uid"],
                 "replicas": d["spec"].get("replicas", 1),
                 "ready": d.get("status", {}).get("readyReplicas", 0),
                 "template_sha256": digest(d["spec"]["template"]),
                 "image": d["spec"]["template"]["spec"]["containers"][0]["image"]}
    return result


def idle_check(cfg):
    states = controllers(cfg)
    if states["vllm"]["replicas"] != 1 or states["vllm"]["ready"] < 1:
        raise RuntimeError("Etat de départ attendu : vLLM=1 prêt. Ne pas masquer une panne existante.")
    if states["api"]["replicas"] != 0 or states["notebook"]["replicas"] != 0:
        raise RuntimeError("API ou notebook QR actif : refus d'interrompre un autre travail.")
    for h in obj("hpa", "-A").get("items", []):
        if (h["metadata"]["namespace"] == cfg["vllm_namespace"] and
            h["spec"].get("scaleTargetRef", {}).get("name") == cfg["vllm_deployment"]):
            raise RuntimeError("HPA vLLM présent : arrêt refusé")
    for p in obj("pods", "-A").get("items", []):
        if not alive(p) or not gpu_requested(p):
            continue
        if not (p["metadata"]["namespace"] == cfg["vllm_namespace"] and
                p["metadata"].get("labels", {}).get("app") == "vllm"):
            raise RuntimeError("Autre charge GPU active : " + p["metadata"]["namespace"] + "/" + p["metadata"]["name"])
    return states


def health(cfg, inference=True):
    code = '''import json,urllib.request
b="http://127.0.0.1:8000"
assert urllib.request.urlopen(b+"/health",timeout=15).status==200
m=json.load(urllib.request.urlopen(b+"/v1/models",timeout=15))["data"][0]["id"]
'''
    if inference:
        code += '''p={"model":m,"messages":[{"role":"user","content":"Réponds seulement OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
r=urllib.request.Request(b+"/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
v=json.load(urllib.request.urlopen(r,timeout=75));assert v.get("choices")
'''
    code += 'print(json.dumps({"health":True,"model":m,"tiny_inference":' + str(inference) + '}))'
    # This is the interpreter observed in the production vLLM image.
    # Do not install an alias into the live container or change its image.
    p = k("exec", "--request-timeout=105s", "-n", cfg["vllm_namespace"],
          "deployment/" + cfg["vllm_deployment"], "-c", "vllm", "--",
          "/usr/bin/python3", "-c", code, timeout=110)
    result = json.loads(p.stdout)
    if result.get("health") is not True or not result.get("model"):
        raise RuntimeError("Le contrôle vLLM n'a pas renvoyé un état valide")
    return result


def build_image(repo, install, output, base):
    # Docker and k3s use different image stores. Reuse exactly the deployed base.
    local = command(["docker", "image", "inspect", base], check=False, quiet=True)
    docker_root = Path(command(["docker", "info", "--format", "{{.DockerRootDir}}"], quiet=True).stdout.strip())
    required_gib = 35 if local.returncode else 6
    if shutil.disk_usage(docker_root).free < required_gib * 2**30:
        raise RuntimeError(f"Stockage Docker insuffisant ({required_gib} Gio libres requis). Pas de nettoyage automatique.")
    if local.returncode:
        canonical = base if "/" in base else "docker.io/library/" + base
        archive = output / "base-image.tar"
        command(["/usr/local/bin/k3s", "ctr", "images", "export", str(archive), canonical], timeout=1800)
        command(["docker", "load", "-i", archive], timeout=1800)
        archive.unlink()
    image = "prooftag-qr-night:" + install.name[:16]
    command(["docker", "build", "--pull=false", "--build-arg", "BASE_IMAGE=" + base,
             "-t", image, "-f", install / "nightops/Dockerfile", install / "nightops"], timeout=1800)
    iid = json.loads(command(["docker", "image", "inspect", image], quiet=True).stdout)[0]["Id"]
    archive = output / "night-image.tar"
    command(["docker", "save", "-o", archive, image], timeout=1800)
    command(["/usr/local/bin/k3s", "ctr", "images", "import", archive], timeout=1800)
    archive.unlink()
    return image, iid


def make_job(cfg, action, name, deadline, extra=(), gpu=False):
    seconds = max(1, int(deadline - time.time()))
    c = {"name": "worker", "image": cfg["image"], "imagePullPolicy": "Never",
         "command": ["python", "-B", "-m", "qrnight.worker", action,
                     "--deadline", str(deadline), *extra],
         "resources": {"requests": {"cpu": "2", "memory": "4Gi"},
                       "limits": {"cpu": str(cfg["cpu_per_worker"]), "memory": cfg["worker_memory"]}},
         "envFrom": [{"configMapRef": {"name": "prooftag-qr-config"}}],
         "env": [{"name": key, "value": value} for key, value in {
             "QRNIGHT_IMAGE": cfg["image"], "QRNIGHT_IMAGE_DIGEST": cfg["image_id"],
             "OMP_NUM_THREADS": str(cfg["cpu_per_worker"]), "MKL_NUM_THREADS": str(cfg["cpu_per_worker"]),
             "OPENBLAS_NUM_THREADS": str(cfg["cpu_per_worker"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
             "HF_HOME": "/cache/huggingface", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "PROOFTAG_QR_MODEL_CACHE_DIR": "/cache", "PROOFTAG_QR_DATA_DIR": "/night/scratch",
             "PROOFTAG_QR_DATABASE_BACKEND": "sqlite", "PROOFTAG_QR_DATABASE_URL_OVERRIDE": "sqlite:////night/scratch/unused.sqlite3",
             "MPLCONFIGDIR": "/tmp/matplotlib", "PYTHONHASHSEED": "0",
             **({} if gpu else {"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"})
         }.items()],
         "volumeMounts": [{"name": "source", "mountPath": "/data", "readOnly": True},
                          {"name": "out", "mountPath": "/night"},
                          {"name": "cache", "mountPath": "/cache"},
                          {"name": "config", "mountPath": "/config", "readOnly": True},
                          {"name": "shm", "mountPath": "/dev/shm"}],
         "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}}
    if gpu:
        c["resources"]["requests"].update({"nvidia.com/gpu": "1", "memory": "8Gi"})
        c["resources"]["limits"].update({"nvidia.com/gpu": "1", "memory": "32Gi"})
    spec = {"restartPolicy": "Never", "automountServiceAccountToken": False,
            "nodeSelector": {"kubernetes.io/hostname": cfg["node"]},
            "securityContext": {"runAsUser": 10001, "runAsGroup": 10001},
            "terminationGracePeriodSeconds": 30, "containers": [c],
            "volumes": [{"name": "source", "persistentVolumeClaim": {"claimName": cfg["data_claim"], "readOnly": True}},
                        {"name": "out", "hostPath": {"path": cfg["artifacts"], "type": "Directory"}},
                        {"name": "cache", "persistentVolumeClaim": {"claimName": cfg["cache_claim"]}},
                        {"name": "config", "configMap": {"name": cfg["configmap"]}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}}]}
    if gpu:
        spec["runtimeClassName"] = "nvidia"
    if cfg.get("tolerations"):
        spec["tolerations"] = cfg["tolerations"]
    return {"apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": name, "namespace": cfg["namespace"], "labels": {LABEL: cfg["run_id"]}},
            "spec": {"backoffLimit": 0, "activeDeadlineSeconds": seconds,
                     "template": {"metadata": {"labels": {LABEL: cfg["run_id"]}}, "spec": spec}}}


def heartbeat(r):
    write(r / "heartbeat.json", {"at": time.time(), "monotonic": time.monotonic(),
                                 "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})


def launch(r, cfg, action, deadline, extra=(), gpu=False, suffix=""):
    with locked(STATE_ROOT / "mutation.lock"):
        if (r / "CANCEL").exists():
            raise RuntimeError("Run annulé / restauration engagée")
        if gpu and time.time() >= cfg["generate_until"] - 90:
            raise TimeoutError("Budget GPU épuisé")
        name = cfg["run_id"] + "-" + action + suffix
        manifest = make_job(cfg, action, name, deadline, extra, gpu)
        write(r / "jobs" / (name + ".json"), manifest)
        k("create", "-f", "-", data=json.dumps(manifest))
        log(r, "Job créé : " + name)
        return name


def job_status(cfg, name):
    d = obj("job", name, "-n", cfg["namespace"])
    if d["metadata"].get("labels", {}).get(LABEL) != cfg["run_id"]:
        raise RuntimeError("Job non détenu par cette exécution")
    st = d.get("status", {})
    if st.get("succeeded", 0): return "success"
    if st.get("failed", 0) or any(c.get("type") == "Failed" and c.get("status") == "True" for c in st.get("conditions", [])):
        return "failed"
    return "running"


def collect_logs(r, cfg, name):
    p = k("logs", "-n", cfg["namespace"], "job/" + name, "--tail=1500", check=False)
    (r / "logs").mkdir(exist_ok=True)
    (r / "logs" / (name + ".txt")).write_text(p.stdout + p.stderr, encoding="utf-8")


def delete_owned(cfg, name):
    d = obj("job", name, "-n", cfg["namespace"])
    if d["metadata"].get("labels", {}).get(LABEL) != cfg["run_id"]:
        raise RuntimeError("Suppression refusée : job étranger")
    k("delete", "job", name, "-n", cfg["namespace"], "--cascade=foreground", "--wait=false")


def wait(r, cfg, names, deadline):
    pending = set(names); states = {}
    wall, mono = time.time(), time.monotonic()
    while pending and time.time() < deadline and time.monotonic() - mono < max(0, deadline - wall):
        if (r / "CANCEL").exists():
            raise RuntimeError("Arrêt demandé par le gardien")
        if abs((time.time() - wall) - (time.monotonic() - mono)) > 30:
            raise RuntimeError("Saut d'horloge détecté; restauration prioritaire")
        heartbeat(r)
        for name in list(pending):
            state = job_status(cfg, name)
            if state != "running":
                states[name] = state; pending.remove(name); collect_logs(r, cfg, name)
        if pending: time.sleep(5)
    for name in pending:
        collect_logs(r, cfg, name)
        delete_owned(cfg, name)
        states[name] = "time_budget_exhausted"
    write(r / "last-jobs.json", states)
    for name, state in states.items():
        write(r / "job-results" / (name + ".json"), {"job": name, "status": state, "at": utc()})
    return states


def wait_no_owned_pods(cfg, timeout=300):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        pods = obj("pods", "-n", cfg["namespace"], "-l", LABEL + "=" + cfg["run_id"]).get("items", [])
        if not any(alive(p) for p in pods): return
        time.sleep(3)
    raise TimeoutError("Des pods de la campagne ne sont pas encore terminés")


def scale_from_snapshot(cfg, snap, replicas):
    old = snap["vllm"]
    d = obj("deployment", old["name"], "-n", old["namespace"])
    if d["metadata"]["uid"] != old["uid"] or digest(d["spec"]["template"]) != old["template_sha256"]:
        raise RuntimeError("vLLM a changé pendant la campagne : intervention nécessaire, aucun template écrasé")
    if replicas > 0 and d["spec"].get("replicas", 1) not in (0, old["replicas"]):
        raise RuntimeError("Nombre de réplicas modifié par un autre opérateur; pas d'écrasement")
    patch = [{"op": "test", "path": "/metadata/uid", "value": old["uid"]},
             {"op": "test", "path": "/metadata/resourceVersion", "value": d["metadata"]["resourceVersion"]},
             {"op": "replace", "path": "/spec/replicas", "value": replicas}]
    k("patch", "deployment", old["name"], "-n", old["namespace"], "--type=json", "-p", json.dumps(patch))


def cut(r, cfg):
    timer = command(["systemctl", "is-active", cfg["run_id"] + "-guard.timer"], check=False, quiet=True)
    if timer.returncode:
        raise RuntimeError("Gardien indépendant non actif : aucune coupure vLLM")
    with locked(STATE_ROOT / "mutation.lock"):
        if (r / "CANCEL").exists(): raise RuntimeError("Annulé avant coupure")
        now = idle_check(cfg)
        snap = read(r / "snapshot.json")
        if now["vllm"]["uid"] != snap["vllm"]["uid"] or now["vllm"]["template_sha256"] != snap["vllm"]["template_sha256"]:
            raise RuntimeError("vLLM modifié depuis le début de la nuit")
        write(r / "CUT.json", {"at": utc(), "intent_recorded_before_scale": True})
        scale_from_snapshot(cfg, snap, 0)
    until = time.monotonic() + 300
    while time.monotonic() < until:
        active = [p for p in obj("pods", "-A").get("items", []) if alive(p) and gpu_requested(p)]
        if not active:
            mem = command(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], quiet=True).stdout.strip()
            if float(mem.splitlines()[0]) > 2048:
                raise RuntimeError("GPU occupé hors pods attendus, ne pas tuer le processus étranger")
            log(r, "vLLM arrêté ; GPU libéré pour les seuls Jobs de cette campagne")
            return
        time.sleep(3)
    raise TimeoutError("Le GPU n'a pas été libéré dans les cinq minutes")


def recover(r, force=False):
    cfg = read(r / "config.json")
    if (r / "RESTORED.json").exists(): return
    # CANCEL is durable and checked under the same lock as every Job creation.
    with locked(STATE_ROOT / "mutation.lock"):
        write(r / "CANCEL", {"at": utc()})
        jobs = obj("jobs", "-n", cfg["namespace"], "-l", LABEL + "=" + cfg["run_id"]).get("items", [])
        for j in jobs:
            name = j["metadata"]["name"]
            collect_logs(r, cfg, name)
            if not j.get("status", {}).get("succeeded", 0) and not j.get("status", {}).get("failed", 0):
                delete_owned(cfg, name)
    wait_no_owned_pods(cfg)
    if not (r / "CUT.json").exists():
        write(r / "RESTORED.json", {"at": utc(), "vllm_was_not_stopped": True})
        log(r, "Terminé sans coupure vLLM")
        return
    snap = read(r / "snapshot.json")
    with locked(STATE_ROOT / "mutation.lock"):
        for p in obj("pods", "-A").get("items", []):
            if alive(p) and gpu_requested(p) and not (p["metadata"]["namespace"] == cfg["vllm_namespace"] and
                                                       p["metadata"].get("labels", {}).get("app") == "vllm"):
                raise RuntimeError("Autre pod GPU détecté à la restitution; arrêt des manipulations")
        scale_from_snapshot(cfg, snap, snap["vllm"]["replicas"])
    k("rollout", "status", "deployment/" + cfg["vllm_deployment"], "-n", cfg["vllm_namespace"],
      "--timeout=900s", timeout=930)
    check = health(cfg, True)
    baseline = read(r / "baseline-health.json")
    if check["model"] != baseline["model"]:
        raise RuntimeError("Le modèle annoncé au redémarrage n'est plus celui du départ")
    write(r / "RESTORED.json", {"at": utc(), "checks": check,
           "before_requested_deadline": time.time() <= cfg["ready_epoch"],
           "scope": "vLLM health + models + tiny inference; not the entire Microsoft/RAG UI"})
    log(r, "vLLM restauré et petite inférence réussie : " + check["model"])


def guard(r):
    if not (r / "SCHEDULED").exists() or (r / "RESTORED.json").exists(): return
    cfg = read(r / "config.json")
    should = time.time() >= cfg["stop_epoch"] or (r / "CANCEL").exists()
    if (r / "STARTING.json").exists() and not (r / "STARTED").exists():
        should |= time.time() - read(r / "STARTING.json")["epoch"] > 180
    if (r / "STARTED").exists() and (r / "heartbeat.json").exists():
        h = read(r / "heartbeat.json")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        should |= h["boot"] != boot or time.monotonic() - h["monotonic"] > 180
    if should:
        try: recover(r)
        except Exception as exc:
            write(r / "RECOVERY_ERROR.json", {"at": utc(), "error": str(exc)})
            log(r, "ALERTE RESTAURATION : " + str(exc))
            raise


def run(r):
    """A recovered service is not a successful scientific run."""
    try:
        return _run_pipeline(r)
    except Exception as exc:
        failure = {"at": utc(), "type": type(exc).__name__, "error": str(exc)[:6000],
                   "phase": read(r / "PHASE.json") if (r / "PHASE.json").exists() else "STARTUP",
                   "scientific_success": False}
        write(r / "FAILED.json", failure)
        log(r, "ECHEC CAMPAGNE : " + failure["error"])
        raise


def _run_pipeline(r):
    cfg = read(r / "config.json"); work = Path(cfg["artifacts"])
    if not (r / "PREPARED.json").exists() or not (r / "SCHEDULED").exists():
        raise RuntimeError("Run non validé et programmé : utiliser prepare puis schedule")
    if command(["systemctl", "is-active", cfg["run_id"] + "-guard.timer"], check=False, quiet=True).returncode:
        raise RuntimeError("Timer de restauration indépendant absent")
    if (r / "STARTED").exists() or (r / "CANCEL").exists():
        raise RuntimeError("Run déjà commencé ou annulé; aucune reprise GPU implicite")
    if cfg.get("launch_mode") == "immediate":
        auth = read(r / "IMMEDIATE_AUTHORIZED.json")
        if (auth.get("run_id") != cfg["run_id"]
                or auth.get("confirmation") != "COUPURE-VLLM-AUTORISEE"
                or not auth["epoch"] <= time.time() < auth["epoch"] + 300
                or time.time() >= cfg["stop_epoch"]):
            raise RuntimeError("Autorisation immédiate absente, expirée ou invalide")
    elif not cfg["start_epoch"] <= time.time() < cfg["start_epoch"] + 900:
        raise RuntimeError("Fenêtre de démarrage manquée; ne pas démarrer en journée")
    clock_ok()
    # Hold the existing E046 scheduler's lock as well as our own instance lock.
    lockpath = "/tmp/prooftag-e046-large.runner.lock"
    try:
        fd = os.open(lockpath, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)
        os.fchown(fd, cfg["operator_uid"], cfg["operator_gid"])
    except FileExistsError:
        fd = os.open(lockpath, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(r / "snapshot.json", idle_check(cfg))
        write(r / "baseline-health.json", health(cfg, False))
        write(r / "STARTED", {"at": utc()})
        heartbeat(r)
        try:
            write(r / "PHASE.json", {"phase": "INVENTORY", "at": utc()})
            inv = launch(r, cfg, "inventory", min(time.time() + 1200, cfg["score_until"] - 120))
            states = wait(r, cfg, [inv], min(time.time() + 1200, cfg["score_until"] - 120))
            if states[inv] != "success" or not (work / "inventory.json").is_file():
                raise RuntimeError("Inventaire incomplet en erreur")
            write(r / "PHASE.json", {"phase": "SCORING_CPU", "at": utc()})
            jobs = [launch(r, cfg, "score", cfg["score_until"],
                           ["--shard", str(i), "--shards", str(cfg["cpu_workers"])], suffix=f"-{i}")
                    for i in range(cfg["cpu_workers"])]
            scoring_states = wait(r, cfg, jobs, cfg["score_until"])
            if not any((work / "scores").glob("*.json")):
                raise RuntimeError("Aucune observation scorée : " + json.dumps(scoring_states))
            wait_no_owned_pods(cfg)  # no concurrent writes while training snapshot is frozen
            train_end = min(cfg["train_until"], time.time() + 2700)
            if train_end > time.time() + 180:
                write(r / "PHASE.json", {"phase": "TRAINING_CPU", "at": utc()})
                tr = launch(r, cfg, "train", train_end)
                wait(r, cfg, [tr], train_end)
            if time.time() < cfg["generate_until"] - cfg["max_generation_seconds"]:
                q = launch(r, cfg, "tasks", time.time() + 180)
                s = wait(r, cfg, [q], time.time() + 180)
                if s[q] == "success":
                    generation = read(work / "generation-plan.json")["tasks"]
                    if generation:
                        write(r / "PHASE.json", {"phase": "GENERATION_GPU", "at": utc(),
                              "planned_tasks": len(generation)})
                        cut(r, cfg)
                    for i, task in enumerate(generation):
                        if time.time() > cfg["generate_until"] - 180: break
                        end = min(time.time() + cfg["max_generation_seconds"], cfg["generate_until"])
                        j = launch(r, cfg, "generate", end, ["--task", task["id"]], gpu=True, suffix=f"-{i:02d}")
                        wait(r, cfg, [j], end)
                        wait_no_owned_pods(cfg)
            if time.time() < cfg["stop_epoch"] - 90:
                end = min(time.time() + 1000, cfg["stop_epoch"] - 30)
                write(r / "PHASE.json", {"phase": "REPORT_CPU", "at": utc()})
                j = launch(r, cfg, "report", end)
                wait(r, cfg, [j], end)
            training_path = work / "model/training-report.json"
            training = read(training_path) if training_path.is_file() else {}
            completed = len(list((work / "generated").glob("*/result.json")))
            write(r / "FINISHED.json", {"at": utc(),
                  "report_exists": (work / "report/index.html").is_file(),
                  "advisor_trained": training.get("trained", False),
                  "completed_generations": completed,
                  "scope": "Pipeline terminé ou borné; inspecter chaque résultat et les échecs."})
        finally:
            recover(r)
    finally:
        os.close(fd)


def unit_files(r, cfg):
    runner = cfg["install"] + "/nightops"
    py = "/usr/bin/python3"
    env = f"Environment=PYTHONPATH={runner}\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n"
    name = cfg["run_id"]
    main = f"""[Unit]
Description=Prooftag QR nightly {name}
After=k3s.service network-online.target
[Service]
Type=simple
{env}ExecStart={py} -B -m qrnight.host run --run-id {name}
ExecStopPost={py} -B -m qrnight.host recover --run-id {name}
TimeoutStopSec=1200
KillMode=control-group
Nice=10
UMask=0077
"""
    restore = f"""[Unit]
Description=Prooftag QR independent recovery {name}
After=k3s.service
[Service]
Type=oneshot
{env}ExecStart={py} -B -m qrnight.host guard --run-id {name}
TimeoutStartSec=1500
UMask=0077
"""
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    start = f"""[Unit]
Description=Single nightly start {name}
[Timer]
OnCalendar={fmt(cfg['start_epoch'])}
AccuracySec=1s
Persistent=false
Unit={name}.service
[Install]
WantedBy=timers.target
"""
    watchdog = f"""[Unit]
Description=Deadline and stalled-run recovery {name}
[Timer]
OnCalendar={fmt(cfg['stop_epoch'])}
OnBootSec=30s
OnUnitInactiveSec=60s
AccuracySec=1s
Persistent=true
Unit={name}-guard.service
[Install]
WantedBy=timers.target
"""
    return {name + ".service": main, name + ".timer": start,
            name + "-guard.service": restore, name + "-guard.timer": watchdog}


def prepare(args):
    repo = Path(args.repo).resolve(); clock_ok()
    git = ["git", "-c", "safe.directory=" + str(repo), "-C", repo]
    head = command([*git, "rev-parse", "HEAD"], quiet=True).stdout.strip()
    command([*git, "merge-base", "--is-ancestor", BASE_COMMIT, head], quiet=True)
    command([*git, "diff", "--quiet", BASE_COMMIT, "--", "prooftag_qr", "data", "qr_verify_bridge"])
    # Adding this overlay as a new commit is allowed; modifying scientific core is not.
    cfg = read(repo / "nightops/config.json")
    cfg["operator_uid"] = repo.stat().st_uid
    cfg["operator_gid"] = repo.stat().st_gid
    cfg["repo_head"] = head
    cfg.update(schedule(args.start, args.ready_by))
    cfg["demo_prompts"] = read(repo / "nightops/prompts.json")
    cfg["run_id"] = "qrn-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg["configmap"] = cfg["run_id"] + "-config"
    states = idle_check(cfg)
    vllm_preflight = health(cfg, True)
    print("vLLM préflight : santé + modèle + petite inférence OK : " + vllm_preflight["model"], flush=True)
    if args.cpu_workers is not None:
        if args.cpu_workers not in (1, 2): raise ValueError("1 ou 2 workers CPU autorisés")
        cfg["cpu_workers"] = args.cpu_workers
    output_root = Path(args.output_root).resolve()
    new_output_root = not output_root.exists()
    output_root.mkdir(parents=True, exist_ok=True)
    if new_output_root:
        os.chown(output_root, cfg["operator_uid"], cfg["operator_gid"])
        os.chmod(output_root, 0o750)
    if shutil.disk_usage(output_root).free < cfg["min_free_gib"] * 2**30:
        raise RuntimeError("Espace libre insuffisant pour image et résultats")
    source = repo / "nightops"
    release_hash = digest({str(p.relative_to(repo)): sha(p) for p in source.rglob("*") if p.is_file() and "__pycache__" not in str(p)})
    install = INSTALL_ROOT / release_hash[:16]
    if not install.exists():
        (install / "nightops").mkdir(parents=True)
        shutil.copytree(source, install / "nightops", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for p in install.rglob("*"):
        if p.is_symlink(): raise RuntimeError("Symlink inattendu dans le code de supervision")
        os.chown(p, 0, 0)
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    cfg["install"] = str(install)
    r = location(cfg["run_id"]); r.mkdir(parents=True, exist_ok=False); os.chmod(r, 0o700)
    write(STATE_ROOT / "latest.json", {"run_id": cfg["run_id"]})
    out = output_root / cfg["run_id"]
    out.mkdir(mode=0o755)
    os.chmod(out, 0o755)
    artifacts = out / "artifacts"; artifacts.mkdir(mode=0o750); os.chown(artifacts, 10001, 10001)
    cfg["artifacts"] = str(artifacts)
    cfg["output_dir"] = str(out)
    for claim in (cfg["data_claim"], cfg["cache_claim"]):
        pvc = obj("pvc", claim, "-n", cfg["namespace"])
        if pvc.get("status", {}).get("phase") != "Bound": raise RuntimeError("PVC non Bound : " + claim)
        pv = obj("pv", pvc["spec"]["volumeName"])
        local = pv["spec"].get("local", {}).get("path") or pv["spec"].get("hostPath", {}).get("path")
        if local:
            protected = Path(local).resolve()
            if output_root == protected or protected in output_root.parents:
                raise RuntimeError("Le dossier de sortie doit être hors du PVC historique et du cache modèles")
    api = obj("deployment", cfg["api_deployment"], "-n", cfg["namespace"])
    cfg["tolerations"] = api["spec"]["template"]["spec"].get("tolerations", [])
    print("Construction de l'image additionnelle (CPU seulement), puis import dans K3s...", flush=True)
    cfg["image"], cfg["image_id"] = build_image(repo, install, out, states["api"]["image"])
    cfg["release_sha256"] = release_hash
    write(r / "config.json", cfg)
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
          "metadata": {"name": cfg["configmap"], "namespace": cfg["namespace"], "labels": {LABEL: cfg["run_id"]}},
          "data": {"config.json": json.dumps(cfg)}}
    k("create", "-f", "-", data=json.dumps(cm))
    print("Validation CPU du plan, des bibliothèques, du scoring et du payload. vLLM reste en service.", flush=True)
    j = launch(r, cfg, "preflight", time.time() + 1200)
    results = wait(r, cfg, [j], time.time() + 1200)
    if results[j] != "success" or not (artifacts / "preflight.json").is_file():
        raise RuntimeError("Préflight CPU échoué. Voir " + str(r / "logs") + ". vLLM n'a pas été arrêté.")
    if cfg["start_epoch"] <= time.time() + 120:
        raise RuntimeError("Préparation trop tardive pour ce départ. Reprogrammer une heure future.")
    write(r / "PREPARED.json", {"at": utc(), "preflight": "PASS"})
    write(STATE_ROOT / "latest.json", {"run_id": cfg["run_id"]})
    print("\nPRÊT, NON PROGRAMMÉ. vLLM n'a pas été arrêté.")
    print("RUN_ID=" + cfg["run_id"])
    print("Départ : " + cfg["start_paris"] + " ; arrêt calcul : " + cfg["stop_paris"] + " ; objectif prêt : " + cfg["ready_paris"])
    print("Résultats : " + cfg["artifacts"])


def arm(r, acknowledge):
    cfg = read(r / "config.json")
    if acknowledge != "COUPURE-VLLM-AUTORISEE":
        raise ValueError("Autorisation explicite requise : --confirm COUPURE-VLLM-AUTORISEE")
    if not (r / "PREPARED.json").is_file(): raise RuntimeError("Préflight non validé")
    if cfg["start_epoch"] < time.time() + 120: raise RuntimeError("Heure de départ dépassée / trop proche")
    if (r / "CANCEL").exists(): raise RuntimeError("Run annulé")
    for other in (STATE_ROOT / "runs").iterdir():
        if other != r and (other / "SCHEDULED").exists() and not (other / "RESTORED.json").exists():
            raise RuntimeError("Une autre nuit est programmée ou active")
    clock_ok(); idle_check(cfg)
    units = unit_files(r, cfg)
    for name, contents in units.items():
        Path("/etc/systemd/system", name).write_text(contents)
    command(["systemd-analyze", "verify", *["/etc/systemd/system/" + name for name in units]], quiet=True)
    command(["systemctl", "daemon-reload"], quiet=True)
    write(r / "SCHEDULED", {"at": utc()})
    # Independent recovery is armed FIRST; if arming it fails, do not arm start.
    command(["systemctl", "enable", "--now", cfg["run_id"] + "-guard.timer"], quiet=True)
    command(["systemctl", "enable", "--now", cfg["run_id"] + ".timer"], quiet=True)
    log(r, "PROGRAMMÉ : " + cfg["start_paris"] + " ; calculs arrêtés " + cfg["stop_paris"])


def status(r):
    cfg = read(r / "config.json")
    print(json.dumps({k: cfg[k] for k in ("run_id", "start_paris", "stop_paris", "ready_paris", "artifacts")}, indent=2))
    if (r / "FAILED.json").exists():
        print("CAMPAGNE=ECHEC (RESTORED ne signifie pas entraînement réussi)")
    elif (r / "FINISHED.json").exists():
        print("CAMPAGNE=PIPELINE_TERMINE; vérifier les résultats scientifiques ci-dessous")
    elif (r / "RESTORED.json").exists():
        print("CAMPAGNE=INTERROMPUE_OU_NON_DEMARREE; restauration seule")
    elif (r / "STARTED").exists():
        print("CAMPAGNE=EN_COURS")
    else:
        print("CAMPAGNE=NON_DEMARREE")
    for name in ("PREPARED.json", "SCHEDULED", "STARTING.json", "STARTED", "PHASE.json", "CUT.json", "FAILED.json", "FINISHED.json", "RESTORED.json", "RECOVERY_ERROR.json"):
        p = r / name
        if p.exists(): print(name, p.read_text()[:2500] if p.suffix == ".json" else "OUI")
    for p in Path(cfg["artifacts"]).glob("score-progress-*.json"):
        print(p.name, p.read_text())
    command(["systemctl", "show", cfg["run_id"] + ".service", "-p", "ActiveState", "-p", "SubState",
             "-p", "Result", "-p", "ExecMainStatus"], check=False)
    command(["systemctl", "list-timers", "--all", cfg["run_id"] + "*", "--no-pager"], check=False)
    print("STATE=" + str(r))


def export(r):
    import tarfile
    cfg = read(r / "config.json"); base = Path(cfg["artifacts"])
    dest = Path(cfg["output_dir"]) / (cfg["run_id"] + "-livrables.tar.gz")
    with tarfile.open(dest, "w:gz") as tf:
        for p in base.rglob("*"):
            if p.is_file() and not p.is_symlink() and "cache" not in p.parts and p.suffix != ".safetensors":
                tf.add(p, arcname=str(p.relative_to(base)), recursive=False)
        for p in (r / "job-results").glob("*.json"):
            tf.add(p, arcname="operations/job-results/" + p.name)
        for name in ("RESTORED.json", "FAILED.json", "FINISHED.json", "PHASE.json", "RECOVERY_ERROR.json", "events.log", "last-jobs.json"):
            if (r / name).exists(): tf.add(r / name, arcname="operations/" + name)
    if os.environ.get("SUDO_UID"):
        os.chown(dest, int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"]))
    print("ARCHIVE=" + str(dest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "start-now", "schedule", "run", "guard", "recover", "status", "logs", "cancel", "export", "cleanup"])
    p.add_argument("--repo", default=str(Path.cwd()))
    p.add_argument("--start")
    p.add_argument("--reuse-prepared-run", help="Run dont l'image worker déjà validée est réutilisée")
    p.add_argument("--max-hours", type=float, default=12,
                   help="Budget total depuis la commande, dernière heure réservée à vLLM (5 à 17 h)")
    p.add_argument("--ready-by")
    p.add_argument("--run-id")
    p.add_argument("--confirm")
    p.add_argument("--cpu-workers", type=int)
    p.add_argument("--output-root", default="/home/paul/qr-night-runs")
    args = p.parse_args(); root_only()
    STATE_ROOT.mkdir(parents=True, exist_ok=True); os.chmod(STATE_ROOT, 0o700)
    if args.action == "start-now":
        if not args.reuse_prepared_run:
            p.error("start-now exige --reuse-prepared-run; aucun ancien run n'est réarmé")
        from .immediate import start_now
        start_now(args); return
    if args.action == "prepare":
        if not args.start or not args.ready_by: p.error("prepare exige --start et --ready-by")
        prepare(args); return
    r = location(args.run_id)
    if args.action == "schedule": arm(r, args.confirm)
    elif args.action == "run": run(r)
    elif args.action == "guard": guard(r)
    elif args.action in ("recover", "cancel"):
        cfg = read(r / "config.json")
        if args.action == "cancel":
            command(["systemctl", "disable", "--now", cfg["run_id"] + ".timer"], check=False, quiet=True)
        recover(r, True)
    elif args.action == "status": status(r)
    elif args.action == "logs":
        name = read(r / "config.json")["run_id"]
        # No subprocess capture here: Ctrl+C only detaches the journal viewer.
        os.execvp("journalctl", ["journalctl", "-u", name + ".service", "-u", name + "-guard.service", "-f", "-n", "100"])
    elif args.action == "export": export(r)
    elif args.action == "cleanup":
        if not (r / "RESTORED.json").exists():
            raise RuntimeError("Restauration non validée : le gardien doit rester actif")
        name = read(r / "config.json")["run_id"]
        command(["systemctl", "disable", "--now", name + ".timer", name + "-guard.timer"], check=False)
        print("Timers de cette exécution désactivés. Données et journaux conservés.")


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print("ERREUR : " + str(exc), file=sys.stderr)
        raise SystemExit(1)
