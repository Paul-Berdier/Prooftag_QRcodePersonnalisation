from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "scripts" / "deploy-e037-notebook.sh"
RUN = ROOT / "scripts" / "run-e037-holdout.sh"
JOB = ROOT / "deploy" / "k8s" / "e037-holdout-job.yaml"
REMOTE = ROOT / "scripts" / "e037-remote.ps1"


def test_e037_bash_scripts_parse() -> None:
    subprocess.run(["bash", "-n", str(DEPLOY)], check=True)
    subprocess.run(["bash", "-n", str(RUN)], check=True)


def test_e037_job_is_gpu_and_runs_frozen_module() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "kind: Job" in text
    assert 'nvidia.com/gpu: "1"' in text
    assert "prooftag_qr.e037_holdout" in text
    assert "--e036-results-dir" in text
    assert "--source-commit" in text
    assert "CUBLAS_WORKSPACE_CONFIG" in text


def test_e037_runner_avoids_follow_logs_fsnotify_path() -> None:
    text = RUN.read_text(encoding="utf-8")
    assert "polling sans watcher" in text
    assert "logs -n" in text
    assert "--tail=300" in text
    assert "--all-containers=true -f &" not in text


def test_e037_remote_is_crlf_safe() -> None:
    text = REMOTE.read_text(encoding="utf-8")
    assert "ToBase64String" in text
    assert 'Replace("`r`n", "`n")' in text
    assert "base64 --decode | bash" in text
