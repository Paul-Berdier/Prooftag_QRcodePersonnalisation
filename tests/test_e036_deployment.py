from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "scripts/deploy-e036-notebook.sh"
RUN = ROOT / "scripts/run-e036-trust-region.sh"
JOB = ROOT / "deploy/k8s/e036-trust-region-job.yaml"


def test_e036_bash_scripts_have_valid_syntax() -> None:
    subprocess.run(["bash", "-n", str(DEPLOY)], check=True)
    subprocess.run(["bash", "-n", str(RUN)], check=True)


def test_e036_job_is_gpu_exclusive_and_gamma_runner_is_used() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert 'nvidia.com/gpu: "1"' in text
    assert "prooftag_qr.e036_trust_region" in text
    assert "__PARENT_DIR__" in text
    assert "__E035_RESULTS_DIR__" in text
    assert "__RESULTS_DIR__" in text


def test_run_script_waits_for_pod_before_following_logs() -> None:
    text = RUN.read_text(encoding="utf-8")
    assert "condition=Ready" in text
    assert "logs -n" in text
    assert "condition=complete" in text
    assert "scale deployment \"$api_deployment\"" in text
