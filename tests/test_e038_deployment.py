from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
RUN = ROOT / "scripts/run-e038-recipe-frontier.sh"
DEPLOY = ROOT / "scripts/deploy-e038-notebook.sh"
REMOTE = ROOT / "scripts/e038-remote.ps1"
JOB = ROOT / "deploy/k8s/e038-recipe-frontier-job.yaml"


def test_e038_bash_scripts_have_valid_syntax() -> None:
    for path in (RUN, DEPLOY, ROOT / "scripts/download-e038-results.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_e038_job_is_gpu_one_shot_and_has_all_inputs() -> None:
    text = JOB.read_text(encoding="utf-8")
    rendered = (
        text.replace("__JOB_NAME__", "test")
        .replace("__NAMESPACE__", "qr-core")
        .replace("__IMAGE__", "prooftag-qr:test")
        .replace("__PARENT_DIR__", "/data/p")
        .replace("__E035_RESULTS_DIR__", "/data/e35")
        .replace("__E036_RESULTS_DIR__", "/data/e36")
        .replace("__RESULTS_DIR__", "/data/e38")
        .replace("__EXPECTED_PARENT_COMMIT__", "a" * 40)
    )
    document = yaml.safe_load(rendered)
    container = document["spec"]["template"]["spec"]["containers"][0]
    assert document["kind"] == "Job"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "prooftag_qr.e038_recipe_frontier" in " ".join(container["command"])
    assert "--e035-results-dir" in container["args"]
    assert "--e036-results-dir" in container["args"]


def test_e038_runner_avoids_fsnotify_follow_logs() -> None:
    text = RUN.read_text(encoding="utf-8")
    assert "kubectl logs -f" not in text
    assert "logs -n" in text
    assert "polling" in text.lower()
    assert "recipe_count" in text


def test_e038_remote_is_crlf_safe_and_uses_dedicated_notebook() -> None:
    text = REMOTE.read_text(encoding="utf-8")
    assert "ToBase64String" in text
    assert "33_e038_srmpgd_ssr_aesthetic_frontier.ipynb" in text
    assert "advisor-cpu" in text
