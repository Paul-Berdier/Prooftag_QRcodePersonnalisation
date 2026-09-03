from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_foundation_job_is_cpu_only_and_fail_closed():
    doc = yaml.safe_load(
        (ROOT / "deploy/k8s/e045-foundation-job.yaml").read_text(encoding="utf-8")
    )
    spec = doc["spec"]
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    assert spec["backoffLimit"] == 0
    assert "runtimeClassName" not in pod
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert "--force-recover-stale" in container["args"]


def test_scripts_never_delete_persistent_e045_data_or_follow_logs():
    run = (ROOT / "scripts/run-e045-foundation.sh").read_text(encoding="utf-8")
    deploy = (ROOT / "scripts/deploy-e045-notebook.sh").read_text(encoding="utf-8")
    assert "rm -rf" not in run
    assert "rm -rf" not in deploy
    assert "logs -f" not in run
    assert "backoffLimit" not in run
    assert "Aucun résultat /data n'a été supprimé" in deploy


def test_notebook_image_runtime_check_receives_heredoc_on_stdin():
    deploy = (ROOT / "scripts/deploy-e045-notebook.sh").read_text(encoding="utf-8")
    assert 'docker run --rm -i --entrypoint python "$notebook_image" -' in deploy
