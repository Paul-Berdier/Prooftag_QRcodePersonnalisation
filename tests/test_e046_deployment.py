from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_jobs_fail_closed_and_mount_persistent_data():
    for name in ("e046-parent-job.yaml", "e046-refinement-job.yaml"):
        doc = yaml.safe_load(
            (ROOT / "deploy/k8s" / name).read_text(encoding="utf-8")
        )
        assert doc["spec"]["backoffLimit"] == 0
        pod = doc["spec"]["template"]["spec"]
        assert pod["runtimeClassName"] == "nvidia"
        container = pod["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
        assert any(
            volume["persistentVolumeClaim"]["claimName"] == "prooftag-qr-data"
            for volume in pod["volumes"]
            if "persistentVolumeClaim" in volume
        )


def test_run_script_is_resume_aware_and_never_deletes_data():
    text = (ROOT / "scripts/run-e046-controlled-campaign.sh").read_text(
        encoding="utf-8"
    )
    assert "list-parents" in text
    assert "list-refinements" in text
    assert "GENERATION_COMPLETE" in text
    assert "rm -rf" not in text
    assert "logs -f" not in text
    assert "backoffLimit" not in text
    assert "Le Job n'est pas supprimé" in text


def test_deploy_runtime_check_uses_kubernetes_not_docker_run():
    text = (ROOT / "scripts/deploy-e046-notebooks.sh").read_text(
        encoding="utf-8"
    )
    assert 'deployment/"$notebook_deployment"' in text
    assert '-c "$notebook_container"' in text
    assert "Runtime notebook E046 OK" in text
    assert "docker run --rm" not in text


def test_run_script_restores_api_after_completed_or_failed_jobs():
    text = (ROOT / "scripts/run-e046-controlled-campaign.sh").read_text(
        encoding="utf-8"
    )
    assert "restore_runtime_if_idle" in text
    assert "restore-runtime)" in text
    assert "on_exit()" in text
    assert "API laissée à 0 pour ne pas voler la RTX" in text
    assert "API=1, notebook=0, vLLM=0" in text


def test_remote_notebook_refuses_to_compete_with_active_gpu_job():
    text = (ROOT / "scripts/e046-remote.ps1").read_text(encoding="utf-8")
    assert "status.phase=Running" in text
    assert "Un Job GPU E046 est actif" in text
    assert "notebook non demarre" in text


def test_failed_gpu_job_requires_explicit_retry_authorization():
    text = (ROOT / "scripts/run-e046-controlled-campaign.sh").read_text(
        encoding="utf-8"
    )
    assert 'retry_failed_jobs="${PROOFTAG_E046_RETRY_FAILED:-0}"' in text
    assert "Aucune relance identique automatique" in text
    assert "PROOFTAG_E046_RETRY_FAILED=1" in text
    assert "Job $job réussi mais marqueur GENERATION_COMPLETE absent" in text
