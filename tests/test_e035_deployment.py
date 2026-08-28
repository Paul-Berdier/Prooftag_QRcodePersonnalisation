from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "deploy-e035-notebook.sh"
LOSS_JOB = ROOT / "deploy" / "k8s" / "e035-loss-fidelity-job.yaml"
CAPTURE_JOB = ROOT / "deploy" / "k8s" / "e035-parent-capture-job.yaml"
STAGE1 = ROOT / "docs" / "e035-assets" / "e034-observed-stage1.png"


def test_e035_deployment_script_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_e035_deployment_is_exclusive_gpu_and_restores_api() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    assert "run_exclusive_gpu_job" in text
    assert 'scale deployment "$api_deployment"' in text
    assert "--replicas=0" in text
    assert "condition=complete" in text
    assert "trap cleanup_gpu_job RETURN" in text
    assert "prooftag_qr.e035_parent_artifact" in text
    assert "production_ready" in text
    assert "capture-parent" in text
    assert "verify-parent" in text


def test_prepare_uploads_and_verifies_the_exact_e034_stage1() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert STAGE1.is_file()
    assert "upload_stage1_to_pvc" in text
    assert "verify_local_stage1" in text
    assert "verify_remote_stage1" in text
    assert "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079" in text
    assert "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71" in text


def test_parent_verification_reads_json_from_file_and_checks_provenance() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'python - "$metadata_file" "$expected_parent_commit"' in text
    assert 'Path(sys.argv[1]).read_text(encoding="utf-8")' in text
    assert 'method == "e033_public_demo_srpg_from_fixed_e034_stage1"' in text
    assert 'method == "e033_public_demo_srpg_exact_e034_export"' in text
    assert "stage1_file_sha256" in text
    assert "stage1_regenerated" in text
    assert 'python - "$expected_parent_commit" <"$metadata_file"' not in text


def test_e035_loss_job_has_no_stage_generation_command() -> None:
    text = LOSS_JOB.read_text(encoding="utf-8")
    assert "kind: Job" in text
    assert 'nvidia.com/gpu: "1"' in text
    assert "prooftag_qr.e035_loss_fidelity" in text
    assert "--branches" in text and "both" in text
    assert "e035_parent_capture" not in text
    assert "--stage1-image" not in text


def test_parent_capture_is_a_separate_explicit_stage2_only_job() -> None:
    text = CAPTURE_JOB.read_text(encoding="utf-8")
    assert "kind: Job" in text
    assert 'nvidia.com/gpu: "1"' in text
    assert "prooftag_qr.e035_parent_capture" in text
    assert "e035_loss_fidelity" not in text
    assert "--source-commit" in text
    assert "--stage1-image" in text
    assert "--expected-stage1-image-sha256" in text
    assert "--expected-stage1-file-sha256" in text
    assert "https://ptag.io/t/e033" in text
