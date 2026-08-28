from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
CAPTURE = ROOT / "prooftag_qr" / "e035_parent_capture.py"
CAPTURE_JOB = ROOT / "deploy" / "k8s" / "e035-parent-capture-job.yaml"
PARENT_ARTIFACT = ROOT / "prooftag_qr" / "e035_parent_artifact.py"
RUNNER = ROOT / "prooftag_qr" / "e035_loss_fidelity.py"


def test_parent_capture_freezes_stage2_only_from_exact_e034_stage1() -> None:
    text = CAPTURE.read_text(encoding="utf-8")
    required = (
        'DEFAULT_PAYLOAD = "https://ptag.io/t/e033"',
        "seed: int = 51001",
        "srpg_steps: int = 40",
        "srpg_controlnet_scale: float = 1.05",
        "srpg_qr_weight: float = 50.0",
        "srpg_perceptual_weight: float = 20.0",
        'stage2_initialization: str = "public_random"',
        'stage2_target_mode: str = "binary_exact"',
        'UPSTREAM_REVISION = "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00"',
        "E034_OBSERVED_STAGE1_IMAGE_SHA256",
        "E034_OBSERVED_STAGE1_FILE_SHA256",
        "backend._run_stage2(stage1, blueprint, request, config.seed)",
        '"stage1_regenerated": False',
        '"parent_origin": "stage2_replayed_from_exact_e034_stage1"',
        "export_parent_artifact(",
    )
    for value in required:
        assert value in text
    for forbidden in (
        "backend.generate(",
        "backend._run_stage1(",
        "backend.variants(",
        "torch.load(",
    ):
        assert forbidden not in text


def test_shared_parent_contract_pins_the_archived_stage1_hashes() -> None:
    text = PARENT_ARTIFACT.read_text(encoding="utf-8")
    assert "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079" in text
    assert "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71" in text
    assert "generation.stage1_regenerated must be exactly false" in text


def test_parent_capture_job_requires_the_fixed_stage1_and_both_hashes() -> None:
    text = CAPTURE_JOB.read_text(encoding="utf-8")
    for required in (
        "--stage1-image",
        "__STAGE1_IMAGE__",
        "--expected-stage1-image-sha256",
        "__STAGE1_IMAGE_SHA256__",
        "--expected-stage1-file-sha256",
        "__STAGE1_FILE_SHA256__",
        "e035-parent-capture-from-e034-stage1-v1",
    ):
        assert required in text


def test_loss_runner_cannot_implicitly_capture_or_regenerate_parent() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "e035_parent_capture",
        "capture_parent(",
        "backend.generate(",
        "_run_stage1(",
        "_run_stage2(",
    )
    for value in forbidden:
        assert value not in text
