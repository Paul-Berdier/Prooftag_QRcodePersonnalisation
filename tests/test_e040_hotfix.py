from __future__ import annotations

from pathlib import Path


def test_api_image_has_advisor_runtime_and_stage1_asset() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")
    assert ".[gpu,quality,advisor-runtime]" in text
    assert "import hpsv2, joblib, lpips, sklearn" in text
    assert "docs/e035-assets/e034-observed-stage1.png" in text


def test_pyproject_declares_small_advisor_runtime_extra() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "advisor-runtime = [" in text
    assert '"joblib>=1.4,<2"' in text
    assert '"scikit-learn>=1.5,<2"' in text


def test_model_bridge_optional_advisor_cannot_destroy_e040() -> None:
    text = Path("prooftag_qr/e040_model_bridge.py").read_text(encoding="utf-8")
    assert '"advisor load/inference failed"' in text
    assert "except Exception as exc" in text


def test_finalizer_is_checkpoint_only_and_preserves_gamma() -> None:
    text = Path("prooftag_qr/e040_finalize.py").read_text(encoding="utf-8")
    assert "expected 45 checkpoints" in text
    assert '"gamma": 1000.0' in text
    assert '"finalized_from_existing_checkpoints": True' in text
    assert "99-FINAL-QR.png" in text
    assert "99-FINAL-latent.safetensors" in text
    assert "full-pipeline-contact-sheet.png" in text


def test_finalize_job_is_cpu_only() -> None:
    text = Path("deploy/k8s/e040-finalize-job.yaml").read_text(encoding="utf-8")
    assert "prooftag_qr.e040_finalize" in text
    assert "nvidia.com/gpu" not in text
    assert "runtimeClassName" not in text


def test_finalize_script_refuses_incomplete_partial() -> None:
    text = Path("scripts/finalize-e040-partial.sh").read_text(encoding="utf-8")
    assert "45/45 checkpoints présents" in text
    assert "gamma=1000" in text
