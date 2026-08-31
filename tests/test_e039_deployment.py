from __future__ import annotations

from pathlib import Path


def test_e039_runner_avoids_live_log_following() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/run-e039-limiter-scanaware.sh").read_text(encoding="utf-8")
    assert "kubectl logs -f" not in text
    assert "ATTENTE E039 (polling)" in text
    assert "43200" in text


def test_e039_remote_refuses_before_verdict() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/e039-remote.ps1").read_text(encoding="utf-8")
    assert "verdict.json" in text
    assert "E039 n'est pas encore termine" in text
    assert "Replace(\"`r`n\", \"`n\")" in text


def test_e039_job_uses_gpu_and_determinism_env() -> None:
    root = Path(__file__).parents[1]
    text = (root / "deploy/k8s/e039-limiter-scanaware-job.yaml").read_text(encoding="utf-8")
    assert 'nvidia.com/gpu: "1"' in text
    assert "CUBLAS_WORKSPACE_CONFIG" in text
    assert "e039_limiter_scanaware" in text
