from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "prooftag_qr" / "e035_loss_fidelity.py"


def test_e035_runner_freezes_the_authorized_protocol() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "max_iterations: int = 4" in text
    assert "step_size: float = 1000.0" in text
    assert "gradient_scale: float = 32768.0" in text
    assert "lpips_weight: float = 0.01" in text
    assert 'BRANCH_PAPER = "e035_paper_srl_control"' in text
    assert 'BRANCH_UPSTREAM = "e035_upstream_code_srl"' in text
    assert "ConservativeQRVerifyScorer" in text
    assert "repetitions=3" in text
    assert '"production_ready": False' in text
    assert '"advisor_training_authorized": False' in text
    assert '"automatic_expansion_authorized": False' in text
    assert '"mini_holdout_authorized": decision.startswith("GO_MINI_HOLDOUT")' in text
    assert 'output_dir / "branch-pairing.json"' in text
    assert 'output_dir / "parent-fp32-redecoded.png"' in text
    assert "E034_OBSERVED_STAGE1_IMAGE_SHA256" in text
    assert "E034_OBSERVED_STAGE1_FILE_SHA256" in text


def test_upstream_branch_uses_and_crosschecks_the_actual_pinned_class() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert (
        "from diffqrcoder.losses.scanning_robust_loss import "
        "ScanningRobustLoss"
    ) in text
    assert "official_upstream_srl(srl_unit, upstream_target)" in text
    assert "torch.allclose(" in text
    assert 'phase="evaluation"' in text
    assert 'phase="gradient"' in text
    assert '"upstream_reference_match": True' in text


def test_e035_runner_does_not_call_generation_entrypoints() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "generation_service.generate",
        "backend.generate(",
        "_run_stage1(",
        "_run_stage2(",
        "/v1/lab/campaigns",
    )
    for value in forbidden:
        assert value not in text
