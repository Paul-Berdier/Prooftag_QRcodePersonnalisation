from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_e041_phase_b_hotfix_contract():
    source = (ROOT / "prooftag_qr/e041_gamma_functional_frontier.py").read_text(encoding="utf-8")
    recovery = (ROOT / "prooftag_qr/e041_recover_phase_b.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/recover-e041-phase-b.sh").read_text(encoding="utf-8")
    job = (ROOT / "deploy/k8s/e041-phase-b-recovery-job.yaml").read_text(encoding="utf-8")
    assert "_functional_tone_exact_diffqrcoder" in source
    assert "expected_canvas = expected_core + 2 * QR_PADDING_PX" in source
    phase_b = source[source.index("phase_b_images"):]
    assert "prepare_scan_ready_image(" not in phase_b
    assert "recovered_from_existing_phase_a" in recovery
    assert "54/54 checkpoints" in runner
    assert "nvidia.com/gpu" not in job
    assert "runtimeClassName" not in job
