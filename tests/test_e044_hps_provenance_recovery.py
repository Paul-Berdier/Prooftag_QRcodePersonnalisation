from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_e044_hps_build_records_pep610_revision_and_checks_it():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HPSV2_COMMIT=866735ecaae999fa714bd9edfa05aa2672669ee3" in text
    assert "direct_url.json" in text
    assert "'commit_id':commit" in text
    assert "_installed_distribution_source('hpsv2')" in text
    assert "revision==commit" in text


def test_e044_recovery_refuses_to_recompute_scientific_state():
    text = (ROOT / "scripts/recover-e044-quality-scoring.sh").read_text(encoding="utf-8")
    assert "SR-MPGD     : AUCUN RECALCUL" in text
    assert "_score_prompt(" in text
    assert "_run_trajectory(" not in text
    assert "_fresh_parent(" not in text
    assert "EXPECTED_CHECKPOINTS_PER_PROMPT" in text
    assert "recovered_from_complete_pre_scoring_attempt" in text


def test_e044_aggregate_repairs_attempt_paths_after_atomic_move():
    text = (ROOT / "prooftag_qr/e044_aggregate.py").read_text(encoding="utf-8")
    assert "_canonical_artifact_path" in text
    assert "_repair_row_paths" in text
    assert 'for marker in ("trajectories", "parent", "pipeline", "scoring")' in text
    assert "canonical_prompt_artifact_paths" in text
