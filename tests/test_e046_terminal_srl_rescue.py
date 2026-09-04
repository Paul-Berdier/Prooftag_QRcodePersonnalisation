from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_terminal_srl_rescue_is_fail_closed_and_resume_aware():
    text = (
        ROOT / "scripts/resume-e046-terminal-srl.sh"
    ).read_text(encoding="utf-8")

    assert "local upstream SRL port diverged from the pinned official class" in text
    assert "scientific_fidelity_mismatch" in text
    assert '"retryable": False' in text
    assert '"usable": False' in text
    assert "GENERATION_COMPLETE.json" in text
    assert "terminal-refinements" in text
    assert "score_all_refinements" in text
    assert "aggregate(" in text
    assert "verify(" in text
    assert "rm -rf" not in text
    assert "upstream_reference_atol" not in text
    assert "upstream_reference_rtol" not in text


def test_rescue_uses_plan_image_identity_not_new_git_head():
    text = (
        ROOT / "scripts/resume-e046-terminal-srl.sh"
    ).read_text(encoding="utf-8")

    assert 'plan_source_commit="${identity[1]}"' in text
    assert 'deployed_commit" != "$plan_source_commit' in text
    assert 's|__SOURCE_COMMIT__|$plan_source_commit|g' in text
