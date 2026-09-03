from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_campaign_never_serializes_complete_settings_or_known_secrets():
    text = (ROOT / "prooftag_qr/e046_campaign.py").read_text(encoding="utf-8")
    assert "_safe_settings_provenance" in text
    assert "settings.model_dump" not in text
    assert "database_password" not in text.lower()
    assert "uniform_quiet_zone_replacement" in text


def test_scoring_uses_qr_verify_wechat_and_not_decoder_voting():
    text = (ROOT / "prooftag_qr/e046_campaign.py").read_text(encoding="utf-8")
    bridge_policy = (ROOT / "docs/e046-wechat-scoring-policy.md").read_text(
        encoding="utf-8"
    )
    assert "_score_qr_verify" in text
    assert "wechat_exact_presets" in text
    assert "qr-scanner-wechat" in bridge_policy
    assert "Aucun mélange de votes" in bridge_policy
    for forbidden in ("OpenCVDecoder(", "PyzbarDecoder(", "ZXingCPPDecoder("):
        assert forbidden not in text


def test_generation_and_scoring_are_separate_commands():
    text = (ROOT / "prooftag_qr/e046_campaign.py").read_text(encoding="utf-8")
    for command in (
        '"generate-parent"',
        '"score-parents"',
        '"generate-refinement"',
        '"score-refinements"',
    ):
        assert command in text
    assert "promote_attempt(" in text
    assert "GENERATION_COMPLETE.json" in text
    assert "SCORING_COMPLETE.json" in text
