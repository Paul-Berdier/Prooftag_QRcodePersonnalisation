from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_quality_extra_no_longer_requires_git_smart_http():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "hpsv2==1.2.0" in pyproject
    assert "git+https://github.com/tgxs002/HPSv2" not in pyproject

def test_api_dockerfile_pins_hps_source_archive_and_bpe_guard():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HPSV2_COMMIT=866735ecaae999fa714bd9edfa05aa2672669ee3" in text
    assert "codeload.github.com/tgxs002/HPSv2/tar.gz/${HPSV2_COMMIT}" in text
    assert "bpe_simple_vocab_16e6.txt.gz" in text
    assert "git clone https://github.com/jwliao1209/DiffQRCoder.git" not in text
    assert "codeload.github.com/jwliao1209/DiffQRCoder/tar.gz/${DIFFQRCODER_COMMIT}" in text

def test_notebook_dockerfile_avoids_git_clone_for_pinned_sources():
    text = (ROOT / "Dockerfile.notebook").read_text(encoding="utf-8")
    assert "git clone https://github.com/jwliao1209/DiffQRCoder.git" not in text
    assert "codeload.github.com/jwliao1209/DiffQRCoder/tar.gz/${DIFFQRCODER_COMMIT}" in text
