from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_e045_deploy_selects_python3_on_debian_host():
    text = (ROOT / "scripts/deploy-e045-notebook.sh").read_text(encoding="utf-8")
    assert 'command -v python3' in text
    assert 'host_python="python3"' in text
    assert '"$host_python" -m py_compile' in text
    assert 'PROOFTAG_HOST_PYTHON' in text


def test_container_python_calls_are_not_rewritten_to_host_python():
    text = (ROOT / "scripts/deploy-e045-notebook.sh").read_text(encoding="utf-8")
    assert "python - <<'PY'" in text
    assert 'docker run --rm -i --entrypoint python' in text
