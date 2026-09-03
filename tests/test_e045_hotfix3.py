from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_e045_final_notebook_validation_does_not_require_local_docker_runtime():
    text = (ROOT / "scripts/deploy-e045-notebook.sh").read_text(encoding="utf-8")
    assert 'notebook_container="${PROOFTAG_QR_NOTEBOOK_CONTAINER:-notebook}"' in text
    assert 'Runtime notebook E045 OK:' in text
    assert 'docker run --rm -i --entrypoint python "$notebook_image"' not in text
