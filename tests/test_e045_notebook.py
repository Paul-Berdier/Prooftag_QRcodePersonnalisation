import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_e045_notebook_is_complete_and_analysis_only():
    path = ROOT / "notebooks/47_e045_foundation_and_resilience.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 40
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    for phrase in (
        "E000–E044",
        "Les 98 paramètres",
        "Labels téléphone physiques",
        "Reprise : démonstration exécutée",
        "Architecture cible E046–E049",
        "Galerie des images",
    ):
        assert phrase in source
    assert "run_prompt(" not in source
    assert "torch.optim" not in source
