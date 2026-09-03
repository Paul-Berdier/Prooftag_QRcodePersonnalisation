import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> tuple[dict, str]:
    notebook = json.loads(
        (ROOT / "notebooks" / name).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    return notebook, source


def test_main_notebook_is_visual_and_complete():
    notebook, source = _source("48_e046_controlled_best_generator.ipynb")
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 35
    for phrase in (
        "WeChat exacte / 37",
        "Quiet zone : brut contre scene-preserving",
        "SR-MPGD : trajectoires",
        "Pipeline complète et QR final par prompt",
        "Tournoi automatique multiobjectif par prompt",
        "Préparation E047",
    ):
        assert phrase in source
    assert "torch.optim" not in source


def test_atlas_displays_every_pipeline_stage():
    notebook, source = _source("49_e046_visual_atlas.ipynb")
    assert len(notebook["cells"]) >= 15
    for phrase in (
        "Stage 1 / Stage 2",
        "Toutes les trajectoires SR-MPGD",
        "Différence brut → scene-preserving",
        "Top logiciel sous garde visuelle",
        "Pareto complet",
    ):
        assert phrase in source
