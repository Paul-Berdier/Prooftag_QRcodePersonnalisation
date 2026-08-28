from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "32_e037_prospective_global_trust_holdout.ipynb"
BUILDER = ROOT / "scripts" / "build_e037_prospective_global_trust_holdout_notebook.py"


def test_e037_notebook_exists_and_all_code_cells_parse() -> None:
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert notebook.metadata["prooftag"]["experiment"] == "e037-prospective-global-trust-mini-holdout-v1"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_e037_notebook_displays_global_and_per_case_images() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "e037-final-contact-sheet.png" in text
    assert "comparison.png" in text
    assert "DisplayImage" in text
    assert "for row in summary.to_dict" in text
    assert "10 comparaisons" in text or "Comparaisons visuelles cas par cas" in text


def test_e037_builder_is_deterministic_by_construction() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    assert 'cell["id"] = f"e037-{index:02d}"' in text
    assert 'newline="\\n"' in text
