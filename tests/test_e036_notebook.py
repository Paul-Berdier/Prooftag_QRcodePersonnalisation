from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks/31_e036_gamma1000_trust_region.ipynb"
BUILDER = ROOT / "scripts/build_e036_gamma1000_trust_region_notebook.py"


def test_e036_notebook_exists_and_all_code_cells_parse() -> None:
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_e036_notebook_explicitly_displays_comparison_images() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "e036-final-contact-sheet.png" in text
    assert "E035 upstream non contraint" in text
    assert "e036_gamma1000_global_trust/images/iteration-004.png" in text
    assert "e036_gamma1000_strict_trust/images/iteration-004.png" in text
    assert "e036_gamma1000_local_preserve/images/iteration-004.png" in text
    assert "DisplayImage" in text


def test_builder_escapes_manifest_newline_correctly() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    assert '+ "\\\\n"' in text
