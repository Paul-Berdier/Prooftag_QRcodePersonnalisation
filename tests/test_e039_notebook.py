from __future__ import annotations

import ast
import json
from pathlib import Path


def test_e039_notebook_exists_and_code_parses() -> None:
    root = Path(__file__).parents[1]
    path = root / "notebooks/34_e039_srmpgd_limiter_scanaware.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")


def test_e039_notebook_shows_images_ssr_and_blockers() -> None:
    root = Path(__file__).parents[1]
    text = (root / "notebooks/34_e039_srmpgd_limiter_scanaware.ipynb").read_text(encoding="utf-8")
    assert "e039-all-methods-contact-sheet.png" in text
    assert "blocker-summary.csv" in text
    assert "qr_verify_exact_presets" in text
    assert "rejection-log.csv" in text
    assert "E038 winner vs E039 winner" in text
