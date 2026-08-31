from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts/build_e038_recipe_frontier_notebook.py"
NOTEBOOK = ROOT / "notebooks/33_e038_srmpgd_ssr_aesthetic_frontier.ipynb"


def test_e038_notebook_builder_is_deterministic() -> None:
    before = NOTEBOOK.read_bytes()
    subprocess.run(["python", str(BUILDER)], cwd=ROOT, check=True)
    after = NOTEBOOK.read_bytes()
    assert after == before


def test_e038_notebook_shows_all_images_and_metrics_without_early_failure() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "E038 n'est pas encore exécuté" in source
    assert "e038-all-methods-contact-sheet.png" in source
    assert "E035 upstream libre" in source
    assert "Toutes les nouvelles recettes E038 en grand" in source
    assert "SSR QR-Verify exact / 37" in source
    assert "CLIP-Aesthetic" in source or "clip_aesthetic" in source
    assert "hpsv2_1" in source
    assert "FileNotFoundError" not in source
