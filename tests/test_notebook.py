import ast
import json
from pathlib import Path


def test_srpg_notebook_is_valid_and_all_code_cells_compile():
    path = Path("notebooks/01_srpg_step_by_step.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")


def test_srpg_notebook_exposes_the_debugging_chain():
    source = Path("notebooks/01_srpg_step_by_step.ipynb").read_text(encoding="utf-8")

    assert "raw.png" in source
    assert "srpg.png" in source
    assert "final.png" in source
    assert "srpg_step_*_x0.png" in source
    assert "srpg_step_*_errors.png" in source
    assert "selected_variant" in source
