from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "build_e035_srmpgd_loss_fidelity_gate_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "30_e035_srmpgd_loss_fidelity_gate.ipynb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_e035_notebook_builder_is_deterministic() -> None:
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    first = digest(NOTEBOOK)
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    assert digest(NOTEBOOK) == first


def test_e035_notebook_is_fail_closed_and_has_no_generation_cell() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "verify_parent_artifact" in source
    assert "verdict.json absent" in source
    assert "e035_paper_srl_control" in source
    assert "e035_upstream_code_srl" in source
    assert "37 presets × 3 répétitions" in source
    assert "production_ready" in source
    assert 'method == "e033_public_demo_srpg_from_fixed_e034_stage1"' in source
    assert 'method == "e033_public_demo_srpg_exact_e034_export"' in source
    assert "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079" in source
    assert "/v1/lab/campaigns" not in source
    assert "generate_diffqrcoder_qr(" not in source
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
