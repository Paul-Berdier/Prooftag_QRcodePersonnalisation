from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).parents[1]


def test_e040_notebooks_exist_and_show_pipeline() -> None:
    frontier=ROOT/'notebooks/35_e040_srmpgd_checkpoint_frontier.ipynb'
    pipeline=ROOT/'notebooks/36_final_qr_pipeline_visualizer.ipynb'
    for path in (frontier,pipeline):
        data=json.loads(path.read_text(encoding='utf-8'))
        assert data['nbformat']==4
        assert len(data['cells']) >= 10
    source=pipeline.read_text(encoding='utf-8')
    for token in ('01-qr-reference.png','03-stage1.png','04-stage2.png','99-FINAL-QR.png','Advisor E026/E031','Modèle E016'):
        assert token in source


def test_notebook_builders_are_deterministic_targets() -> None:
    a=(ROOT/'scripts/build_e040_checkpoint_notebook.py').read_text(encoding='utf-8')
    b=(ROOT/'scripts/build_e040_final_pipeline_notebook.py').read_text(encoding='utf-8')
    assert '35_e040_srmpgd_checkpoint_frontier.ipynb' in a
    assert '36_final_qr_pipeline_visualizer.ipynb' in b
