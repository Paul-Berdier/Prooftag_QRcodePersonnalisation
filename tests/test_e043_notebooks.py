from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def test_e043_notebooks_are_valid_json_and_point_to_e043_results():
    for name in ('41_e043_scanner_cell_frontier.ipynb','42_e043_final_pipeline_visualizer.ipynb'):
        doc=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
        assert doc['nbformat']==4
        text=json.dumps(doc,ensure_ascii=False)
        assert '/data/e043-scanner-cell-frontier-v1' in text
        assert 'E043' in text
