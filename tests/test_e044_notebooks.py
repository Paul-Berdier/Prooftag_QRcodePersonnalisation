from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def test_notebooks():
    for n in ['45_e044_multiprompt_complete_audit.ipynb','46_e044_visual_atlas.ipynb']:
        d=json.loads((ROOT/'notebooks'/n).read_text(encoding='utf-8'))
        assert d['nbformat']==4
        txt=json.dumps(d,ensure_ascii=False)
        assert 'E044' in txt and '/data/e044-multi-prompt-best-pipeline-v1' in txt
