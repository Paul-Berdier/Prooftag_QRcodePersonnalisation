from pathlib import Path
import json
ROOT=Path(__file__).parents[1]

def test_e041_notebooks_are_valid_and_distinct():
    for name in ['37_e041_gamma_functional_frontier.ipynb','38_e041_final_pipeline_visualizer.ipynb']:
        p=ROOT/'notebooks'/name
        data=json.loads(p.read_text(encoding='utf-8'))
        assert data['nbformat']==4
        text=''.join(''.join(c.get('source',[])) for c in data['cells'])
        assert 'E041' in text
    assert (ROOT/'notebooks/37_e041_gamma_functional_frontier.ipynb').read_bytes() != (ROOT/'notebooks/38_e041_final_pipeline_visualizer.ipynb').read_bytes()
