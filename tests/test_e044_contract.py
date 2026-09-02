from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_contract():
    t=(ROOT/'prooftag_qr/e044_multiprompt_best_pipeline.py').read_text(encoding='utf-8')
    assert 'GAMMAS = (500.0, 1000.0)' in t
    assert 'LATENT_RADIUS_RMS = 0.200' in t
    assert 'MAX_ITERATIONS = 8' in t
    assert t.count('"id": "p0') >= 7
    assert '_decoded_to_exact_scan_ready' in t
    assert 'functional-pattern toning' in t
