from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'prooftag_qr/e042_decoder_failure_localization.py'


def _text():
    return MODULE.read_text(encoding='utf-8')


def test_module_parses_and_freezes_nine_states():
    ast.parse(_text())
    text = _text()
    assert 'e042-decoder-failure-localization-v1' in text
    assert 'QR_PADDING_PX = 78' in text
    assert 'QR_MODULE_SIZE = 20' in text
    assert 'QR_CORE_MODULES = 29' in text
    assert text.count('SelectedState("') == 9


def test_e042_is_diagnostic_not_optimizer():
    text = _text()
    assert 'diagnostic_only' in text
    assert 'production_ready": False' in text
    assert 'generalization_authorized": False' in text
    assert 'stage1_recomputed": False' in text
    assert 'stage2_recomputed": False' in text
    assert 'srmpgd_recomputed": False' in text


def test_decoder_localization_transforms_are_preregistered():
    text = _text()
    for token in (
        'current-scan-ready', 'raw-vae', 'exact-qz-adaptive', 'exact-qz-white',
        'otsu', 'adaptive', 'grid-mean-050', 'grid-mean-best', 'grid-center-best'
    ):
        assert token in text


def test_notebooks_are_valid_json():
    for name in ('39_e042_decoder_failure_localization.ipynb','40_e042_diagnostic_pipeline_visualizer.ipynb'):
        doc = json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
        assert doc['nbformat'] == 4
        assert len(doc['cells']) >= 5
