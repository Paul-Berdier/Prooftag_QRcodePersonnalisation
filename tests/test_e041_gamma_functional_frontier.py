from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]
MODULE = ROOT / 'prooftag_qr/e041_gamma_functional_frontier.py'

def test_e041_module_parses_and_protocol_is_preregistered():
    source = MODULE.read_text(encoding='utf-8')
    ast.parse(source)
    assert 'GAMMAS = (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)' in source
    assert 'LATENT_RADIUS_RMS = 0.200' in source
    assert 'MAX_ITERATIONS = 8' in source
    assert 'FUNCTIONAL_TONE_FACTORS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30)' in source
    assert 'historical_gamma_baseline' in source
    assert 'gamma=1000 est un contrôle historique' in source or 'gamma=1000 est un controle historique' in source

def test_e041_prompt_is_not_e040_greenhouse():
    source = MODULE.read_text(encoding='utf-8')
    assert 'botanical reading room inside a glass conservatory' in source
    assert 'a sunlit greenhouse filled with tomato plants' not in source

def test_e041_does_not_project_data_modules():
    source = MODULE.read_text(encoding='utf-8')
    assert '_functional_tone_exact_diffqrcoder' in source
    assert 'functional_pattern_mask' in source
    assert 'if not bool(functional[row, col])' in source
    assert 'prepare_scan_ready_image(' not in source
    assert 'data modules are not projected' in source
