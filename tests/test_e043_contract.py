from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'prooftag_qr/e043_scanner_cell_frontier.py'

def test_e043_scientific_contract_is_static_and_paired():
    text=SRC.read_text(encoding='utf-8'); ast.parse(text)
    assert 'GAMMA = 500.0' in text
    assert 'LATENT_RADIUS_RMS = 0.200' in text
    for name in ('e043_A_control_e041_g500','e043_B_cellvar_g500','e043_C_grid_g500','e043_D_critical_g500','whole_cell_margin_loss','intra_module_variance_penalty','grid_consistency_loss','critical_module_losses','_decoded_to_exact_scan_ready'):
        assert name in text
    assert 'risk-weighted data-module margin proxy; not differentiable Reed-Solomon' in text

def test_e043_does_not_change_prompt_or_claim_generalization():
    text=SRC.read_text(encoding='utf-8')
    assert 'from .e041_gamma_functional_frontier import (' in text
    assert 'PROMPT,' in text
    assert '"generalization_authorized": False' in text
    assert '"production_ready": False' in text
