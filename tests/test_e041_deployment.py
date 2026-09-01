from pathlib import Path
ROOT = Path(__file__).parents[1]

def test_e041_runner_has_safe_operational_contract():
    s=(ROOT/'scripts/run-e041-gamma-functional-frontier.sh').read_text(encoding='utf-8')
    assert 'logs -f' not in s
    assert 'PARTIAL' in s and 'Aucune suppression automatique' in s
    assert 'sleep 30' in s
    assert '21600' in s
    assert 'Gamma 1000     : baseline historique, PAS valeur imposée' in s

def test_e041_remote_starts_notebook_and_checks_verdict_in_pvc():
    s=(ROOT/'scripts/e041-remote.ps1').read_text(encoding='utf-8')
    assert 'kubectl scale deployment' in s
    assert 'verdict.json' in s
    assert 'kubectl exec' in s
