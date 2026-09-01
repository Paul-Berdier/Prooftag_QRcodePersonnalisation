from __future__ import annotations
from pathlib import Path
ROOT=Path(__file__).parents[1]


def test_job_and_scripts_use_e040_contract() -> None:
    job=(ROOT/'deploy/k8s/e040-checkpoint-frontier-job.yaml').read_text(encoding='utf-8')
    run=(ROOT/'scripts/run-e040-checkpoint-frontier.sh').read_text(encoding='utf-8')
    deploy=(ROOT/'scripts/deploy-e040-notebooks.sh').read_text(encoding='utf-8')
    remote=(ROOT/'scripts/e040-remote.ps1').read_text(encoding='utf-8')
    assert 'prooftag_qr.e040_checkpoint_frontier' in job
    assert 'CUBLAS_WORKSPACE_CONFIG' in job
    assert 'Gamma          : 1000 (figé)' in run
    assert 'sleep 30' in run
    assert 'logs -f' not in run
    assert 'checkpoint_count' in run
    assert '35_e040_srmpgd_checkpoint_frontier.ipynb' in deploy
    assert '36_final_qr_pipeline_visualizer.ipynb' in deploy
    assert '-Pipeline' in remote
    assert 'verdict.json' in remote
