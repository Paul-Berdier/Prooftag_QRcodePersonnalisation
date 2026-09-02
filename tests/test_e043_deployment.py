from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def test_e043_job_has_nvidia_runtime_and_one_gpu():
    doc=yaml.safe_load((ROOT/'deploy/k8s/e043-scanner-cell-frontier-job.yaml').read_text(encoding='utf-8'))
    spec=doc['spec']['template']['spec']; c=spec['containers'][0]
    assert spec['runtimeClassName']=='nvidia'
    assert c['resources']['requests']['nvidia.com/gpu']=='1'
    assert c['resources']['limits']['nvidia.com/gpu']=='1'

def test_e043_runner_is_polling_and_never_runs_e041():
    text=(ROOT/'scripts/run-e043-scanner-cell-frontier.sh').read_text(encoding='utf-8')
    assert 'sleep 30' in text
    assert 'logs -f' not in text
    assert 'run-e041-gamma-functional-frontier.sh' not in text
    assert 'Aucune suppression automatique' in text
