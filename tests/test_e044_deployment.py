from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_job_gpu_runtime():
    d=yaml.safe_load((ROOT/'deploy/k8s/e044-prompt-job.yaml').read_text(encoding='utf-8'))
    s=d['spec']['template']['spec']; c=s['containers'][0]
    assert s['runtimeClassName']=='nvidia'
    assert c['resources']['limits']['nvidia.com/gpu']=='1'
def test_runner_polling():
    t=(ROOT/'scripts/run-e044-multiprompt-benchmark.sh').read_text(encoding='utf-8')
    assert 'sleep 30' in t and 'logs -f' not in t
