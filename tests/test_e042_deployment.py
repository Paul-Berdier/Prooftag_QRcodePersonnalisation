from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_decode_job_requests_one_gpu_but_diagnose_job_does_not():
    decode=yaml.safe_load((ROOT/'deploy/k8s/e042-decode-selected-latents-job.yaml').read_text(encoding='utf-8'))
    diag=yaml.safe_load((ROOT/'deploy/k8s/e042-decoder-diagnostic-job.yaml').read_text(encoding='utf-8'))
    dc=decode['spec']['template']['spec']['containers'][0]
    cc=diag['spec']['template']['spec']['containers'][0]
    assert dc['resources']['limits']['nvidia.com/gpu'] == '1'
    assert 'nvidia.com/gpu' not in cc['resources']['limits']
    assert 'nvidia.com/gpu' not in cc['resources']['requests']


def test_run_script_uses_polling_not_follow_logs():
    text=(ROOT/'scripts/run-e042-decoder-diagnostic.sh').read_text(encoding='utf-8')
    assert 'sleep 15' in text
    assert 'logs -f' not in text
    assert 'run-e041-gamma-functional-frontier.sh' not in text
