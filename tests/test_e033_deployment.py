from pathlib import Path


def test_e033_deployment_is_immutable_cpu_orchestrated_and_ssh_safe():
    deployer = Path("scripts/deploy-e033-notebook.sh").read_text(encoding="utf-8")
    app_deployer = Path("scripts/deploy-app-image.sh").read_text(encoding="utf-8")
    notebook_deployer = Path("scripts/deploy-notebook-image.sh").read_text(encoding="utf-8")
    notebook = "28_e033_srmpgd_microdiagnostic.ipynb"

    source_guard = deployer.index('if [[ "${BASH_SOURCE[0]}" != "$0" ]]')
    strict_mode = deployer.index("set -Eeuo pipefail")
    assert source_guard < strict_mode
    assert "Ne pas sourcer ce script" in deployer
    assert "La connexion SSH reste ouverte" in deployer

    app_image = deployer.index("bash scripts/deploy-app-image.sh")
    notebook_image = deployer.index('bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"')
    start = deployer.index('bash scripts/notebook-server.sh start "$notebook"')
    assert app_image < notebook_image < start
    assert "bash scripts/notebook-server.sh stop" in deployer
    assert "kubectl scale deployment/vllm -n vllm --replicas=0" in deployer
    assert 'kubectl scale "deployment/${api_deployment}" -n "$namespace" --replicas=1' in deployer

    assert notebook in deployer
    assert 'runtime_mode" != "advisor-cpu"' in deployer
    assert "$(git rev-parse HEAD)" in deployer
    assert "PROOFTAG_GIT_COMMIT" in deployer
    assert "PROOFTAG_RUNTIME_IMAGE_DIGEST" in deployer
    assert "e033_public_demo_srpg" in deployer
    assert "e033_equation_srmpgd_fp16" in deployer
    assert "e033_equation_srmpgd_fp32" in deployer
    assert 'settings["diffqrcoder_stage2_initialization"] == "public_random"' in deployer
    assert 'settings["srmpgd_max_iterations"] == 4' in deployer
    assert 'settings["srmpgd_gradient_scale"] == 32768.0' in deployer
    assert 'settings["srmpgd_decode_precision"] == precision' in deployer
    assert ".\\\\scripts\\\\notebook-remote.ps1 -Notebook" in deployer

    assert 'docker save "$image" | sudo k3s ctr images import -' in app_deployer
    assert 'docker save "$image" | sudo k3s ctr images import -' in notebook_deployer
    assert "docker save --output" not in app_deployer
    assert "docker save --output" not in notebook_deployer


def test_e033_is_registered_as_a_cpu_campaign_in_both_launchers():
    notebook = "28_e033_srmpgd_microdiagnostic.ipynb"
    server = Path("scripts/notebook-server.sh").read_text(encoding="utf-8")
    remote = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert notebook in server
    assert notebook in remote
    assert server.index(notebook) < server.index("advisor_mode=1")
    assert remote.count(notebook) == 2
