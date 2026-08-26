from pathlib import Path


def test_e032_deployment_is_immutable_version_checked_and_ssh_safe():
    deployer = Path("scripts/deploy-e032-notebook.sh").read_text(encoding="utf-8")
    app_deployer = Path("scripts/deploy-app-image.sh").read_text(encoding="utf-8")
    notebook_deployer = Path("scripts/deploy-notebook-image.sh").read_text(
        encoding="utf-8"
    )
    notebook_server = Path("scripts/notebook-server.sh").read_text(encoding="utf-8")
    notebook = "27_e032_srmpgd_paper_reconstruction.ipynb"

    source_guard = deployer.index('if [[ "${BASH_SOURCE[0]}" != "$0" ]]')
    strict_mode = deployer.index("set -Eeuo pipefail")
    assert source_guard < strict_mode
    assert "Ne pas sourcer ce script" in deployer
    assert "La connexion SSH reste ouverte" in deployer

    app_image = deployer.index("bash scripts/deploy-app-image.sh")
    notebook_image = deployer.index(
        'bash scripts/deploy-notebook-image.sh "notebooks/${notebook}"'
    )
    start = deployer.index('bash scripts/notebook-server.sh start "$notebook"')
    reset = deployer.index('bash scripts/notebook-server.sh reset "$notebook"')
    assert app_image < notebook_image < min(start, reset)

    assert notebook in deployer
    assert notebook in notebook_server
    assert 'runtime_mode" != "advisor-cpu"' in deployer
    assert "$(git rev-parse HEAD)" in deployer
    assert "PROOFTAG_GIT_COMMIT" in deployer
    assert "PROOFTAG_RUNTIME_IMAGE_DIGEST" in deployer
    assert "diffqrcoder_paper_srmpgd_guarded" in deployer
    assert "diffqrcoder_paper_srmpgd" in deployer
    assert "paper_equations" in deployer
    assert "guarded_production" in deployer
    assert "diffqrcoder_stage2_target_mode" in deployer
    assert "qart_url_fragment" in deployer
    assert 'paper_settings["srmpgd_max_iterations"]' in deployer
    assert 'guarded_settings["srmpgd_max_iterations"]' in deployer
    assert "== 20" in deployer
    assert 'paper_settings["srmpgd_step_size"]' in deployer
    assert "== 1000.0" in deployer
    assert 'paper_settings["srmpgd_lpips_weight"]' in deployer
    assert "== 0.01" in deployer
    assert 'paper_settings["srmpgd_crop_padding_px"]' in deployer
    assert 'guarded_settings["srmpgd_crop_padding_px"]' in deployer
    assert "== 78" in deployer
    assert "736 - 2 * paper_settings" in deployer
    assert ".\\\\scripts\\\\notebook-remote.ps1 -Notebook" in deployer

    # Les deux grosses images sont importées par pipe, sans archive temporaire.
    assert 'docker save "$image" | sudo k3s ctr images import -' in app_deployer
    assert 'docker save "$image" | sudo k3s ctr images import -' in notebook_deployer
    assert "docker save --output" not in app_deployer
    assert "docker save --output" not in notebook_deployer


def test_generic_api_deployment_accepts_additive_experiment_profiles():
    deployer = Path("scripts/deploy-app-image.sh").read_text(encoding="utf-8")

    assert "required <= set(ids)" in deployer
    assert "len(ids) == len(set(ids))" in deployer
    assert "ids == expected" not in deployer
    assert "20260805-e025-quality-scores-1" not in deployer
    assert "'/lab-assets/app.js?v=' in html" in deployer
