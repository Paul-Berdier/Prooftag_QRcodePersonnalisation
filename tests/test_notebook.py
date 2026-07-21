import ast
import json
from pathlib import Path


def test_srpg_notebooks_are_valid_and_all_code_cells_compile():
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))

        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell.get("source", [])),
                    filename=f"{path.name}-cell-{index}",
                )


def test_srpg_notebook_exposes_the_debugging_chain():
    source = Path("notebooks/01_srpg_step_by_step.ipynb").read_text(encoding="utf-8")

    assert "raw.png" in source
    assert "srpg.png" in source
    assert "final.png" in source
    assert "srpg_step_*_x0.png" in source
    assert "srpg_step_*_errors.png" in source
    assert "selected_variant" in source
    assert "archive_path.parent / run_name" in source
    assert "Path.cwd() / '.notebook-cache'" not in source


def test_live_gpu_notebook_generates_instead_of_reading_an_archive():
    source = Path("notebooks/02_generate_live_on_gpu.ipynb").read_text(encoding="utf-8")

    assert "backend.generate(request, blueprint, SEED)" in source
    assert "run_srpg_controlnet_img2img" in source
    assert "preview_callback=show_srpg_step" in source
    assert "repair_backend.variants" in source
    assert "validator.validate" in source
    assert "tarfile" not in source


def test_remote_gpu_notebook_has_an_isolated_kubernetes_runtime():
    dockerfile = Path("Dockerfile.notebook").read_text(encoding="utf-8")
    manifest = Path("deploy/k8s/notebook.yaml").read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")
    server = Path("scripts/notebook-server.sh").read_text(encoding="utf-8")

    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "02_generate_live_on_gpu.ipynb" in launcher
    assert "-WindowStyle Normal" in launcher
    assert '"-N"' not in launcher
    assert '"ExitOnForwardFailure=yes"' in launcher
    assert '[int]$LocalPort = 18888' in launcher
    assert "Assert-LocalPortAvailable" in launcher
    assert "[System.IO.File]::Delete($pidFile)" in launcher
    assert "nvidia.com/gpu: \"1\"" in manifest
    assert "replicas: 0" in manifest
    assert "prooftag-qr-model-cache" in manifest
    assert "--ServerApp.root_dir=/workspace" in manifest
    assert "mountPath: /workspace/results" in manifest
    assert "restore_previous_state" in server
