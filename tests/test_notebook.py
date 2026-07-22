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
    assert "03_srpg_parameter_search.ipynb" in launcher
    assert "04_e007_contextual_optimizer.ipynb" in launcher
    assert "05_controlnet_model_bakeoff.ipynb" in launcher
    assert "06_nacholmo_generate_live.ipynb" in launcher
    assert "[ValidateSet(" in launcher
    assert "-WindowStyle Normal" in launcher
    assert '"-N"' in launcher
    assert '"ExitOnForwardFailure=yes"' in launcher
    assert "JUPYTER_TARGET" in launcher
    assert "kubectl port-forward" not in launcher
    assert "[int]$LocalPort = 18888" in launcher
    assert "Assert-LocalPortAvailable" in launcher
    assert "[System.IO.File]::Delete($pidFile)" in launcher
    assert 'nvidia.com/gpu: "1"' in manifest
    assert "replicas: 0" in manifest
    assert "prooftag-qr-model-cache" in manifest
    assert "--ServerApp.root_dir=/workspace" in manifest
    assert "mountPath: /workspace/results" in manifest
    assert "restore_previous_state" in server
    assert "JUPYTER_TARGET" in server


def test_parameter_search_notebook_has_reproducible_screen_and_confirmation():
    source = Path("notebooks/03_srpg_parameter_search.ipynb").read_text(encoding="utf-8")

    assert "screening_trials()" in source
    assert "run_srpg_controlnet_img2img" in source
    assert "validator.validate" in source
    assert "aggregate_confirmation" in source
    assert "results.jsonl" in source
    assert "record_raw_baseline" in source
    assert "validations.json" in source
    assert "phone-validation.csv" in source
    assert "trial_rank_key" in source


def test_e007_notebook_is_scannability_first_and_context_adaptive():
    source = Path("notebooks/04_e007_contextual_optimizer.ipynb").read_text(encoding="utf-8")

    assert "require_exclusive_gpu" in source
    assert "factorial_contexts" in source
    assert "TPESampler" in source
    assert "sample_e007_trial" in source
    assert "clip_aesthetic" in source
    assert "clip_score" in source
    assert "ContextualParameterAdvisor" in source
    assert "select_delivery_candidate" in source
    assert "expected_cases=len(holdouts)" in source
    assert "aucune image 26/26" in source
    assert "TrialState.COMPLETE" in source
    assert "essais complets" in source
    assert "state.is_finished()" not in source
    assert "FanovaImportanceEvaluator(n_trees=32, max_depth=16, seed=20260722)" in source
    assert "Calcul fANOVA borné en cours" in source
    assert "Calibration :" in source
    assert "[calibration {position}/{calibration_total}] START" in source
    assert "Holdouts :" in source
    assert "[holdout {position}/{holdout_total}] START" in source
    assert "Calibration incomplète" in source


def test_e008_notebook_compares_controlnets_before_promotion():
    source = Path("notebooks/05_controlnet_model_bakeoff.ipynb").read_text(encoding="utf-8")

    assert "dion_sd15" in source
    assert "monster_sd15_v1" in source
    assert "monster_sd15_v2" in source
    assert "nacholmo_sd15_v2" in source
    assert "raw_mean_pass_rate" in source
    assert "aggregate_controlnet_benchmark" in source
    assert "best_trial_per_model" in source
    assert "select_promotable_controlnet" in source
    assert "model-load-errors.json" in source
    assert "physical-validation-template.csv" in source
    assert "CONTROL_SCALES = (0.90, 1.10, 1.35, 1.60)" in source


def test_nacholmo_live_notebook_uses_the_e008_winner_and_strict_gate():
    source = Path("notebooks/06_nacholmo_generate_live.ipynb").read_text(encoding="utf-8")

    assert "Nacholmo/controlnet-qr-pattern-v2" in source
    assert "BASE_CONTROLNET_SCALE = 1.60" in source
    assert "SRPG_CONTROLNET_SCALE = 1.60" in source
    assert "SRPG_STEPS = 100" in source
    assert "run_srpg_controlnet_img2img" in source
    assert "preview_callback=show_srpg_step" in source
    assert "summarize_validation_records" in source
    assert "CLIPQualityScorer" in source
    assert "06_DELIVERY.png" in source
    assert "06_BEST_OBSERVED_NOT_DELIVERABLE.png" in source
