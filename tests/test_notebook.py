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

    assert "FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" in dockerfile
    assert "DIFFQRCODER_COMMIT=e24ea73ee2e13c7e6e87cb422e8b11784e70ae00" in dockerfile
    assert "git clone https://github.com/jwliao1209/DiffQRCoder.git" in dockerfile
    assert "diffusers==0.32.2" in dockerfile
    assert "02_generate_live_on_gpu.ipynb" in launcher
    assert "03_srpg_parameter_search.ipynb" in launcher
    assert "04_e007_contextual_optimizer.ipynb" in launcher
    assert "05_controlnet_model_bakeoff.ipynb" in launcher
    assert "06_nacholmo_generate_live.ipynb" in launcher
    assert "07_diffqrcoder_official_live.ipynb" in launcher
    assert "08_diffqrcoder_vs_qrbtf_four_prompts.ipynb" in launcher
    assert "09_diffqrcoder_faithful_srmpgd.ipynb" in launcher
    assert "lpips==0.1.4" in dockerfile
    assert "LPIPS VGG cache OK" in dockerfile
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


def test_nacholmo_live_notebook_separates_artistic_text2img_from_srpg_img2img():
    source = Path("notebooks/06_nacholmo_generate_live.ipynb").read_text(encoding="utf-8")

    assert "Nacholmo/controlnet-qr-pattern-v2" in source
    assert "Nacholmo/Counterfeit-V2.5-vae-swapped" in source
    assert '\\"name\\": \\"art\\", \\"scale\\": 0.40, \\"control_end\\": 0.55' in source
    assert '\\"name\\": \\"balanced\\", \\"scale\\": 0.55, \\"control_end\\": 0.70' in source
    assert '\\"name\\": \\"structured\\", \\"scale\\": 0.75, \\"control_end\\": 0.85' in source
    assert "nacholmo_extremes_25" in source
    assert "control_guidance_end" in source
    assert 'controlnet_pipeline_mode=\\"text2img\\"' in source
    assert 'controlnet_pipeline_mode=\\"img2img\\"' in source
    assert "DPMSolverMultistepScheduler" in source
    assert "raw_backend._pipeline = None" in source
    assert "SRPG_CONTROLNET_SCALE = 1.60" in source
    assert "SRPG_STEPS = 100" in source
    assert "run_srpg_controlnet_img2img" in source
    assert "preview_callback=show_srpg_step" in source
    assert "summarize_validation_records" in source
    assert "CLIPQualityScorer" in source
    assert "06_DELIVERY.png" in source
    assert "06_BEST_OBSERVED_NOT_DELIVERABLE.png" in source


def test_diffqrcoder_official_notebook_is_paired_strict_and_auditable():
    source = Path("notebooks/07_diffqrcoder_official_live.ipynb").read_text(encoding="utf-8")

    assert "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00" in source
    assert "DiffQRCoderPipeline" in source
    assert "PerceptualLoss.forward = differentiable_perceptual_forward" in source
    assert "torch.stack(losses).mean()" in source
    assert "upstream-patches.json" in source
    assert "Cetus-Mix_Whalefall" in source
    assert "monster-labs/control_v1p_sd15_qrcode_monster" in source
    assert "CONTROLNET_SUBFOLDER = 'v2'" in source
    assert "qr.make(fit=False)" in source
    assert "QR_VERSION = 3" in source
    assert "QR_MASK_PATTERN = 4" in source
    assert "QR_MODULE_SIZE = 20" in source
    assert "STEPS = 40" in source
    assert "stage2_rng_state" in source
    assert "release_previous_gpu_objects" in source
    assert "output_history.clear()" in source
    assert "run_line_magic('reset_out'" not in source
    assert "Kernel > Restart Kernel" in source
    assert "memory_after_cleanup['free_driver'] < 15.0" in source
    assert "MEMORY_PROFILE = 'rtx_20gb'" in source
    assert "pipe.unet.requires_grad_(False).eval()" in source
    assert "pipe.unet.enable_gradient_checkpointing()" in source
    assert "pipe.controlnet.enable_gradient_checkpointing()" in source
    assert "pipe.enable_attention_slicing('max')" in source
    assert "pipe._run_stage1" in source
    assert "pipe._run_stage2" in source
    assert '"@torch.no_grad()\\n"' in source
    assert '"def run_stage2(profile):\\n"' in source
    assert "srpg_plus_srmpgd" in source
    assert "callback_on_step_end=callback" in source
    assert "validator.validate" in source
    assert "CLIPQualityScorer" in source
    assert "delivery_status = 'DELIVERABLE' if selected['strict_all']" in source
    assert "physical-validation.csv" in source
    assert "comparison-final.png" in source
    assert "kubectl cp -n qr-core" in source
    assert "${{POD}}" in source
    assert "scp paul@pcIA:~/" in source


def test_e011_compares_only_diffqrcoder_and_qrbtf_public_on_four_prompts():
    source = Path(
        "notebooks/08_diffqrcoder_vs_qrbtf_four_prompts.ipynb"
    ).read_text(encoding="utf-8")

    assert "len(PROMPTS) * 4" in source
    assert "p1_simple" in source
    assert "p2_medium" in source
    assert "p3_detailed" in source
    assert "p4_complex" in source
    assert "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00" in source
    assert "Cetus-Mix_Whalefall" in source
    assert "monster-labs/control_v1p_sd15_qrcode_monster" in source
    assert "latentcat/control_v1p_sd15_brightness" in source
    assert "QRBTF public reproduction, not proprietary QRBTF" in source
    assert "paper_stage2_latents" in source
    assert "qart_proxy" in source
    assert "apply_srmpgd" in source
    assert "{step:03d}.jpg" in source
    assert "make_gif" in source
    assert "software_ssr" in source
    assert "original_ssr" in source
    assert "clip_aesthetic" in source
    assert "clip_score" in source
    assert "physical-ssr.csv" in source
    assert "comparison-4x4.png" in source
    assert "kubectl cp -n qr-core" in source
    assert "RESUME_RUN_NAME" in source
    assert "def completed(" in source
    assert "def reset_incomplete(" in source


def test_e012_uses_exact_stage2_latent_original_qr_and_paper_srmpgd_objective():
    source = Path("notebooks/09_diffqrcoder_faithful_srmpgd.ipynb").read_text(
        encoding="utf-8"
    )

    assert "len(PROMPTS) * len(STAGE2_PROFILES) * 2" in source
    assert "'name': 'paper40', 'steps': 40" in source
    assert "'name': 'observed100', 'steps': 100" in source
    assert "original_control_records" in source
    assert "QR témoin illisible sans dégradation" in source
    assert "00_qr_control_validations.json" in source
    assert "'control_validation': control_validation" in source
    assert "all(item.exact_payload_match for item in control_records)" not in source
    assert "DiffQRCoderPipeline" in source
    assert "Cetus-Mix_Whalefall" in source
    assert "monster-labs/control_v1p_sd15_qrcode_monster" in source
    assert "ScanningRobustLoss(module_size=QR_MODULE_SIZE)" in source
    assert "SRMPGDConfig(" in source
    assert "step_size=1000.0" in source
    assert "lpips_weight=0.01" in source
    assert "lpips.LPIPS(net='vgg'" in source
    assert "output_type='latent'" in source
    assert "stage2-final-latent.safetensors" in source
    assert "scanning_loss=official_srmpgd_srl" in source
    assert "pipe, base_latent, blueprint, SRMPGD_CONFIG" in source
    assert "base_latent.float()" in source
    assert "'srmpgd_target': 'original binary QR y, not QArt proxy'" in source
    assert "'srmpgd_initialization': 'exact clean Stage 2 latent, never PNG re-encoding'" in source
    assert "decoded_latent_state_mer" in source
    assert "validation_by_iteration" in source
    assert "srmpgd-frames" in source
    assert "make_gif" in source
    assert "module_error_rate" in source
    assert "software_ssr" in source
    assert "original_ssr" in source
    assert "clip_aesthetic" in source
    assert "clip_score" in source
    assert "worst_decoder_pass_rate" in source
    assert "worst_scenario_pass_rate" in source
    assert "image_change_metrics(image, reference_image)" in source
    assert "comparison-4x4.png" in source
    assert "physical-validation.csv" in source
    assert "manifest.json" in source
    assert "def drop_result_keys(" in source
    assert "def trace_artifacts_complete(" in source
    assert "stage1-time.json" in source
    assert "stage2-time.json" in source
    assert "force_profile_regeneration" in source
    assert "e012-diffqrcoder-public-binary-srmpgd-v2" in source
    assert "STAGE2_CONDITION_IMPLEMENTATION" in source
    assert "target = qr_image.copy()" in source
    assert "stage2-binary-qr-condition.png" in source
    assert "exact Reed-Solomon QArt transform unavailable" in source
    assert "def qart_proxy(" not in source
    assert "matrix-preserving visual proxy" not in source
    assert "Le proxy QArt de" not in source
    assert "latent_from_image" not in source
    assert "SRMPGD_LR = 0.1" not in source


def test_e013_compares_exact_geometry_sd15_sd21_and_builds_policy_dataset():
    source = Path("notebooks/10_exact_geometry_sd15_sd21_policy.ipynb").read_text(
        encoding="utf-8"
    )

    assert "e013-exact-geometry-sd15-sd21-policy-v1" in source
    assert "generate_aligned_qr" in source
    assert "'canvas': 744" in source
    assert "'canvas': 768" in source
    assert "'module_size': 16" in source
    assert "'module_size': 20" in source
    assert "padding=(canvas - core_modules*module_size)/2; no QR resize" in source
    assert "aligned_module_diagnostics" in source
    assert "assert original_passed == len(originals)" in source
    assert "DiffQRCoderPipeline" in source
    assert "monster-labs/control_v1p_sd15_qrcode_monster" in source
    assert "DionTimmer/controlnet_qrcode-control_v11p_sd21" in source
    assert "StableDiffusionControlNetImg2ImgPipeline" in source
    assert "UPSTREAM_SRMPGD_LR = 0.1" in source
    assert "step_size=1000.0" in source
    assert "lpips_weight=0.01" in source
    assert "output_type='latent'" in source
    assert "srmpgd_num_iteration=UPSTREAM_SRMPGD_ITERATIONS" in source
    assert "TPESampler(" in source
    assert "paper_step_size" in source
    assert "negative_profile" in source
    assert "control_start" in source
    assert "constraints_func=constraints_func" in source
    assert "directions=['maximize', 'maximize', 'maximize', 'minimize']" in source
    assert "policy-dataset.csv" in source
    assert "CatBoostClassifier" in source
    assert "GroupKFold" in source
    assert "POLICY_MIN_ROWS = 100" in source
    assert "deliverable_candidate" in source
    assert "DELIVERY_TARGET = 0.999" in source
    assert "physical-validation.csv" in source
    assert "{step_index:03d}.jpg" in source
    assert "RESUME_RUN_NAME" in source
