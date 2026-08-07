import ast
import json
import tomllib
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
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
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
    assert "zxing-cpp>=3.0,<4" in project["project"]["optional-dependencies"]["notebook"]
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
    assert "[switch]$Reset" in launcher
    assert "bash scripts/notebook-server.sh $remoteAction $Notebook" in launcher
    assert "verify_running_notebook" in server
    assert 'expected_notebook_path="/workspace/notebooks/${expected_notebook}"' in server
    assert 'prooftag.io/notebook-mode":"advisor-cpu"' in server
    assert "prepare_runtime" in server
    assert 'kubectl scale "deployment/${api_deployment}"' in server
    assert "reset)" in server


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
    assert "trial.state == optuna.trial.TrialState.COMPLETE" in source
    assert "catch=(FloatingPointError,)" in source
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


def test_e014a_uses_real_qart_and_keeps_exact_payload_claims_separate():
    source = Path("notebooks/11_e014a_qart_blueprint_bakeoff.ipynb").read_text(
        encoding="utf-8"
    )
    dockerfile = Path("Dockerfile.notebook").read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "6e0e00804a1994db7098432c19fadfc552071e30" in source
    assert "/usr/local/bin/qart" in source
    assert "canonical_url_match" in source
    assert "exact_payload_mask_search_m" in source
    assert "adaptive_exact_payload_m" in source
    assert "e014a-deterministic-blueprint-pairing-v2" in source
    assert "SEED_OFFSET = 30000" in source
    assert "Seeds effectives" in source
    assert "seed_everything" in source
    assert "torch.cuda.manual_seed_all(seed)" in source
    assert "CUBLAS_WORKSPACE_CONFIG" in source
    assert "torch.use_deterministic_algorithms(True, warn_only=True)" in source
    assert "binary_mask4_m_duplicate" in source
    assert "'selection_eligible': False" in source
    assert "initial_latent_sha256" in source
    assert "final_latent_sha256" in source
    assert "final_image_sha256" in source
    assert "determinism-audit.json" in source
    assert "fully_reproducible" in source
    assert "RUNTIME_VERSIONS" in source
    assert "all_mask_costs" in source
    assert "QART_REPEATS = 3" in source
    assert "qart-screening.json" in source
    assert "f'{step_index:03d}.jpg'" in source
    assert "f'qart-raw-threshold-{threshold}-repeat-{repeat}.png'" in source
    assert "f'{{step_index:03d}}.jpg'" not in source
    assert "f'qart-raw-threshold-{{threshold}}-repeat-{{repeat}}.png'" not in source
    assert "load_saved_target" in source
    assert "stage1-control-preflight.json" in source
    assert "paired_stage2_latents" in source
    assert "selected-blueprint.png" in source
    assert "selected-matrix.npy" in source
    assert "exact payload only" in source
    assert "FROM rust:1.85-slim-bookworm AS qart-builder" in dockerfile
    assert "zxing-cpp" in dockerfile
    assert "11_e014a_qart_blueprint_bakeoff.ipynb" in launcher


def test_e014b_factorizes_channel_timestep_alpha_and_decoder_gradient_audit():
    source = Path("notebooks/12_e014b_freeqr_latent_fusion.ipynb").read_text(
        encoding="utf-8"
    )

    assert "target_timestep_after_step" in source
    assert "for channel in range(4)" in source
    assert "WINDOWS" in source
    assert "ALPHAS = [0.05, 0.10, 0.15, 0.22]" in source
    assert "blueprint_latent" in source
    assert "differentiable_module_loss" in source
    assert "baseline_no_fusion" in source
    assert "FreeQR-inspired channel/timestep reconstruction" in source
    assert "free_gib < 18.0" in source
    assert "release_stage2_guidance" in source
    assert "pipe.unet.enable_gradient_checkpointing()" in source
    assert "*-e014a-deterministic-blueprint-pairing-v2" in source


def test_e015_is_an_aesthetic_reference_comparison_not_a_qr_model_claim():
    source = Path("notebooks/13_e015_aesthetic_backbone_reference.ipynb").read_text(
        encoding="utf-8"
    )
    dockerfile = Path("Dockerfile.notebook").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "sd15_cetus" in source
    assert "stabilityai/stable-diffusion-xl-base-1.0" in source
    assert "black-forest-labs/FLUX.1-schnell" in source
    assert "enable_model_cpu_offload" in source
    assert "enable_sequential_cpu_offload" in source
    assert "pipeline.vae.enable_slicing()" in source
    assert "pipeline.vae.enable_tiling()" in source
    assert "'offload_mode': 'sequential_cpu'" in source
    assert "resolved-model-revisions.json" in source
    assert "model_info(spec['repo'], token=HF_TOKEN).sha" in source
    assert "check_model_access" in source
    assert "model-access.json" in source
    assert "REQUIRE_ALL_MODELS = True" in source
    assert "device='cpu' if spec['kind'] == 'flux' else 'cuda'" in source
    assert "import sentencepiece" in source
    assert "from google.protobuf import __version__ as protobuf_version" in source
    assert '"sentencepiece==0.2.0"' in pyproject
    assert '"protobuf==5.29.3"' in pyproject
    assert "sentencepiece.__version__" in dockerfile
    assert "protobuf_version" in dockerfile
    assert "CLIPQualityScorer" in source
    assert "selection_scope" in source
    assert "no claim about QR diffusion compatibility" in source


def test_e016_labels_with_real_decoders_prevents_leakage_and_audits_gradients():
    source = Path("notebooks/14_e016_differentiable_scan_surrogate.ipynb").read_text(
        encoding="utf-8"
    )
    deployment = Path("deploy/k8s/notebook.yaml").read_text(encoding="utf-8")

    assert "DEFAULT_SCENARIOS" in source
    assert "*-e014a-deterministic-blueprint-pairing-v2" in source
    assert "decode_safely" in source
    assert "error_{decoder.name}" in source
    assert "len(decoder_names) < 3" in source
    assert "source_id" in source
    assert "source_relative_path" in source
    assert "non_exact_payload_contract" in source
    assert "excluded-sources.json" in source
    assert "Collision de dataset interdite" in source
    assert "MIN_GROUP_CLASS_COUNT_PER_DECODER = 3" in source
    assert "select_group_partitions" in source
    assert "GroupShuffleSplit" not in source
    assert "isdisjoint" in source
    assert "BCEWithLogitsLoss" in source
    assert "average_precision_score" in source
    assert "brier_score_loss" in source
    assert "calibration_curve" in source
    assert "real_decoder_improved" in source
    assert "physical-captures-template.csv" in source
    assert "production_usable" in source
    assert "scan-surrogate.research-only.torchscript.pt" in source
    assert "REJECTED-SURROGATE.json" in source
    assert "scan-surrogate.torchscript.pt" not in source
    assert "shutil.disk_usage('/dev/shm')" in source
    assert "DATALOADER_WORKERS" in source
    assert "'prefetch_factor': 1" in source
    assert "'persistent_workers': True" in source
    assert "num_workers=2" not in source
    assert "mountPath: /dev/shm" in deployment
    assert "sizeLimit: 2Gi" in deployment


def test_e014c_isolates_stage2_nondeterminism_before_more_campaigns():
    source = Path(
        "notebooks/15_e014c_stage2_determinism_diagnostic.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "e014c-stage2-divergence-ablation-v3" in source
    assert "*-e014a-deterministic-blueprint-pairing-v2" in source
    assert "REPRO_STEPS = 40" in source
    assert "STOP_AFTER_STEP = 7" in source
    assert "torch.use_deterministic_algorithms(True, warn_only=False)" in source
    assert "ZeroGuidance" in source
    assert "torch.autograd.grad(image.sum() * 0.0, latents)" in source
    assert "ScanningOnlyGuidance" in source
    assert "PerceptualOnlyGuidance" in source
    assert "zero_connected_gc_on_40_stop7" in source
    assert "srl_only_gc_on_40_stop7" in source
    assert "lpips_only_gc_on_40_stop7" in source
    assert "combined_gc_on_40_stop7" in source
    assert "_pipeline._interrupt = True" in source
    assert "OutOfMemoryError" in source
    assert "initial_latent_sha256" in source
    assert "latent_sha256" in source
    assert "determinism-isolation.json" in source
    assert "error.txt" in source
    assert "15_e014c_stage2_determinism_diagnostic.ipynb" in launcher


def test_e014b_v2_uses_balanced_repetitions_and_statistical_promotion():
    source = Path(
        "notebooks/16_e014b_statistical_freeqr_confirmation.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "e014b-statistical-freeqr-confirmation-v2" in source
    assert "PROMPT_ID = 'p3_detailed'" in source
    assert "REPEATS = 4" in source
    assert "LATIN_ORDERS" in source
    assert "WILLIAMS_FIRST_ROW = [0, 1, 3, 2]" in source
    assert "len(transitions) == len(set(transitions)) == 12" in source
    assert "baseline" in source
    assert "fusion_all" in source
    assert "fusion_early" in source
    assert "fusion_gradient" in source
    assert "fresh_pipeline_per_repeat" in source
    assert "Répétition {repeat} partielle" in source
    assert "paired_differences" in source
    assert "bootstrap_mean_ci" in source
    assert "exceeds_baseline_span" in source
    assert "positive_repeats'] >= 3" in source
    assert "aesthetic_drop'] <= 0.5" in source
    assert "DECISION.json" in source
    assert "39/39" in source
    assert "16_e014b_statistical_freeqr_confirmation.ipynb" in launcher


def test_e014b_v3_confirms_generalization_with_corrected_gates():
    source = Path(
        "notebooks/17_e014b_multicontext_generalization.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "e014b-multicontext-generalization-v3" in source
    assert "CONTEXT_IDS = ['p1_simple', 'p2_medium', 'p4_complex']" in source
    assert "REPEATS = 4" in source
    assert "['baseline', 'fusion_all']" in source
    assert "['fusion_all', 'baseline']" in source
    assert "'channel': 1, 'alpha': 0.15" in source
    assert "fresh_pipeline_per_block" in source
    assert "Bloc {context_id}/{repeat} partiel" in source
    assert "paired_within_every_block" in source
    assert "initial-latent-audit.json" in source
    assert "Cross-block variability" in source
    assert "original_3of3_required_in_every_repeat" in source
    assert "(fusion.original_passed == fusion.original_total).all()" in source
    assert "MIN_MEAN_GAIN = 3 / 39" in source
    assert "production_candidate" in source
    assert "generalized_not_strict" in source
    assert "decoder-results.csv" in source
    assert "scenario-results.csv" in source
    assert "resolved-model-revisions.json" in source
    assert "hf_hub_download" in source
    assert "snapshot_download" in source
    assert "CONFIG_MODEL_REPO" in source
    assert "revision=resolved_revisions['base_model']" in source
    assert "revision=resolved_revisions['config_model']" in source
    assert "config=BASE_CONFIG_PATH" in source
    assert "diffqrcoder_commit" in source
    assert "pipeline_source_sha256" in source
    assert "physical-validation-template.csv" in source
    assert "pipeline.to('cpu')" not in source
    assert "17_e014b_multicontext_generalization.ipynb" in launcher


def test_e014d_uses_late_functional_rediffusion_without_pixel_compositing():
    source = Path(
        "notebooks/18_e014d_functional_late_rediffusion.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "CONTEXT_IDS = ['p1_simple', 'p2_medium', 'p3_detailed', 'p4_complex']" in source
    assert "RESCUE_STEPS = 8" in source
    assert "GLOBAL_FUSION_CHANNEL = 1" in source
    assert "GLOBAL_FUSION_ALPHA = 0.15" in source
    assert "'structural_strength': 0.15" in source
    assert "'structural_strength': 0.30" in source
    assert "'structural_strength': 0.45" in source
    assert "functional_pattern_mask" in source
    assert "quiet zone plus finder/separator" in source
    assert "class LateWindowDDIMScheduler(DDIMScheduler):" in source
    assert "self.num_inference_steps != BASE_STEPS" in source
    assert "LateWindowDDIMScheduler.from_config" in source
    assert "timesteps=[int(value)" in source
    assert "num_inference_steps=None" in source
    assert "original_probe" in source
    assert "fixed_recipe_gate" in source
    assert "fresh_pipeline_per_rescue_candidate" in source
    assert "Nettoyage du candidat partiel sans image finale" in source
    assert "paired-input-audit.json" in source
    assert "rescue_noise_sha256" in source
    assert "def source_original_all(row):" in source
    assert "row.get('original_passed')" in source
    assert "winner = max(candidates, key=source_row_rank)" in source
    assert "'post_diffusion_pixel_projection': False" in source
    assert "context-adaptive-oracle.csv" in source
    assert "fixed-recipe-aggregates.csv" in source
    assert "physical-validation-template.csv" in source
    assert "18_e014d_functional_late_rediffusion.ipynb" in launcher


def test_e014e_separates_mechanisms_before_testing_late_windows():
    source = Path(
        "notebooks/19_e014e_mechanism_window_ablation.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")

    assert "e014e-mechanism-window-ablation-v1" in source
    assert "SCREENING_CONTEXT_IDS = ['p2_medium', 'p3_detailed']" in source
    assert "HOLDOUT_CONTEXT_IDS = ['p1_simple', 'p4_complex']" in source
    assert "PHASE_A_STEPS = 4" in source
    assert "PHASE_B_STEP_COUNTS = [2, 4, 6, 8]" in source
    assert "'rediffusion_only'" in source
    assert "'global_a03'" in source
    assert "'mask_s05'" in source
    assert "'combined_a03_s05'" in source
    assert "REFERENCE_RECIPE_ID = 'combined_a15_s15'" in source
    assert "reference_forced" in source
    assert "LateWindowDDIMScheduler" in source
    assert "late_schedule(pipeline, rescue_steps)" in source
    assert "fresh_pipeline_per_candidate" in source
    assert "paired-input-audit.json" in source
    assert "Nettoyage du candidat non indexé" in source
    assert "ERRORS_PATH" in source
    assert "'post_diffusion_pixel_projection': False" in source
    assert "'selector_training_allowed': False" in source
    assert "phase-a-aggregates.csv" in source
    assert "phase-b-fixed-aggregates.csv" in source
    assert "physical-validation-template.csv" in source
    assert "19_e014e_mechanism_window_ablation.ipynb" in launcher


def test_e014f_uses_unseen_contexts_and_a_strict_delivery_cascade():
    source = Path(
        "notebooks/20_e014f_unseen_generalization_cascade.ipynb"
    ).read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")
    deployer = Path("scripts/deploy-notebook-image.sh").read_text(
        encoding="utf-8"
    )

    assert "e014f-unseen-generalization-cascade-v1" in source
    assert "RESCUE_STEP_COUNTS = [2, 3, 4]" in source
    assert "CANVAS_SIZE = 768" in source
    assert "CANVAS_SIZE - (29 * QR_MODULE_SIZE) >= 8 * QR_MODULE_SIZE" in source
    assert "'combined_a06_s10'" in source
    assert "'combined_a10_s15'" in source
    assert "'combined_a15_s15'" in source
    assert "len(contexts_spec) == 24" in source
    assert "== 16" in source
    assert "== 8" in source
    assert "adaptive_exact_payload" in source
    assert "SOURCE_FUSION_ALPHA = 0.15" in source
    assert "fresh_pipeline_per_candidate" in source
    assert "paired-input-audit.json" in source
    assert "CASCADE_ORDER" in source
    assert "exact payload, 39/39 software validations" in source
    assert "'selector_training_allowed': False" in source
    assert "'post_diffusion_pixel_projection': False" in source
    assert "physical-validation-template.csv" in source
    assert "20_e014f_unseen_generalization_cascade.ipynb" in launcher
    assert "expected_notebook" in deployer


def test_e026_notebook_uses_qr_verify_as_a_calibrated_first_objective():
    source = Path("notebooks/21_e026_prompt_parameter_advisor.ipynb").read_text(
        encoding="utf-8"
    )
    advisor = Path("prooftag_qr/parameter_advisor.py").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile.notebook").read_text(encoding="utf-8")
    launcher = Path("scripts/notebook-remote.ps1").read_text(encoding="utf-8")
    deployer = Path("scripts/deploy-notebook-image.sh").read_text(encoding="utf-8")
    stack_deployer = Path("scripts/deploy-e026-notebook.sh").read_text(encoding="utf-8")

    assert "E026ParameterAdvisor" in source
    assert "quality_qr_verify_any_exact" in advisor
    assert "MINIMUM_PROMPT_GROUPS = 12" in source
    assert "SCAN_PROBABILITY_THRESHOLD = 0.80" in source
    assert "GroupKFold by SHA-256(prompt text)" in advisor
    assert "WeekCampaignRunner" in source
    assert "RUN_COLLECTION = True" in source
    assert "COLLECTION_PROMPT_COUNT = 300" in source
    assert "progress_callback=collection_progress" in source
    assert "notebook-progress.jsonl" in source
    assert "les lots terminés sont ignorés" in source
    assert "/data/e026-week/*/exports/*.csv" in source
    assert "3 exploitation + 3 maximum uncertainty" in source
    assert "prooftag-e026-parameter-advisor.joblib" in source
    assert "21_e026_prompt_parameter_advisor.ipynb" in launcher
    assert "21_e026_prompt_parameter_advisor.ipynb" in deployer
    assert '--build-arg "EXPECTED_NOTEBOOK=${expected_notebook}"' in deployer
    assert "docker run" not in deployer
    assert "ARG EXPECTED_NOTEBOOK=" in dockerfile
    assert 'test -f "/workspace/${EXPECTED_NOTEBOOK}"' in dockerfile
    assert "bash scripts/deploy-app-image.sh" in stack_deployer
    assert "bash scripts/deploy-notebook-image.sh" in stack_deployer
    assert 'bash scripts/notebook-server.sh reset "$notebook"' in stack_deployer
