import ast
import json
import re
from pathlib import Path

NOTEBOOK = Path("notebooks/29_e034_srmpgd_four_iteration_gate.ipynb")
BUILDER = Path("scripts/build_e034_srmpgd_four_iteration_gate_notebook.py")


def _notebook_source(*, code_only: bool = False) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if not code_only or cell["cell_type"] == "code"
    )


def test_e034_is_one_exact_four_iteration_campaign_without_expansion():
    source = _notebook_source()

    assert "e034-srmpgd-four-iteration-gate-v1" in source
    assert (
        "COLLECTION_PAYLOAD = 'https://ptag.io/t/e033'"
    ) in source
    assert "PROOFTAG_E034_PAYLOAD" not in source
    assert "12834cad09eb0680af5a71c0f8c20627fba9746c117e0da3e1c6a14f18952475" in source
    assert "e033_simple_greenhouse" in source
    assert "SEED = 51_001" in source
    assert "MILESTONE_ITERATIONS = [0, 1, 2, 4]" in source
    assert "AUTOMATIC_EXPANSION_AUTHORIZED = False" in source
    assert "len(plan.campaigns) == 1" in source
    assert "plan.public['trial_count'] == 4" in source
    assert "maximum_campaign_attempts=1" in source
    assert "reject_campaigns_with_errors=True" in source
    assert "stop_on_first_failed_campaign=True" in source

    for profile_id in (
        "diffqrcoder_stage1",
        "e033_public_demo_srpg",
        "e034_equation_srmpgd_fp16",
        "e034_equation_srmpgd_fp32",
    ):
        assert profile_id in source
    assert "diffqrcoder_paper_srpg" not in source
    assert "srmpgd_max_iterations'] == 4" in source
    assert "srmpgd_step_size'] == 1000.0" in source
    assert "srmpgd_gradient_scale'] == 32768.0" in source
    assert "srmpgd_lpips_weight'] == 0.01" in source
    assert "srmpgd_lpips_device'] == 'cpu'" in source
    assert "srmpgd_decode_precision'] == 'model'" in source
    assert "srmpgd_decode_precision'] == 'float32'" in source
    assert "'prediction_contract': prediction_contract" in source
    assert "'local_qr_verify_preflight': local_qr_verify_binding" in source
    assert "preflight_score.conservative_exact_presets == 37" in source
    assert "preflight_scorer.close()" in source


def test_e034_freezes_e033_parent_and_proves_every_direct_raster():
    source = _notebook_source()

    for expected_hash in (
        "02e0bea8e5c539cda599f6158a3a07bf1a9eed3db2f0e468d58b56f430458d53",
        "8cb36a623aa999567f51402615cceb8917505f3ba7e9c06c34e6bbef045e9721",
        "6bd10526053cb9af9a80b123b29c66919e60523f6703a9f7d4cf10a5506e2146",
    ):
        assert expected_hash in source
    assert "row.stage1_source_run_id == stage1.generation_run_id" in source
    assert "row.stage2_source_run_id == parent.generation_run_id" in source
    assert "row.stage2_source_latent_sha256 == row.stage2_latent_sha256" in source
    assert "row.srmpgd_stage2_image_sha256 == parent.final_image_sha256" in source
    assert "finite(row.srmpgd_iteration_zero_exact) == 1.0" in source
    assert "variant = f'srmpgd_iteration_{iteration:03d}'" in source
    assert "redecode_variant = 'srmpgd_redecoded_iteration_000'" in source
    assert "trace_step.get('image_sha256')" in source
    assert "trace_step.get('next_step_rms')" in source
    assert "trace_step.get('step_scale')" in source
    assert "DIRECT_MILESTONE_HASHES_VERIFIED" in source
    assert "milestones.image_raster_sha256 == milestones.trace_image_sha256" in source
    assert "iteration_zero_hashes == {parent_raster_sha256}" in source
    assert "FINAL_RASTER_IS_I4" in source
    assert "artifact_catalog =" not in source


def test_e034_rescores_and_displays_parent_vae_i0_i1_i2_i4():
    source = _notebook_source()

    assert "ConservativeQRVerifyScorer" in source
    assert "canonical_conservative_qr_verify_evidence(qr_score)" in source
    assert "qr_score.to_dict()" not in source
    assert "repetitions=3" in source
    assert "diffqrcoder_module_error_rate(" in source
    assert "image_quality_metrics(image)" in source
    assert "image_change_metrics(image, parent_image)" in source
    assert "e034-local-raster-scores.csv" in source
    assert "e034-local-qr-verify-details.jsonl" in source
    assert "len(local_scores) == 11" in source
    assert "milestone_column_count = 2 + len(MILESTONE_ITERATIONS)" in source
    assert "QRV=" in source
    assert "MER=" in source
    assert "e034-final-contact-sheet.png" in source
    assert "e034-milestone-contact-sheet.png" in source


def test_e034_separates_mechanism_scan_visual_and_production_verdicts():
    source = _notebook_source()

    for marker in (
        "TRACE_COMPLETE",
        "TRACE_PROTOCOL_EXACT",
        "UPDATE_GRADIENTS_VALID",
        "LPIPS_GRADIENTS_VALID",
        "LPIPS_INITIAL_ZERO",
        "LPIPS_REFERENCE_PAPER",
        "LATENT_DELTAS_VALID",
        "LPIPS_WEIGHTING_CONSISTENT",
        "PAPER_STEPS_UNCLIPPED_AND_CONSISTENT",
        "OBJECTIVES_CONSISTENT",
        "OBJECTIVE_FINAL_LOWER",
        "SRL_FINAL_LOWER",
        "FP16_FP32_CONSISTENT",
        "SCAN_PROGRESS_PASS",
        "SINGLE_CASE_QR_VERIFY_PASS",
        "VISUAL_PROXY_PASS",
        "'production_ready': False",
        "'automatic_expansion_authorized': False",
    ):
        assert marker in source
    assert "positive_trace_value(iteration, field)" in source
    assert "for iteration in [1, 2, 3]" in source
    assert "paper_stage2_float" in source
    assert "lpips_nulle_a_i0" in source
    assert "gradient_lpips_actif_i1_a_i3" in source
    assert "gradient_lpips_actif_i0_a_i3" not in source
    assert "math.isclose(applied, requested" in source
    assert "math.isclose(scale, 1.0" in source
    assert "FP16_FP32_IDENTICAL" in source
    assert "allow_nan=False" in source
    assert "objective_monotonic_steps_out_of_four" in source
    assert "manual_visual_review_required" in source

    final_sheet = source.index("e034-final-contact-sheet.png")
    milestone_sheet = source.index("e034-milestone-contact-sheet.png")
    gates = source.index("e034-primary-fp32-gates.csv")
    archive = source.index("e034-artifact-manifest.json")
    final_reading = source.index("**PASS local sous réserve de la revue humaine.**")
    assert final_sheet < milestone_sheet < gates < archive < final_reading
    scientific_stop = source[
        source.index("if not READY_FOR_MANUAL_REVIEW") : archive
    ]
    assert "raise RuntimeError" not in scientific_stop


def test_e034_technical_failure_is_archived_and_never_retried():
    source = _notebook_source()

    assert "E034_TECHNICAL_STOP = runner_summary['status'] != 'completed'" in source
    assert "technical-failure.json" in source
    assert "attempt-history.csv" in source
    assert "export-diagnostics.csv" in source
    assert "failed-trials.csv" in source
    assert "remote-campaign-diagnostics.csv" in source
    assert "state-read-error.txt" in source
    assert "control-copy-diagnostics.csv" in source
    assert "'status': 'unreadable_state'" in source
    assert "prepare_incident_bundle" in source
    assert "resolve_incident_archive" in source
    assert "incident_identity_sha256" in source
    assert "api-runtime.json" in source
    assert "lab-schema.json" in source
    assert "inspect_archive_without_regenerating" in source
    assert "-technical-failure.tar.gz" in source
    assert "STOP technique E034 — aucune nouvelle génération" in source
    assert "Ne relancez pas la campagne." in source
    assert "maximum_campaign_attempts=2" not in source

    execution = source[source.index("RUNNER_EXCEPTION = None") : source.index("## 5.")]
    assert "except Exception as exc:" in execution
    assert "'status': 'runner_exception'" in execution
    assert "archive_post_gpu_failure" in execution
    assert "snapshot_tree_once(" in execution
    assert "excluded_top_level={'post-gpu-failure'}" in execution
    assert "kind='technical_failure'" in execution
    assert "kind='post_gpu_failure'" in execution
    assert "bundle.add(runner.output_dir" not in execution
    assert "E034_POST_GPU_STOP = False" in execution
    scientific = source[source.index("## 5.") :]
    assert scientific.count("if E034_TECHNICAL_STOP:") == 8
    assert scientific.count("Cellule ignorée après le STOP technique") == 8
    assert scientific.count("elif E034_POST_GPU_STOP:") == 8
    assert scientific.count("STOP post-GPU archivé") == 8


def test_e034_archive_is_atomic_verified_and_names_are_not_stale_e033_outputs():
    source = _notebook_source()

    assert "os.replace(temporary_archive, archive_path)" in source
    assert "with tarfile.open(archive_path, 'r:gz')" in source
    assert "if not archive_path.is_file():" in source
    assert "vérification sans réécriture ni génération GPU" in source
    assert "actual_sha256 != expected_sha256" in source
    assert "analysis_identity_sha256" in source
    assert "Archive E034 existante mais obsolète" in source
    assert "atomic_copy_once(source, control_dir / source.name)" in source
    assert "atomic_json_once(control_dir / 'api-runtime.json', api_runtime)" in source
    assert "atomic_json_once(control_dir / 'lab-schema.json', schema)" in source
    assert "Archive créée même en cas de STOP" in source
    assert "e034-artifact-manifest.json" in source
    forbidden = re.findall(r"['\"](e033-[^'\"]+)['\"]", source)
    assert forbidden == []


def test_e034_generated_code_is_syntax_valid_and_builder_is_idempotent():
    ast.parse(_notebook_source(code_only=True))

    before = NOTEBOOK.read_bytes()
    builder_source = BUILDER.read_text(encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(BUILDER)}
    exec(compile(builder_source, str(BUILDER), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before
