import ast
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/28_e033_srmpgd_microdiagnostic.ipynb")
BUILDER = Path("scripts/build_e033_srmpgd_microdiagnostic_notebook.py")


def _notebook_source(*, code_only: bool = False) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if not code_only or cell["cell_type"] == "code"
    )


def test_e033_notebook_is_one_resumable_paired_microdiagnostic():
    source = _notebook_source()

    assert "e033-srmpgd-microdiagnostic-v1" in source
    assert "e033_simple_greenhouse" in source
    assert "SEED = 51_001" in source
    assert "MILESTONE_ITERATIONS = [0, 1]" in source
    assert "AUTOMATIC_EXPANSION_AUTHORIZED = False" in source
    assert "len(plan.campaigns) == 1" in source
    assert "plan.public['campaign_count'] == 1" in source
    assert "plan.public['trial_count'] == 5" in source
    assert "maximum_campaign_attempts=1" in source
    assert "reject_campaigns_with_errors=True" in source
    assert "stop_on_first_failed_campaign=True" in source

    for profile_id in (
        "diffqrcoder_stage1",
        "diffqrcoder_paper_srpg",
        "e033_public_demo_srpg",
        "e033_equation_srmpgd_fp16",
        "e033_equation_srmpgd_fp32",
    ):
        assert profile_id in source

    assert "diffqrcoder_stage2_initialization'] == 'public_random'" in source
    assert "diffqrcoder_stage2_target_mode'] == 'binary_exact'" in source
    assert "srpg_controlnet_scale'] == 1.05" in source
    assert "srpg_qr_weight'] == 50.0" in source
    assert "srpg_perceptual_weight'] == 20.0" in source
    assert "srmpgd_decode_precision'] == 'model'" in source
    assert "srmpgd_decode_precision'] == 'float32'" in source
    assert "srmpgd_gradient_scale'] == 32768.0" in source
    assert "srmpgd_max_iterations'] == 1" in source
    assert "srmpgd_lpips_device'] == 'cpu'" in source
    assert "candidate_signature != public_stage2_signature" in source


def test_e033_notebook_proves_pairing_and_downloads_milestones_directly():
    source = _notebook_source()

    assert "row.stage1_source_run_id == stage1.generation_run_id" in source
    assert "row.stage1_image_sha256 == stage1.final_image_sha256" in source
    assert "row.stage2_source_run_id == parent.generation_run_id" in source
    assert "row.stage2_source_method_id == parent.method_id" in source
    assert "row.stage2_source_latent_sha256 == row.stage2_latent_sha256" in source
    assert "row.srmpgd_stage2_image_sha256 == parent.final_image_sha256" in source
    assert "finite(row.srmpgd_iteration_zero_exact) == 1.0" in source
    assert "e033-pairing-audit.csv" in source

    assert "variant = f'srmpgd_iteration_{iteration:03d}'" in source
    assert "redecode_variant = 'srmpgd_redecoded_iteration_000'" in source
    assert "endpoint = f'/v1/generations/{run_id}/variants/{variant}'" in source
    assert "download_direct_png(" in source
    assert "artifact_catalog =" not in source
    assert "api_json(f'/v1/generations/{run_id}/artifacts')" not in source
    assert "iteration_zero_hashes == {parent_raster_sha256}" in source
    assert "REDECODE_CONTROLS_VERIFIED" in source
    assert "except HTTPError as exc:" in source
    assert "if exc.code != 404:" in source
    assert "'available': image_path is not None" in source
    assert "arrêt anticipé / 404" in source
    assert "PRIMARY_MILESTONES_AVAILABLE" in source
    assert "primary_steps.get(1, {}).get('latent_delta_rms')" in source
    assert "len(milestones) != len(MILESTONE_METHOD_IDS) * len(MILESTONE_ITERATIONS)" in source
    assert "milestone_column_count = 2 + len(MILESTONE_ITERATIONS)" in source


def test_e033_contact_sheets_precede_fp32_verdict_and_archive_is_unconditional():
    source = _notebook_source()

    final_sheet = source.index("e033-final-contact-sheet.png")
    milestone_sheet = source.index("e033-milestone-contact-sheet.png")
    gates = source.index("e033-primary-fp32-gates.csv")
    archive = source.index("e033-artifact-manifest.json")
    final_reading = source.index("if PRIMARY_FP32_GATES_PASSED:")
    assert final_sheet < milestone_sheet < gates < archive < final_reading

    assert "gradient_0 is not None and gradient_0 > 0" in source
    assert "image_gradient_0 is not None and image_gradient_0 > 0" in source
    assert "applied_step_0 is not None and applied_step_0 > 0" in source
    assert "latent_delta_1 is not None and latent_delta_1 > 0" in source
    assert "srl_1 < srl_0" in source
    assert "manual_review_then_design_four_iteration_gate" in source
    assert "primary_fp32_gates_passed" in source
    assert "stop_and_fix_numerics_without_expanding" in source
    assert "Archive créée même en cas de STOP" in source
    assert "if not PRIMARY_FP32_GATES_PASSED:" in source
    scientific_stop = source[source.index("if not PRIMARY_FP32_GATES_PASSED:") : archive]
    assert "raise RuntimeError" not in scientific_stop


def test_e033_technical_failure_is_explained_archived_and_never_regenerated():
    source = _notebook_source()

    assert "maximum_campaign_attempts=1" in source
    assert "maximum_campaign_attempts=2" not in source
    assert "aucune campagne terminale n'est automatiquement régénérée" in source
    assert "E033_TECHNICAL_STOP = runner_summary['status'] != 'completed'" in source
    assert "technical-failure.json" in source
    assert "attempt-history.csv" in source
    assert "export-diagnostics.csv" in source
    assert "failed-trials.csv" in source
    assert "remote-campaign-diagnostics.csv" in source
    assert "inspect_archive_without_regenerating" in source
    assert "-technical-failure.tar.gz" in source
    assert "STOP technique E033 — aucune nouvelle génération" in source
    assert "Ne relancez pas la campagne." in source

    execution = source[source.index("runner_summary = runner.run()") : source.index("## 5.")]
    assert "raise RuntimeError" not in execution
    assert "(remote or {}).get('error')" in execution
    assert "record.get('error')" in execution

    scientific = source[source.index("## 5.") :]
    assert scientific.count("if E033_TECHNICAL_STOP:") == 7
    assert scientific.count("Cellule ignorée après le STOP technique") == 7


def test_e033_generated_code_is_syntax_valid_and_builder_is_idempotent():
    code_source = _notebook_source(code_only=True)
    ast.parse(code_source)

    before = NOTEBOOK.read_bytes()
    builder_source = BUILDER.read_text(encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(BUILDER)}
    exec(compile(builder_source, str(BUILDER), "exec"), namespace)
    assert NOTEBOOK.read_bytes() == before


def test_e033_finite_helper_rejects_booleans_and_non_finite_values():
    module = ast.parse(_notebook_source(code_only=True))
    helper = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "finite"
    )
    namespace = {"math": __import__("math")}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    finite = namespace["finite"]

    for value in (None, True, False, "", "nan", "inf", "-inf", object()):
        assert finite(value) is None
    assert finite("0") == 0.0
    assert finite("1e-9") == 1e-9
