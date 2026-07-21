import runpy
from pathlib import Path

benchmark = runpy.run_path(str(Path("scripts/benchmark.py")))


def test_prometheus_parser_extracts_variant_quality_metrics():
    content = """
prooftag_qr_repair_variant_scan_pass_rate{variant="uncertain_48"} 1.0
prooftag_qr_repair_variant_image_quality{metric="changed_pixel_ratio",variant="uncertain_48"} 0.453
"""

    metrics = benchmark["variant_metrics"](content)

    assert metrics["uncertain_48"]["scan_pass_rate"] == 1.0
    assert metrics["uncertain_48"]["changed_pixel_ratio"] == 0.453


def test_report_contains_comparison_and_gallery():
    row = {
        "case": "botanical-short",
        "status": "accepted",
        "selected_variant": "uncertain_48",
        "scan_pass_rate": 1.0,
        "changed_pixel_ratio": 0.45,
        "entropy_bits": 4.15,
        "total_ms": 7600,
        "guided_artifact_available": True,
        "guided_control_artifact_available": True,
        "guided_mask_artifact_available": True,
        "guided_unprojected_artifact_available": True,
        "guided_projected_artifact_available": True,
        "srpg_artifact_available": True,
        "srpg_stage_status": "accepted",
        "srpg_scan_pass_rate": 1.0,
        "srpg_module_error_rate": 0.01,
        "srpg_step_metrics": [
            {"module_error_rate": 0.2},
            {"module_error_rate": 0.1},
        ],
        "latent_artifact_variant": "guided_latent_srl",
    }
    summary = {
        "git_commit": "abc123",
        "created_at": "2026-07-20T12:00:00Z",
        "acceptance_rate": 1.0,
        "accepted_cases": 1,
        "case_count": 1,
        "mean_scan_pass_rate": 1.0,
        "mean_changed_pixel_ratio": 0.45,
        "mean_total_ms": 7600,
        "raw_acceptance_rate": 0.5,
        "post_latent_acceptance_rate": 1.0,
        "latent_rescue_cases": 1,
        "results": [row],
    }

    report = benchmark["render_report"]("run-1", summary, None, [])

    assert "botanical-short" in report
    assert "uncertain_48" in report
    assert "cases/botanical-short/final.png" in report
    assert "attempt_1_guided.png" in report
    assert "attempt_1_guided_control.png" in report
    assert "attempt_1_guided_mask.png" in report
    assert "attempt_1_guided_unprojected.png" in report
    assert "attempt_1_guided_projected.png" in report
    assert "attempt_1_srpg.png" in report
    assert "Erreur de modules par étape SRPG" in report
    assert "attempt_1_guided_latent_srl.png" in report
    assert "Brut strict" in report
    assert "Première référence" in report


def test_guided_prefixed_variants_are_kept_as_debug_artifacts():
    is_debug_variant = benchmark["is_debug_variant"]

    assert is_debug_variant("guided")
    assert is_debug_variant("guided_latent_srl")
    assert is_debug_variant("guided_latent_uncertain_48")
    assert not is_debug_variant("guided_centers_95")


def test_refinement_csv_exports_unprojected_change_metrics():
    source = Path("scripts/benchmark.py").read_text(encoding="utf-8")

    assert '"unprojected_changed_pixel_ratio",' in source
    assert '"unprojected_mean_absolute_change",' in source
    assert 'run_dir / "srpg-steps.csv"' in source
    assert '"noise_delta_rms",' in source
    assert '"preview_steps",' in source
    assert "BENCHMARK_INCOMPLETE=" in source
    assert "srpg_step_{step:02d}_x0" in source
    assert "srpg_step_{step:02d}_errors" in source
