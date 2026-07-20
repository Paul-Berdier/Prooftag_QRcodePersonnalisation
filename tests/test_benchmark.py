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
    assert "Brut strict" in report
    assert "Première référence" in report
