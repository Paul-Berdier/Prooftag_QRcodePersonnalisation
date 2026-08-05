import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/e025-quality-retest.py")
    spec = importlib.util.spec_from_file_location("e025_quality", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema():
    return {
        "validation": {
            "engine": "antfu/qr-verify@0.2.0",
            "tolerance_presets": 37,
            "acceptance": "at_least_one_exact_preset",
            "physical_probability": False,
        },
        "quality_scoring": {
            "clip_enabled": True,
            "hpsv2_1_enabled": True,
            "acceptance_effect": "none",
            "metrics": {
                name: {}
                for name in (
                    "clip_similarity",
                    "clip_score",
                    "clip_aesthetic",
                    "hpsv2_1",
                )
            },
        },
        "profiles": [
            {"id": "diffqrcoder_srpg", "enabled": True},
            {"id": "diffqrcoder_paper_srpg", "enabled": True},
        ],
    }


def test_e025_reuses_the_paired_e024_campaign():
    manifest = _module().build_manifest(
        _schema(), payload="https://ptag.io/t/e025", seeds=[61001]
    )

    assert len(manifest["prompts"]) == 10
    assert [method["id"] for method in manifest["methods"]] == [
        "diffqrcoder_srpg",
        "diffqrcoder_paper_srpg",
    ]
    assert manifest["name"] == "E025 - QR-Verify + CLIP/HPS - all"


def test_e025_refuses_disabled_or_decision_affecting_scores():
    schema = _schema()
    schema["quality_scoring"]["hpsv2_1_enabled"] = False

    with pytest.raises(RuntimeError, match="not running E025"):
        _module().build_manifest(schema, payload="https://ptag.io/t/e025")
