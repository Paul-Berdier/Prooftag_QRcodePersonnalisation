import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/e024-qr-verify-retest.py")
    spec = importlib.util.spec_from_file_location("e024_qr_verify", path)
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
        "profiles": [
            {"id": "diffqrcoder_srpg", "enabled": True},
            {"id": "diffqrcoder_paper_srpg", "enabled": True},
        ],
    }


def test_e024_reuses_all_ten_e022_prompts_and_two_generation_recipes():
    manifest = _module().build_manifest(
        _schema(), payload="https://ptag.io/t/e024", seeds=[61001]
    )

    assert len(manifest["prompts"]) == 10
    assert [method["id"] for method in manifest["methods"]] == [
        "diffqrcoder_srpg",
        "diffqrcoder_paper_srpg",
    ]
    assert manifest["seeds"] == [61001]


def test_e024_refuses_an_api_with_another_validation_contract():
    schema = _schema()
    schema["validation"]["engine"] = "legacy"

    with pytest.raises(RuntimeError, match="not running E024"):
        _module().build_manifest(schema, payload="https://ptag.io/t/e024")
