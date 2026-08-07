import csv
import json
from pathlib import Path

import numpy as np
import pytest

from prooftag_qr.parameter_advisor import E026ParameterAdvisor, load_lab_exports


def _write_training_export(path: Path, prompts: int = 12) -> None:
    fields = [
        "campaign_id",
        "trial_id",
        "prompt_id",
        "prompt_text",
        "method_id",
        "method_configuration_json",
        "payload_length",
        "error_correction",
        "status",
        "seed",
        "quality_qr_verify_any_exact",
        "quality_qr_verify_tolerance_score",
        "quality_clip_aesthetic",
        "quality_clip_score",
        "quality_hpsv2_1",
        "quality_high_saturation_pixel_ratio",
        "total_ms",
    ]
    safe = {
        "id": "safe",
        "backend": "controlnet",
        "output_variant": "srpg",
        "generation": {"steps": 40, "guidance_scale": 7.5},
        "tools": {"settings": {"srpg_qr_weight": 600, "srpg_perceptual_weight": 2}},
    }
    pretty = {
        "id": "pretty",
        "backend": "controlnet",
        "output_variant": "srpg",
        "generation": {"steps": 28, "guidance_scale": 9.0},
        "tools": {"settings": {"srpg_qr_weight": 150, "srpg_perceptual_weight": 4}},
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for prompt_index in range(prompts):
            for repeat in range(5):
                for method_id, configuration, success in [
                    ("safe", safe, 1),
                    ("pretty", pretty, 0),
                ]:
                    writer.writerow(
                        {
                            "campaign_id": "campaign",
                            "trial_id": f"p{prompt_index}-{repeat}-{method_id}",
                            "prompt_id": f"p{prompt_index}",
                            "prompt_text": f"Unseen visual prompt number {prompt_index}",
                            "method_id": method_id,
                            "method_configuration_json": json.dumps(configuration),
                            "payload_length": 24,
                            "error_correction": "M",
                            "status": "accepted" if success else "rejected",
                            "seed": 1000 + repeat,
                            "quality_qr_verify_any_exact": success,
                            "quality_qr_verify_tolerance_score": 0.9 if success else 0.2,
                            "quality_clip_aesthetic": 5.5 if success else 7.5,
                            "quality_clip_score": 0.65 if success else 0.8,
                            "quality_hpsv2_1": 0.24 if success else 0.3,
                            "quality_high_saturation_pixel_ratio": 0.01 if success else 0.08,
                            "total_ms": 45000 if success else 30000,
                        }
                    )


def test_e026_loads_exports_without_using_output_metrics_as_features(tmp_path):
    export = tmp_path / "campaign.csv"
    _write_training_export(export)
    dataset = load_lab_exports(
        [export],
        embedding_provider=lambda prompt: np.asarray([len(prompt), prompt.count("number")]),
    )

    assert dataset.audit["usable_rows"] == 120
    assert dataset.audit["prompt_groups"] == 12
    assert dataset.audit["recipes"] == 2
    assert dataset.audit["embedding_dimensions"] == 2
    record = dataset.records[0]
    assert "qr_success" in record.targets
    assert "quality_qr_verify_any_exact" not in record.context_features
    assert "quality_clip_aesthetic" not in record.context_features


def test_e026_grouped_model_keeps_scan_as_a_hard_first_objective(tmp_path):
    export = tmp_path / "campaign.csv"
    _write_training_export(export)
    dataset = load_lab_exports(
        [export], embedding_provider=lambda prompt: np.asarray([len(prompt), 1.0])
    )
    advisor = E026ParameterAdvisor(trees=48)
    report = advisor.fit(dataset.records)
    recommendations = advisor.recommend(
        prompt="A completely unseen glass garden",
        prompt_embedding=[32.0, 1.0],
        payload_length=24,
        error_correction="M",
        candidates=dataset.candidates,
        scan_probability_threshold=0.70,
        limit=2,
    )

    assert report["validation"].startswith("GroupKFold")
    assert recommendations[0].candidate.method_id == "safe"
    assert recommendations[0].predicted_qr_success > recommendations[1].predicted_qr_success
    assert recommendations[0].rank == 1

    model_path = tmp_path / "advisor.joblib"
    advisor.save(model_path)
    restored = E026ParameterAdvisor.load(model_path)
    assert restored.training_report["rows"] == 120


def test_e026_refuses_a_tiny_dataset(tmp_path):
    export = tmp_path / "campaign.csv"
    _write_training_export(export, prompts=2)
    dataset = load_lab_exports([export])

    with pytest.raises(ValueError, match="insufficient E026 dataset"):
        E026ParameterAdvisor(trees=16).fit(dataset.records)
