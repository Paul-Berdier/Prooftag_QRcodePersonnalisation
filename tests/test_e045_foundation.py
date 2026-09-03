import json
from pathlib import Path

from PIL import Image

from prooftag_qr.e045_foundation import (
    FoundationConfig,
    run_foundation,
    verify_complete,
)


def test_foundation_builds_and_resumes_without_touching_sources(tmp_path: Path):
    data = tmp_path / "data"
    output = data / "e045-foundation-v1"
    prompt_dir = data / "e044-multi-prompt-best-pipeline-v1/prompts/p01"
    scoring = prompt_dir / "scoring"
    scoring.mkdir(parents=True)

    image_path = prompt_dir / "pipeline/99-FINAL-QR.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(image_path)

    comparison = [{
        "experiment": "e044-multi-prompt-best-pipeline-v1",
        "prompt": "one blue vase",
        "prompt_family": "minimal",
        "method_id": "e044_test_gamma500",
        "stage": "srmpgd",
        "seed": 7,
        "configuration_json": json.dumps({
            "stage1": {"steps": 40},
            "stage2": {"srg_weight": 50},
            "srmpgd": {"gamma": 500},
        }),
        "image_path": str(image_path),
        "qr_verify_exact_presets": 12,
        "original_exact": False,
        "clip_aesthetic": 5.2,
        "status": "succeeded",
    }]
    (scoring / "comparison.json").write_text(
        json.dumps(comparison), encoding="utf-8"
    )

    # E031 must remain evaluation-only even when it has a strong score.
    holdout = data / "e031-results"
    holdout.mkdir()
    (holdout / "results.csv").write_text(
        "experiment,prompt,method_id,configuration_json,qr_verify_exact_presets,status\n"
        'e031-holdout,frozen prompt,fixed,"{""stage2"":{""steps"":40}}",37,succeeded\n',
        encoding="utf-8",
    )

    source_before = (scoring / "comparison.json").read_bytes()
    config = FoundationConfig(
        data_root=data,
        output_root=output,
        source_commit="a" * 40,
        worker_id="pytest",
        max_files=1000,
        max_hash_bytes=8 * 1024 * 1024,
        max_parse_bytes=8 * 1024 * 1024,
    )
    first = run_foundation(config, force_recover_stale=True)
    assert first["complete"] is True
    assert first["resilience_selftest_passed"] is True
    assert first["advisor_training_authorized"] is False
    assert (scoring / "comparison.json").read_bytes() == source_before

    # Idempotent same-commit resume: no source is regenerated or removed.
    second = run_foundation(config, force_recover_stale=True)
    assert second["plan_id"] == first["plan_id"]
    verified = verify_complete(output)
    assert verified["valid"] is True

    rows = [
        json.loads(line)
        for line in (config.plan_dir / "canonical-observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    e44 = next(row for row in rows if row["experiment_id"] == "E044")
    e31 = next(row for row in rows if row["experiment_id"] == "E031")
    assert e44["eligible_parameter_advisor"] == 1
    assert e31["evaluation_only"] == 1
    assert e31["eligible_parameter_advisor"] == 0
