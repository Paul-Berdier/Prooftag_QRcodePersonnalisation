from pathlib import Path
import json

from PIL import Image

from prooftag_qr.e045_foundation import (
    FoundationConfig,
    _inventory_connection,
    _walk_relevant_files,
    extract_observations,
    inventory_artifacts,
)
from prooftag_qr.resilient_experiment import classify_failure


def _config(tmp_path: Path) -> FoundationConfig:
    data = tmp_path / "data"
    return FoundationConfig(
        data_root=data,
        output_root=data / "e045-foundation-v1",
        source_commit="a" * 40,
        worker_id="pytest",
        max_files=1000,
        max_hash_bytes=8 * 1024 * 1024,
        max_parse_bytes=8 * 1024 * 1024,
    )


def test_generic_artifact_rasters_are_deferred_but_structured_files_remain(tmp_path: Path):
    cfg = _config(tmp_path)
    generic = cfg.data_root / "artifacts/run"
    canonical = cfg.data_root / "e044-test"
    generic.mkdir(parents=True)
    canonical.mkdir(parents=True)

    Image.new("RGB", (16, 16), "white").save(generic / "frame-000001.png")
    Image.new("RGB", (16, 16), "white").save(generic / "99-FINAL-QR.png")
    (generic / "results.json").write_text("[]", encoding="utf-8")
    Image.new("RGB", (16, 16), "white").save(canonical / "frame.png")

    stats = {}
    selected = {
        path.relative_to(cfg.data_root).as_posix()
        for path in _walk_relevant_files(cfg, selection_stats=stats)
    }

    assert "artifacts/run/frame-000001.png" not in selected
    assert "artifacts/run/99-FINAL-QR.png" not in selected
    assert "artifacts/run/results.json" in selected
    assert "e044-test/frame.png" in selected
    assert stats["generic_artifact_images_deferred"] == 2
    assert stats["generic_artifact_priority_images"] == 0


def test_referenced_generic_artifact_image_is_hashed_on_demand(tmp_path: Path):
    cfg = _config(tmp_path)
    generic = cfg.data_root / "artifacts/e044-export"
    generic.mkdir(parents=True)
    image = generic / "frame-unimportant-name.png"
    Image.new("RGB", (32, 32), "white").save(image)
    rows = [{
        "experiment": "e044-multi-prompt-best-pipeline-v1",
        "prompt": "test prompt",
        "method_id": "candidate",
        "configuration_json": json.dumps({"srmpgd": {"gamma": 500}}),
        "image_path": str(image),
        "qr_verify_exact_presets": 12,
        "status": "succeeded",
    }]
    (generic / "comparison.json").write_text(json.dumps(rows), encoding="utf-8")

    plan = cfg.output_root / "plan"
    plan.mkdir(parents=True)
    summary = inventory_artifacts(cfg, plan)
    assert summary["generic_artifact_images_deferred"] >= 1

    extract_observations(cfg, plan)
    conn = _inventory_connection(plan / "foundation.sqlite")
    try:
        artifact = conn.execute(
            "SELECT pixel_sha256 FROM artifacts WHERE path=?",
            (str(image),),
        ).fetchone()
        observation = conn.execute(
            "SELECT image_sha256 FROM observations WHERE image_path=?",
            (str(image),),
        ).fetchone()
    finally:
        conn.close()

    assert artifact is not None and len(artifact["pixel_sha256"]) == 64
    assert observation is not None
    assert observation["image_sha256"] == artifact["pixel_sha256"]


def test_max_files_is_configuration_error_not_transient_retry():
    decision = classify_failure(
        RuntimeError(
            "configuration max_files insuffisante: limite max_files dépassée (200000)"
        )
    )
    assert decision.kind == "deterministic"
    assert decision.retryable is False
    assert decision.operator_action_required is True

def test_all_generic_artifact_rasters_are_deferred_even_if_filename_looks_important(tmp_path: Path):
    cfg = _config(tmp_path)
    generic = cfg.data_root / "artifacts" / "some-run" / "stage2" / "frames"
    generic.mkdir(parents=True)

    names = [
        "frame-000001.png",
        "winner-stage2-final.png",
        "selected-srmpgd.png",
        "stage1.png",
        "contact-sheet.png",
        "pipeline-final.webp",
    ]
    paths = []
    for name in names:
        path = generic / name
        Image.new("RGB", (16, 16), "white").save(path)
        paths.append(path)

    stats = {}
    selected = set(_walk_relevant_files(cfg, selection_stats=stats))

    assert all(path not in selected for path in paths)
    assert stats["generic_artifact_images_deferred"] == len(paths)
    assert stats["generic_artifact_priority_images"] == 0


def test_non_artifacts_experiment_images_are_still_inventory_candidates(tmp_path: Path):
    cfg = _config(tmp_path)
    exp = cfg.data_root / "e044-multi-prompt-best-pipeline-v1" / "prompts" / "p01"
    exp.mkdir(parents=True)
    image = exp / "final.png"
    Image.new("RGB", (16, 16), "white").save(image)

    selected = set(_walk_relevant_files(cfg, selection_stats={}))

    assert image in selected

