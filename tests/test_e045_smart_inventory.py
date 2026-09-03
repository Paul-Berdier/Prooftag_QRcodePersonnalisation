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
    assert "artifacts/run/99-FINAL-QR.png" in selected
    assert "artifacts/run/results.json" in selected
    assert "e044-test/frame.png" in selected
    assert stats["generic_artifact_images_deferred"] == 1


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

def test_priority_token_in_parent_directory_does_not_promote_all_frames(tmp_path: Path):
    """Régression du run réel 4c059d9: le dossier stage2 promouvait tous ses PNG."""
    cfg = _config(tmp_path)
    generic = cfg.data_root / "artifacts" / "some-run" / "stage2" / "frames"
    generic.mkdir(parents=True)

    ordinary = generic / "frame-000001.png"
    explicit = generic / "winner-stage2-final.png"
    Image.new("RGB", (16, 16), "white").save(ordinary)
    Image.new("RGB", (16, 16), "white").save(explicit)

    stats = {}
    selected = {
        path.relative_to(cfg.data_root).as_posix()
        for path in _walk_relevant_files(cfg, selection_stats=stats)
    }

    assert "artifacts/some-run/stage2/frames/frame-000001.png" not in selected
    assert "artifacts/some-run/stage2/frames/winner-stage2-final.png" in selected
    assert stats["generic_artifact_images_deferred"] == 1
    assert stats["generic_artifact_priority_images"] == 1


def test_priority_token_in_srmpgd_parent_directory_does_not_promote_frame(tmp_path: Path):
    cfg = _config(tmp_path)
    generic = cfg.data_root / "artifacts" / "srmpgd" / "trajectory"
    generic.mkdir(parents=True)

    image = generic / "000123.png"
    Image.new("RGB", (16, 16), "white").save(image)

    stats = {}
    selected = list(_walk_relevant_files(cfg, selection_stats=stats))

    assert image not in selected
    assert stats["generic_artifact_images_deferred"] == 1
    assert stats["generic_artifact_priority_images"] == 0

