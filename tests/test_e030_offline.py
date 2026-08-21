import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from prooftag_qr.e030_offline import (
    E029_EXPERIMENT,
    discover_e029_archive,
    discover_e029_export_directory,
    e029_export_sha256,
    selective_extract_e029_archive,
    sha256_file,
    validate_e029_export,
    validate_rescore_journal_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _e029_export(root: Path) -> Path:
    root.mkdir(parents=True)
    gallery = root / "e029-gallery"
    images = gallery / "images"
    images.mkdir(parents=True)
    image_rows = []
    state_rows = []
    for index, color in enumerate(((0, 0, 0), (255, 255, 255)), 1):
        image = images / f"{index:03d}.png"
        Image.new("RGB", (8, 8), color).save(image)
        trial_id = f"trial-{index}"
        state_rows.append({"trial_id": trial_id, "status": "accepted"})
        image_rows.append(
            {
                "trial_id": trial_id,
                "local_image": f"images/{image.name}",
                "image_sha256": sha256_file(image),
            }
        )
    _write_csv(root / "e029-state-results.csv", state_rows)
    _write_csv(gallery / "gallery-index.csv", image_rows)
    (gallery / "gallery-audit.json").write_text("{}", encoding="utf-8")
    Image.new("RGB", (8, 8)).save(gallery / "paired-advisor-sample.png")
    Image.new("RGB", (8, 8)).save(gallery / "advisor-deliverable-winners.png")
    payload = "https://ptag.io/t/e029"
    import hashlib

    manifest = {
        "experiment": E029_EXPERIMENT,
        "plan": {
            "trial_count": 2,
            "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "prompt_count": 1,
            "seed_count": 2,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in (
        "training-report.json",
        "e029-pairing-audit.csv",
        "e029-srmpgd-iteration-zero-raster-audit.csv",
        "e029-policy-decisions.csv",
        "e029-policy-report.json",
    ):
        (root / name).write_text("{}" if name.endswith(".json") else "ok\n", encoding="utf-8")
    return root


def _archive(source: Path, target: Path) -> Path:
    with tarfile.open(target, "w:gz") as bundle:
        bundle.add(source, arcname=source.name)
        payload = b"large advisor deliberately excluded" * 100
        info = tarfile.TarInfo(f"{source.name}/e029-initial-prompt-parameter-advisor.joblib")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return target


def test_e030_validates_and_hashes_a_complete_export(tmp_path):
    source = _e029_export(tmp_path / "e029-complete")
    audit = validate_e029_export(source)
    assert audit["state_rows"] == 2
    assert audit["gallery_rows"] == 2
    assert audit["unique_rasters"] == 2
    assert len(e029_export_sha256(source)) == 64


def test_e030_extracts_only_reports_and_gallery_not_the_large_model(tmp_path):
    source = _e029_export(tmp_path / "e029-complete")
    archive = _archive(source, tmp_path / "source.tar.gz")
    output = selective_extract_e029_archive(archive, tmp_path / "selected", reserve_bytes=0)

    assert validate_e029_export(output)["state_rows"] == 2
    assert not (output / "e029-initial-prompt-parameter-advisor.joblib").exists()
    assert len(list((output / "e029-gallery" / "images").glob("*.png"))) == 2


def test_e030_rejects_traversal_even_when_member_would_not_be_selected(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"bad"
        info = tarfile.TarInfo("root/../../escape.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    try:
        selective_extract_e029_archive(archive, tmp_path / "selected", reserve_bytes=0)
    except ValueError as exc:
        assert "unsafe tar member" in str(exc)
    else:
        raise AssertionError("tar traversal was accepted")


def test_e030_discovers_persistent_export_before_archive_fallback(tmp_path):
    older = _e029_export(tmp_path / "persistent" / "older")
    newer = _e029_export(tmp_path / "persistent" / "newer")
    manifest = newer / "manifest.json"
    manifest.touch()
    assert discover_e029_export_directory((tmp_path / "persistent",)) == newer.resolve()

    archive = _archive(older, tmp_path / f"20260820-{E029_EXPERIMENT}.tar.gz")
    assert discover_e029_archive(tmp_path) == archive.resolve()


def test_e030_refuses_incomplete_or_modified_gallery(tmp_path):
    source = _e029_export(tmp_path / "e029-complete")
    (source / "e029-gallery" / "images" / "001.png").write_bytes(b"modified")
    try:
        validate_e029_export(source)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("modified gallery image was accepted")


def _rescore_identity() -> dict[str, object]:
    return {
        "engine_version": "qr-verify@0.2.0",
        "scoring_version": "qr-verify-conservative-v1",
        "implementation_sha256": "c" * 64,
        "repetitions": 5,
        "preset_count": 37,
    }


def _rescore_row(source_hash: str, raster_hash: str) -> dict[str, object]:
    return {
        "run_id": "run-1",
        "source_png_sha256": source_hash,
        "score": {
            "image_sha256": raster_hash,
            "payload_sha256": "b" * 64,
            **_rescore_identity(),
        },
    }


def test_e030_rescore_journal_binds_score_to_pixels_and_complete_identity():
    source_hash = "a" * 64
    raster_hash = "d" * 64
    row = _rescore_row(source_hash, raster_hash)

    validated = validate_rescore_journal_rows(
        [row],
        run_id="run-1",
        expected_raster_sha256_by_source={source_hash: raster_hash},
        payload_sha256="b" * 64,
        scorer_identity=_rescore_identity(),
    )

    assert validated == [row]


def test_e030_rescore_journal_rejects_another_raster():
    source_hash = "a" * 64
    row = _rescore_row(source_hash, "e" * 64)

    with pytest.raises(ValueError, match="raster hash mismatch"):
        validate_rescore_journal_rows(
            [row],
            run_id="run-1",
            expected_raster_sha256_by_source={source_hash: "d" * 64},
            payload_sha256="b" * 64,
            scorer_identity=_rescore_identity(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("engine_version", "other-engine"),
        ("scoring_version", "other-scorer"),
        ("implementation_sha256", "e" * 64),
        ("repetitions", 3),
        ("preset_count", 36),
        ("payload_sha256", "e" * 64),
    ],
)
def test_e030_rescore_journal_rejects_identity_mismatch(field, value):
    source_hash = "a" * 64
    raster_hash = "d" * 64
    row = _rescore_row(source_hash, raster_hash)
    row["score"][field] = value

    with pytest.raises(ValueError, match="mismatch"):
        validate_rescore_journal_rows(
            [row],
            run_id="run-1",
            expected_raster_sha256_by_source={source_hash: raster_hash},
            payload_sha256="b" * 64,
            scorer_identity=_rescore_identity(),
        )


def test_e030_rescore_journal_rejects_conflicting_duplicate():
    source_hash = "a" * 64
    raster_hash = "d" * 64
    first = _rescore_row(source_hash, raster_hash)
    conflicting = json.loads(json.dumps(first))
    conflicting["score"]["cache_key"] = "different"

    with pytest.raises(ValueError, match="conflicting.*duplicate"):
        validate_rescore_journal_rows(
            [first, conflicting],
            run_id="run-1",
            expected_raster_sha256_by_source={source_hash: raster_hash},
            payload_sha256="b" * 64,
            scorer_identity=_rescore_identity(),
        )
