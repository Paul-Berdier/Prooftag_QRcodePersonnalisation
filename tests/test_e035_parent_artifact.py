from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from prooftag_qr.e035_parent_artifact import (
    E034_OBSERVED_STAGE1_FILE_SHA256,
    E034_OBSERVED_STAGE1_IMAGE_SHA256,
    IMAGE_FILENAME,
    LATENT_FILENAME,
    METADATA_FILENAME,
    canonical_json_sha256,
    export_parent_artifact,
    image_raster_sha256,
    load_parent_artifact,
    verify_parent_artifact,
)

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def source_metadata() -> dict:
    return {
        "payload": "https://ptag.io/t/e033",
        "error_correction": "M",
        "qr_version": 3,
        "qr_mask_pattern": 4,
        "qr_module_size": 20,
        "qr_padding_px": 78,
        "source_commit": "933a947ee226e9e6b36145d35e41d53c90f50484",
        "source_plan": "e034-srmpgd-four-iteration-gate-v1",
        "source_run_id": "test-run",
        "source_method_id": "e033_public_demo_srpg_exact_e034_export",
        "parent_origin": "exact_e034_stage2_export",
        "vae_scaling_factor": 0.18215,
        "base_model_id": "test/model",
        "base_model_revision": "test-revision",
        "diffqrcoder_revision": "e24ea73ee2e13c7e6e87cb422e8b11784e70ae00",
        "stage1_image_sha256": E034_OBSERVED_STAGE1_IMAGE_SHA256,
        "stage1_file_sha256": E034_OBSERVED_STAGE1_FILE_SHA256,
        "generation": {"stage1_regenerated": False},
    }


def test_parent_artifact_round_trip_and_contract(tmp_path: Path) -> None:
    latent = torch.arange(4 * 8 * 8, dtype=torch.float16).reshape(1, 4, 8, 8)
    image = Image.new("RGB", (736, 736), (210, 220, 230))
    metadata = export_parent_artifact(
        tmp_path,
        latent=latent,
        image=image,
        source=source_metadata(),
    )
    assert (tmp_path / IMAGE_FILENAME).is_file()
    assert (tmp_path / LATENT_FILENAME).is_file()
    assert (tmp_path / METADATA_FILENAME).is_file()
    assert metadata["files"]["image"]["raster_sha256"] == image_raster_sha256(image)
    verified = verify_parent_artifact(
        tmp_path,
        expected={"source_commit": source_metadata()["source_commit"]},
    )
    assert verified["contract_sha256"] == metadata["contract_sha256"]
    loaded = load_parent_artifact(tmp_path)
    assert torch.equal(loaded.latent, latent)
    assert loaded.image.size == (736, 736)


def test_parent_artifact_rejects_file_tampering(tmp_path: Path) -> None:
    export_parent_artifact(
        tmp_path,
        latent=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        image=Image.new("RGB", (736, 736), "white"),
        source=source_metadata(),
    )
    with (tmp_path / LATENT_FILENAME).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_parent_artifact(tmp_path)


def test_parent_artifact_rejects_metadata_tampering(tmp_path: Path) -> None:
    export_parent_artifact(
        tmp_path,
        latent=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        image=Image.new("RGB", (736, 736), "white"),
        source=source_metadata(),
    )
    metadata_path = tmp_path / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source"]["source_commit"] = "0" * 40
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash mismatch"):
        verify_parent_artifact(tmp_path)


def test_parent_artifact_rejects_noncanonical_filenames(tmp_path: Path) -> None:
    export_parent_artifact(
        tmp_path,
        latent=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
        image=Image.new("RGB", (736, 736), "white"),
        source=source_metadata(),
    )
    metadata_path = tmp_path / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["image"]["name"] = "../parent-stage2.png"
    unsigned = dict(metadata)
    unsigned.pop("contract_sha256")
    metadata["contract_sha256"] = canonical_json_sha256(unsigned)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        verify_parent_artifact(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("qr_version", 0, "qr_version"),
        ("qr_mask_pattern", 8, "qr_mask_pattern"),
        ("qr_module_size", 0, "qr_module_size"),
        ("qr_padding_px", -1, "qr_padding_px"),
        ("source_commit", "not-a-sha", "source_commit"),
        ("diffqrcoder_revision", "ABC", "diffqrcoder_revision"),
        ("stage1_image_sha256", "0" * 64, "stage1_image_sha256"),
        ("stage1_file_sha256", "0" * 64, "stage1_file_sha256"),
        ("source_method_id", "unapproved", "source_method_id"),
        ("generation", {"stage1_regenerated": True}, "stage1_regenerated"),
    ),
)
def test_parent_artifact_rejects_invalid_source_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = source_metadata()
    source[field] = value
    with pytest.raises(ValueError, match=message):
        export_parent_artifact(
            tmp_path,
            latent=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            image=Image.new("RGB", (736, 736), "white"),
            source=source,
        )


def test_parent_artifact_refuses_pickle_latent_by_contract() -> None:
    script = Path(__file__).parents[1] / "scripts" / "export_e035_parent_artifact.py"
    text = script.read_text(encoding="utf-8")
    assert "only safetensors input is accepted" in text
    assert "torch.load" not in text
