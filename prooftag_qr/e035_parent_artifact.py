"""Immutable parent artefact contract for E035.

The paired E035 loss runner never regenerates Stage 1 or Stage 2. It accepts only a
canonical Stage-2 PNG plus its exact latent stored in safetensors, both bound by
SHA-256 and explicit VAE/model provenance. Parent preparation is a separate, audited
operation that must start from the exact archived E034 Stage-1 raster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA = "prooftag.e035.parent-artifact.v1"
IMAGE_FILENAME = "parent-stage2.png"
LATENT_FILENAME = "parent-stage2-latent.safetensors"
METADATA_FILENAME = "parent-stage2-metadata.json"
LATENT_KEY = "latent"
E034_OBSERVED_STAGE1_IMAGE_SHA256 = (
    "ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079"
)
E034_OBSERVED_STAGE1_FILE_SHA256 = (
    "be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71"
)
ALLOWED_SOURCE_METHODS = frozenset(
    {
        "e033_public_demo_srpg_from_fixed_e034_stage1",
        "e033_public_demo_srpg_exact_e034_export",
    }
)

REQUIRED_SOURCE_FIELDS = (
    "payload",
    "error_correction",
    "qr_version",
    "qr_mask_pattern",
    "qr_module_size",
    "qr_padding_px",
    "source_commit",
    "source_plan",
    "source_run_id",
    "source_method_id",
    "parent_origin",
    "vae_scaling_factor",
    "base_model_id",
    "base_model_revision",
    "diffqrcoder_revision",
    "stage1_image_sha256",
    "stage1_file_sha256",
    "generation",
)


@dataclass(frozen=True, slots=True)
class LoadedParentArtifact:
    root: Path
    image: Image.Image
    latent: Any
    metadata: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Match the project tensor hash: dtype + shape + contiguous raw values."""

    source = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"{source.dtype}:{tuple(source.shape)}:".encode("utf-8"))
    digest.update(source.numpy().tobytes())
    return digest.hexdigest()


def image_raster_sha256(image: Image.Image) -> str:
    """Hash RGB pixels, mode and dimensions independently from PNG encoding."""

    import numpy as np

    source = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{source.width}x{source.height}:".encode("utf-8"))
    digest.update(np.asarray(source, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_png(path: Path, image: Image.Image) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.convert("RGB").save(
        temporary,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    os.replace(temporary, path)


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_SOURCE_FIELDS if name not in source]
    if missing:
        raise ValueError(f"parent source metadata is missing: {missing}")
    cleaned = {str(key): value for key, value in source.items()}
    if not str(cleaned["payload"]).startswith("https://"):
        raise ValueError("parent payload must be an HTTPS URL")
    if cleaned["error_correction"] not in {"M", "Q", "H"}:
        raise ValueError("unsupported parent error correction")
    qr_version = int(cleaned["qr_version"])
    qr_mask_pattern = int(cleaned["qr_mask_pattern"])
    qr_module_size = int(cleaned["qr_module_size"])
    qr_padding_px = int(cleaned["qr_padding_px"])
    if qr_version < 1 or qr_version > 40:
        raise ValueError("qr_version must be between 1 and 40")
    if qr_mask_pattern < 0 or qr_mask_pattern > 7:
        raise ValueError("qr_mask_pattern must be between 0 and 7")
    if qr_module_size < 1:
        raise ValueError("qr_module_size must be positive")
    if qr_padding_px < 0:
        raise ValueError("qr_padding_px cannot be negative")
    cleaned.update(
        {
            "qr_version": qr_version,
            "qr_mask_pattern": qr_mask_pattern,
            "qr_module_size": qr_module_size,
            "qr_padding_px": qr_padding_px,
        }
    )
    scaling_factor = float(cleaned["vae_scaling_factor"])
    if not math.isfinite(scaling_factor) or scaling_factor <= 0:
        raise ValueError("vae_scaling_factor must be finite and positive")
    cleaned["vae_scaling_factor"] = scaling_factor
    for field in (
        "source_commit",
        "source_plan",
        "source_run_id",
        "source_method_id",
        "parent_origin",
        "base_model_id",
        "base_model_revision",
        "diffqrcoder_revision",
    ):
        if not str(cleaned[field]).strip():
            raise ValueError(f"{field} must be non-empty")
    for field in ("source_commit", "diffqrcoder_revision"):
        value = str(cleaned[field])
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field} must be a lowercase 40-character Git SHA")

    source_method = str(cleaned["source_method_id"])
    if source_method not in ALLOWED_SOURCE_METHODS:
        raise ValueError(
            "unsupported E035 parent source_method_id: "
            f"{source_method!r}; allowed={sorted(ALLOWED_SOURCE_METHODS)}"
        )
    expected_origin = {
        "e033_public_demo_srpg_exact_e034_export": "exact_e034_stage2_export",
        "e033_public_demo_srpg_from_fixed_e034_stage1": (
            "stage2_replayed_from_exact_e034_stage1"
        ),
    }[source_method]
    if cleaned["parent_origin"] != expected_origin:
        raise ValueError(
            "parent_origin is inconsistent with source_method_id: "
            f"expected {expected_origin!r}, got {cleaned['parent_origin']!r}"
        )
    for field, expected_hash in (
        ("stage1_image_sha256", E034_OBSERVED_STAGE1_IMAGE_SHA256),
        ("stage1_file_sha256", E034_OBSERVED_STAGE1_FILE_SHA256),
    ):
        value = str(cleaned[field])
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
        if value != expected_hash:
            raise ValueError(
                f"{field} does not match the archived E034 Stage-1 asset: "
                f"expected {expected_hash}, got {value}"
            )
        cleaned[field] = value
    generation = cleaned["generation"]
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    generation = {str(key): value for key, value in generation.items()}
    if generation.get("stage1_regenerated") is not False:
        raise ValueError("generation.stage1_regenerated must be exactly false")
    cleaned["generation"] = generation
    return cleaned


def export_parent_artifact(
    root: str | Path,
    *,
    latent: Any,
    image: Image.Image,
    source: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write and bind a canonical E035 parent artefact."""

    import torch
    from safetensors.torch import save_file

    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    image_path = target / IMAGE_FILENAME
    latent_path = target / LATENT_FILENAME
    metadata_path = target / METADATA_FILENAME
    existing = [path for path in (image_path, latent_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "parent artefact already exists; use overwrite only for an explicit replacement: "
            + ", ".join(str(path) for path in existing)
        )
    if not torch.is_tensor(latent) or latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError("latent must be a BCHW torch tensor with batch size one")
    cpu_latent = latent.detach().cpu().contiguous()
    if not torch.isfinite(cpu_latent).all():
        raise ValueError("parent latent contains non-finite values")
    canonical_source = _validate_source(source)

    temporary_latent = latent_path.with_suffix(latent_path.suffix + ".tmp")
    save_file(
        {LATENT_KEY: cpu_latent},
        str(temporary_latent),
        metadata={
            "schema": SCHEMA,
            "tensor_sha256": tensor_sha256(cpu_latent),
            "source_commit": str(canonical_source["source_commit"]),
            "source_run_id": str(canonical_source["source_run_id"]),
        },
    )
    os.replace(temporary_latent, latent_path)
    _atomic_png(image_path, image)

    metadata = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "files": {
            "image": {
                "name": IMAGE_FILENAME,
                "sha256": sha256_file(image_path),
                "raster_sha256": image_raster_sha256(image),
                "size_bytes": image_path.stat().st_size,
            },
            "latent": {
                "name": LATENT_FILENAME,
                "sha256": sha256_file(latent_path),
                "size_bytes": latent_path.stat().st_size,
                "key": LATENT_KEY,
                "tensor_sha256": tensor_sha256(cpu_latent),
                "shape": list(cpu_latent.shape),
                "dtype": str(cpu_latent.dtype),
                "source_device": str(latent.device),
            },
        },
        "image": {
            "mode": "RGB",
            "width": int(image.width),
            "height": int(image.height),
        },
        "source": canonical_source,
    }
    metadata["contract_sha256"] = canonical_json_sha256(metadata)
    _atomic_json(metadata_path, metadata)
    verify_parent_artifact(target)
    return metadata


def _verify_contract_hash(metadata: dict[str, Any]) -> None:
    expected = str(metadata.get("contract_sha256") or "")
    unsigned = dict(metadata)
    unsigned.pop("contract_sha256", None)
    actual = canonical_json_sha256(unsigned)
    if expected != actual:
        raise ValueError(
            f"parent metadata contract hash mismatch: expected {expected}, got {actual}"
        )


def verify_parent_artifact(
    root: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed on any missing file, hash, shape, dtype or provenance mismatch."""

    from safetensors import safe_open

    target = Path(root)
    metadata_path = target / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing E035 parent metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unsupported parent schema: {metadata.get('schema')!r}")
    _verify_contract_hash(metadata)
    _validate_source(metadata.get("source") or {})

    for kind, default_name in (("image", IMAGE_FILENAME), ("latent", LATENT_FILENAME)):
        entry = metadata.get("files", {}).get(kind) or {}
        filename = str(entry.get("name") or default_name)
        if filename != default_name or Path(filename).name != filename:
            raise ValueError(
                f"parent {kind} filename must be the canonical {default_name!r}"
            )
        path = target / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing E035 parent {kind}: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(
                f"parent {kind} file hash mismatch: expected {entry.get('sha256')}, "
                f"got {actual_hash}"
            )
        if int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"parent {kind} size mismatch")

    image_path = target / metadata["files"]["image"]["name"]
    with Image.open(image_path) as opened:
        opened.verify()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        expected_image = metadata.get("image") or {}
        if image.size != (
            int(expected_image.get("width", -1)),
            int(expected_image.get("height", -1)),
        ):
            raise ValueError("parent image dimensions do not match metadata")
    expected_raster_hash = metadata["files"]["image"].get("raster_sha256")
    if not expected_raster_hash:
        raise ValueError("parent image raster hash is missing")
    actual_raster_hash = image_raster_sha256(image)
    if actual_raster_hash != expected_raster_hash:
        raise ValueError(
            "parent image raster hash mismatch: "
            f"expected {expected_raster_hash}, got {actual_raster_hash}"
        )

    latent_entry = metadata["files"]["latent"]
    latent_path = target / latent_entry["name"]
    with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if keys != [latent_entry.get("key", LATENT_KEY)]:
            raise ValueError(f"unexpected parent latent keys: {keys}")
        latent = handle.get_tensor(keys[0])
        safetensor_metadata = handle.metadata() or {}
    if list(latent.shape) != list(latent_entry.get("shape") or []):
        raise ValueError("parent latent shape mismatch")
    if str(latent.dtype) != str(latent_entry.get("dtype")):
        raise ValueError("parent latent dtype mismatch")
    actual_tensor_hash = tensor_sha256(latent)
    if actual_tensor_hash != latent_entry.get("tensor_sha256"):
        raise ValueError("parent latent tensor hash mismatch")
    embedded_hash = safetensor_metadata.get("tensor_sha256")
    if embedded_hash and embedded_hash != actual_tensor_hash:
        raise ValueError("embedded safetensors tensor hash mismatch")

    for key, value in (expected or {}).items():
        actual = metadata.get("source", {}).get(key)
        if actual != value:
            raise ValueError(
                f"parent source mismatch for {key}: expected {value!r}, got {actual!r}"
            )
    return metadata


def load_parent_artifact(
    root: str | Path,
    *,
    device: str | Any = "cpu",
    expected: Mapping[str, Any] | None = None,
) -> LoadedParentArtifact:
    """Verify first, then load the immutable parent image and latent."""

    from safetensors.torch import load_file

    target = Path(root)
    metadata = verify_parent_artifact(target, expected=expected)
    image_path = target / metadata["files"]["image"]["name"]
    latent_path = target / metadata["files"]["latent"]["name"]
    image = Image.open(image_path).convert("RGB")
    tensors = load_file(str(latent_path), device=str(device))
    latent = tensors[metadata["files"]["latent"].get("key", LATENT_KEY)]
    if tensor_sha256(latent) != metadata["files"]["latent"]["tensor_sha256"]:
        raise ValueError("loaded parent latent hash changed after device transfer")
    return LoadedParentArtifact(
        root=target,
        image=image,
        latent=latent,
        metadata=metadata,
    )


def copy_verified_parent_artifact(
    source: str | Path,
    target: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a verified artefact without changing any byte."""

    source_root = Path(source)
    metadata = verify_parent_artifact(source_root)
    target_root = Path(target)
    if target_root.exists() and any(target_root.iterdir()) and not overwrite:
        raise FileExistsError(f"target parent directory is not empty: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    for entry in metadata["files"].values():
        shutil.copy2(source_root / entry["name"], target_root / entry["name"])
    shutil.copy2(source_root / METADATA_FILENAME, target_root / METADATA_FILENAME)
    return verify_parent_artifact(target_root)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Verify an immutable E035 parent artefact")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    metadata = verify_parent_artifact(args.root)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
