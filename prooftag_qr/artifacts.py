from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings


class ArtifactStore:
    def save_image(self, run_id: str, image: Image.Image) -> str:
        raise NotImplementedError

    def save_variant(self, run_id: str, variant: str, image: Image.Image) -> str:
        raise NotImplementedError

    def load_image(self, location: str) -> Image.Image:
        raise NotImplementedError

    def save_metadata(self, run_id: str, name: str, payload: Any) -> str:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_image(self, run_id: str, image: Image.Image) -> str:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "final.png"
        image.save(path, format="PNG", optimize=True)
        return str(path)

    def save_variant(self, run_id: str, variant: str, image: Image.Image) -> str:
        variant_dir = self.root / run_id / "variants"
        variant_dir.mkdir(parents=True, exist_ok=True)
        path = variant_dir / f"{variant}.png"
        image.save(path, format="PNG", optimize=True)
        return str(path)

    def load_image(self, location: str) -> Image.Image:
        with Image.open(location) as image:
            return image.convert("RGB").copy()

    def save_metadata(self, run_id: str, name: str, payload: Any) -> str:
        metadata_dir = self.root / run_id / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


class S3ArtifactStore(ArtifactStore):
    def __init__(self, settings: Settings):
        import boto3

        if not settings.s3_access_key or not settings.s3_secret_key:
            raise RuntimeError("S3 artifact storage requires access and secret keys")
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
        )
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

    def save_image(self, run_id: str, image: Image.Image) -> str:
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=True)
        stream.seek(0)
        key = f"runs/{run_id}/final.png"
        self.client.upload_fileobj(stream, self.bucket, key, ExtraArgs={"ContentType": "image/png"})
        return f"s3://{self.bucket}/{key}"

    def save_variant(self, run_id: str, variant: str, image: Image.Image) -> str:
        stream = io.BytesIO()
        image.save(stream, format="PNG", optimize=True)
        stream.seek(0)
        key = f"runs/{run_id}/variants/{variant}.png"
        self.client.upload_fileobj(stream, self.bucket, key, ExtraArgs={"ContentType": "image/png"})
        return f"s3://{self.bucket}/{key}"

    def load_image(self, location: str) -> Image.Image:
        prefix = f"s3://{self.bucket}/"
        if not location.startswith(prefix):
            raise ValueError(f"Artifact does not belong to bucket {self.bucket}")
        key = location[len(prefix) :]
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        with Image.open(io.BytesIO(response["Body"].read())) as image:
            return image.convert("RGB").copy()

    def save_metadata(self, run_id: str, name: str, payload: Any) -> str:
        key = f"runs/{run_id}/metadata/{name}.json"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"


def build_artifact_store(settings: Settings) -> ArtifactStore:
    if settings.artifact_store == "s3":
        return S3ArtifactStore(settings)
    return LocalArtifactStore(settings.artifacts_dir)
