from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import Settings

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CLIP_MODEL_REVISION = "092a3b7e31726acc3a0207eea00f6040ac8b03a7"
DEFAULT_AESTHETIC_WEIGHTS_REVISION = "6d122adad522ab246644d9dc1c6d7a3810ee255f"
DEFAULT_AESTHETIC_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/"
    f"{DEFAULT_AESTHETIC_WEIGHTS_REVISION}/sa_0_4_vit_b_32_linear.pth"
)
DEFAULT_AESTHETIC_WEIGHTS_SHA256 = (
    "c7b14cead230694acc7b9447974d3cad78003c72da032e402a303b6c2429e85f"
)
DEFAULT_HPS_PACKAGE_NAME = "hpsv2"
DEFAULT_HPS_PACKAGE_VERSION = "1.2.0"
DEFAULT_HPS_SOURCE_REVISION = "866735ecaae999fa714bd9edfa05aa2672669ee3"
DEFAULT_HPS_MODEL_VERSION = "v2.1"
DEFAULT_HPS_CHECKPOINT_REPO = "xswu/HPSv2"
DEFAULT_HPS_CHECKPOINT_REVISION = "697403c78157020a1ae59d23f111aa58ced35b0a"
DEFAULT_HPS_CHECKPOINT_FILENAME = "HPS_v2.1_compressed.pt"
DEFAULT_HPS_CHECKPOINT_SHA256 = (
    "c57a38fb4a2f7e7c15bf00da2ea377cdf165448b4dd1052a484c215a998c9837"
)
logger = logging.getLogger(__name__)


class QualityScoringError(RuntimeError):
    """A requested quality metric could not be produced reliably."""


class QualityProvenanceError(QualityScoringError):
    """A model, package, or weight does not match its configured immutable pin."""


def _normalized_sha(value: str, *, label: str, length: int) -> str:
    normalized = str(value).strip().lower()
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", normalized):
        raise ValueError(f"{label} must be a {length}-character lowercase hexadecimal SHA")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_distribution_source(package_name: str) -> tuple[str | None, str | None, str | None]:
    """Return installed version, PEP 610 VCS commit and an inspection error."""
    try:
        package = distribution(package_name)
    except PackageNotFoundError:
        return None, None, f"package {package_name!r} is not installed"
    direct_url_text = package.read_text("direct_url.json")
    if not direct_url_text:
        return package.version, None, "direct_url.json is absent"
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        return package.version, None, f"direct_url.json is invalid: {exc}"
    commit = str(direct_url.get("vcs_info", {}).get("commit_id", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        return package.version, None, "PEP 610 VCS commit is absent or invalid"
    return package.version, commit, None


@dataclass(frozen=True, slots=True)
class CLIPQualityScore:
    clip_similarity: float
    clip_score: float
    clip_aesthetic: float
    hpsv2_1: float | None = None


def clip_score_from_similarity(similarity: float) -> float:
    """CLIPScore's published 2.5 * max(cosine, 0) rescaling."""
    return 2.5 * max(float(similarity), 0.0)


def project_embedding(
    embedding: np.ndarray, dimensions: int = 16, seed: int = 20260721
) -> list[float]:
    """Deterministic low-dimensional context vector for the parameter advisor."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    generator = np.random.default_rng(seed)
    projection = generator.normal(
        0.0,
        1.0 / np.sqrt(dimensions),
        size=(vector.size, dimensions),
    ).astype(np.float32)
    return (vector @ projection).astype(float).tolist()


class CLIPQualityScorer:
    """CPU scorer for prompt alignment and LAION's CLIP aesthetic predictor.

    CPU is intentional: diffusion and SRPG keep exclusive ownership of GPU memory on the
    20 GiB card. Models and weights are cached on the existing model-cache PVC.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        model_id: str = DEFAULT_CLIP_MODEL,
        model_revision: str = DEFAULT_CLIP_MODEL_REVISION,
        aesthetic_weights_url: str = DEFAULT_AESTHETIC_WEIGHTS_URL,
        aesthetic_weights_sha256: str = DEFAULT_AESTHETIC_WEIGHTS_SHA256,
        device: str = "cpu",
        hps_enabled: bool = False,
        hps_fail_closed: bool = False,
        hps_package_name: str = DEFAULT_HPS_PACKAGE_NAME,
        hps_package_version: str = DEFAULT_HPS_PACKAGE_VERSION,
        hps_source_revision: str = DEFAULT_HPS_SOURCE_REVISION,
        hps_model_version: str = DEFAULT_HPS_MODEL_VERSION,
        hps_checkpoint_repo: str = DEFAULT_HPS_CHECKPOINT_REPO,
        hps_checkpoint_revision: str = DEFAULT_HPS_CHECKPOINT_REVISION,
        hps_checkpoint_filename: str = DEFAULT_HPS_CHECKPOINT_FILENAME,
        hps_checkpoint_sha256: str = DEFAULT_HPS_CHECKPOINT_SHA256,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.model_revision = _normalized_sha(
            model_revision, label="CLIP model revision", length=40
        )
        self.aesthetic_weights_url = aesthetic_weights_url
        self.aesthetic_weights_sha256 = _normalized_sha(
            aesthetic_weights_sha256,
            label="aesthetic weights SHA-256",
            length=64,
        )
        self.device = device
        self.hps_enabled = hps_enabled
        self.hps_fail_closed = hps_fail_closed
        self.hps_package_name = hps_package_name
        self.hps_package_version = hps_package_version
        self.hps_source_revision = _normalized_sha(
            hps_source_revision, label="HPS source revision", length=40
        )
        self.hps_model_version = hps_model_version
        self.hps_checkpoint_repo = hps_checkpoint_repo
        self.hps_checkpoint_revision = _normalized_sha(
            hps_checkpoint_revision,
            label="HPS checkpoint revision",
            length=40,
        )
        self.hps_checkpoint_filename = hps_checkpoint_filename
        self.hps_checkpoint_sha256 = _normalized_sha(
            hps_checkpoint_sha256,
            label="HPS checkpoint SHA-256",
            length=64,
        )
        self._model: Any | None = None
        self._processor: Any | None = None
        self._aesthetic: Any | None = None
        self._clip_effective_revision: str | None = None
        self._aesthetic_effective_sha256: str | None = None
        self._hps_package_effective_version: str | None = None
        self._hps_source_effective_revision: str | None = None
        self._hps_package_inspection_error: str | None = None
        self._hps_checkpoint_path: Path | None = None
        self._hps_checkpoint_effective_sha256: str | None = None

    @property
    def aesthetic_weights_path(self) -> Path:
        return self.cache_dir / "clip-quality" / "sa_0_4_vit_b_32_linear.pth"

    def _inspect_hps_installation(self) -> None:
        if self._hps_package_effective_version is not None or (
            self._hps_package_inspection_error is not None
        ):
            return
        version, revision, error = _installed_distribution_source(self.hps_package_name)
        self._hps_package_effective_version = version
        self._hps_source_effective_revision = revision
        self._hps_package_inspection_error = error

    def _verify_hps_installation(self) -> None:
        self._inspect_hps_installation()
        if self._hps_package_inspection_error:
            raise QualityProvenanceError(self._hps_package_inspection_error)
        if self._hps_package_effective_version != self.hps_package_version:
            raise QualityProvenanceError(
                "HPS package version mismatch: "
                f"expected {self.hps_package_version}, got "
                f"{self._hps_package_effective_version}"
            )
        if self._hps_source_effective_revision != self.hps_source_revision:
            raise QualityProvenanceError(
                "HPS source revision mismatch: "
                f"expected {self.hps_source_revision}, got "
                f"{self._hps_source_effective_revision}"
            )

    def _ensure_hps_checkpoint(self) -> Path:
        if self._hps_checkpoint_path is not None:
            return self._hps_checkpoint_path
        self._verify_hps_installation()
        from huggingface_hub import hf_hub_download

        configured_hub_cache = os.environ.get("HF_HUB_CACHE")
        if configured_hub_cache:
            hub_cache = Path(configured_hub_cache)
        else:
            hf_home = Path(
                os.environ.get("HF_HOME", str(self.cache_dir / "huggingface"))
            )
            hub_cache = hf_home / "hub"

        checkpoint = Path(
            hf_hub_download(
                repo_id=self.hps_checkpoint_repo,
                filename=self.hps_checkpoint_filename,
                revision=self.hps_checkpoint_revision,
                cache_dir=hub_cache,
            )
        )
        effective_sha256 = _sha256_file(checkpoint)
        self._hps_checkpoint_effective_sha256 = effective_sha256
        if effective_sha256 != self.hps_checkpoint_sha256:
            raise QualityProvenanceError(
                "HPS checkpoint SHA-256 mismatch: "
                f"expected {self.hps_checkpoint_sha256}, got {effective_sha256}"
            )
        self._hps_checkpoint_path = checkpoint
        return checkpoint

    def provenance(self) -> dict[str, Any]:
        """Return requested pins and the effective artifacts verified so far."""
        if self.hps_enabled:
            self._inspect_hps_installation()
        hps_package_verified = None
        if self._hps_package_effective_version is not None:
            hps_package_verified = (
                self._hps_package_inspection_error is None
                and self._hps_package_effective_version == self.hps_package_version
                and self._hps_source_effective_revision == self.hps_source_revision
            )
        return {
            "policy": "fail_closed" if self.hps_fail_closed else "metric_errors_omitted",
            "clip": {
                "model_id": self.model_id,
                "requested_revision": self.model_revision,
                "effective_revision": self._clip_effective_revision,
                "revision_verified": (
                    self._clip_effective_revision == self.model_revision
                    if self._clip_effective_revision is not None
                    else None
                ),
            },
            "clip_aesthetic": {
                "weights_url": self.aesthetic_weights_url,
                "expected_sha256": self.aesthetic_weights_sha256,
                "effective_sha256": self._aesthetic_effective_sha256,
                "sha256_verified": (
                    self._aesthetic_effective_sha256 == self.aesthetic_weights_sha256
                    if self._aesthetic_effective_sha256 is not None
                    else None
                ),
            },
            "hpsv2_1": {
                "enabled": self.hps_enabled,
                "model_version": self.hps_model_version,
                "package_name": self.hps_package_name,
                "expected_package_version": self.hps_package_version,
                "effective_package_version": self._hps_package_effective_version,
                "expected_source_revision": self.hps_source_revision,
                "effective_source_revision": self._hps_source_effective_revision,
                "package_verified": hps_package_verified,
                "package_inspection_error": self._hps_package_inspection_error,
                "checkpoint_repo": self.hps_checkpoint_repo,
                "checkpoint_filename": self.hps_checkpoint_filename,
                "requested_checkpoint_revision": self.hps_checkpoint_revision,
                "expected_checkpoint_sha256": self.hps_checkpoint_sha256,
                "effective_checkpoint_sha256": self._hps_checkpoint_effective_sha256,
                "checkpoint_verified": (
                    self._hps_checkpoint_effective_sha256 == self.hps_checkpoint_sha256
                    if self._hps_checkpoint_effective_sha256 is not None
                    else None
                ),
            },
        }

    def _hps_score(self, image: Image.Image, prompt: str) -> float | None:
        if not self.hps_enabled:
            return None
        try:
            import hpsv2
        except ImportError:
            raise QualityProvenanceError("hpsv2 is enabled but cannot be imported") from None
        os.environ.setdefault("HPS_ROOT", str(self.cache_dir / "hpsv2"))
        try:
            checkpoint = self._ensure_hps_checkpoint()
            # The public hpsv2.score wrapper does not forward a checkpoint
            # path. Calling img_score directly is required to avoid its
            # unpinned hf_hub_download(..., revision=None) fallback.
            if getattr(hpsv2, "__path__", None) is None:
                raise QualityProvenanceError("hpsv2 is not an installed package")
            hps_img_score = importlib.import_module("hpsv2.img_score")
            hps_img_score.device = "cpu"
            result = hps_img_score.score(
                image.convert("RGB"),
                prompt,
                cp=str(checkpoint),
                hps_version=self.hps_model_version,
            )
            values = np.asarray(result, dtype=np.float64).reshape(-1)
            if values.size and np.isfinite(values[0]):
                return float(values[0])
            if self.hps_fail_closed:
                raise QualityScoringError("HPS v2.1 returned no finite score")
            return None
        except QualityProvenanceError:
            raise
        except Exception as exc:
            logger.warning(
                "hpsv2_scoring_failed",
                extra={"error": f"{type(exc).__name__}: {exc}"},
            )
            if self.hps_fail_closed:
                raise QualityScoringError(
                    f"HPS v2.1 scoring failed: {type(exc).__name__}: {exc}"
                ) from exc
            return None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._model, self._processor, self._aesthetic
        import torch
        from transformers import CLIPModel, CLIPProcessor

        quality_cache = self.cache_dir / "clip-quality"
        quality_cache.mkdir(parents=True, exist_ok=True)
        model = CLIPModel.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            cache_dir=quality_cache,
        )
        processor = CLIPProcessor.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            cache_dir=quality_cache,
        )
        effective_revision = str(getattr(model.config, "_commit_hash", "")).strip().lower()
        self._clip_effective_revision = effective_revision or None
        if effective_revision != self.model_revision:
            raise QualityProvenanceError(
                "CLIP model revision mismatch: "
                f"expected {self.model_revision}, got {effective_revision or 'unknown'}"
            )
        model.requires_grad_(False).eval().to(self.device)

        weight_path = self.aesthetic_weights_path
        effective_sha256 = _sha256_file(weight_path) if weight_path.is_file() else None
        if effective_sha256 != self.aesthetic_weights_sha256:
            partial = weight_path.with_suffix(".partial")
            partial.unlink(missing_ok=True)
            try:
                urlretrieve(self.aesthetic_weights_url, partial)
                downloaded_sha256 = _sha256_file(partial)
                if downloaded_sha256 != self.aesthetic_weights_sha256:
                    raise QualityProvenanceError(
                        "downloaded aesthetic weights SHA-256 mismatch: "
                        f"expected {self.aesthetic_weights_sha256}, got "
                        f"{downloaded_sha256}"
                    )
                partial.replace(weight_path)
            finally:
                partial.unlink(missing_ok=True)
            effective_sha256 = _sha256_file(weight_path)
        self._aesthetic_effective_sha256 = effective_sha256
        if effective_sha256 != self.aesthetic_weights_sha256:
            raise QualityProvenanceError(
                "aesthetic weights SHA-256 mismatch after installation: "
                f"expected {self.aesthetic_weights_sha256}, got {effective_sha256}"
            )
        aesthetic = torch.nn.Linear(model.config.projection_dim, 1)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        aesthetic.load_state_dict(state)
        aesthetic.requires_grad_(False).eval().to(self.device)

        self._model = model
        self._processor = processor
        self._aesthetic = aesthetic
        return model, processor, aesthetic

    def embeddings(self, image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        model, processor, _ = self._load()
        inputs = processor(
            text=[prompt],
            images=[image.convert("RGB")],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            image_embedding = functional.normalize(output.image_embeds.float(), dim=-1)
            text_embedding = functional.normalize(output.text_embeds.float(), dim=-1)
        return (
            image_embedding[0].cpu().numpy(),
            text_embedding[0].cpu().numpy(),
        )

    def text_embedding(self, prompt: str) -> np.ndarray:
        """Return a normalized CLIP text vector without processing a dummy image."""
        import torch
        import torch.nn.functional as functional

        model, processor, _ = self._load()
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            embedding = model.get_text_features(**text_inputs)
            embedding = functional.normalize(embedding.float(), dim=-1)
        return embedding[0].cpu().numpy()

    def score(self, image: Image.Image, prompt: str) -> CLIPQualityScore:
        import torch

        _, _, aesthetic = self._load()
        image_embedding, text_embedding = self.embeddings(image, prompt)
        similarity = float(np.dot(image_embedding, text_embedding))
        with torch.inference_mode():
            tensor = torch.as_tensor(image_embedding, device=self.device).unsqueeze(0)
            aesthetic_score = float(aesthetic(tensor).squeeze().item())
        if not np.isfinite(similarity) or not np.isfinite(aesthetic_score):
            raise QualityScoringError("CLIP/CLIP-Aesthetic returned a non-finite score")
        return CLIPQualityScore(
            clip_similarity=similarity,
            clip_score=clip_score_from_similarity(similarity),
            clip_aesthetic=aesthetic_score,
            hpsv2_1=self._hps_score(image, prompt),
        )


def quality_scorer_from_settings(
    settings: Settings,
    *,
    device: str = "cpu",
    scorer_class: type[CLIPQualityScorer] = CLIPQualityScorer,
) -> CLIPQualityScorer:
    """Construct the API scorer exclusively from its audited configuration."""
    return scorer_class(
        settings.model_cache_dir,
        model_id=settings.quality_clip_model_id,
        model_revision=settings.quality_clip_model_revision,
        aesthetic_weights_url=settings.quality_aesthetic_weights_url,
        aesthetic_weights_sha256=settings.quality_aesthetic_weights_sha256,
        device=device,
        hps_enabled=settings.lab_hps_scoring_enabled,
        hps_fail_closed=settings.lab_quality_scoring_fail_closed,
        hps_package_name=settings.quality_hps_package_name,
        hps_package_version=settings.quality_hps_package_version,
        hps_source_revision=settings.quality_hps_source_revision,
        hps_model_version=settings.quality_hps_model_version,
        hps_checkpoint_repo=settings.quality_hps_checkpoint_repo,
        hps_checkpoint_revision=settings.quality_hps_checkpoint_revision,
        hps_checkpoint_filename=settings.quality_hps_checkpoint_filename,
        hps_checkpoint_sha256=settings.quality_hps_checkpoint_sha256,
    )
