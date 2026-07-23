from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_AESTHETIC_WEIGHTS_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/raw/refs/heads/main/sa_0_4_vit_b_32_linear.pth"
)


@dataclass(frozen=True, slots=True)
class CLIPQualityScore:
    clip_similarity: float
    clip_score: float
    clip_aesthetic: float


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
        aesthetic_weights_url: str = DEFAULT_AESTHETIC_WEIGHTS_URL,
        device: str = "cpu",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.aesthetic_weights_url = aesthetic_weights_url
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._aesthetic: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._model, self._processor, self._aesthetic
        import torch
        from transformers import CLIPModel, CLIPProcessor

        quality_cache = self.cache_dir / "clip-quality"
        quality_cache.mkdir(parents=True, exist_ok=True)
        model = CLIPModel.from_pretrained(self.model_id, cache_dir=quality_cache)
        processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=quality_cache)
        model.requires_grad_(False).eval().to(self.device)

        weight_path = quality_cache / "sa_0_4_vit_b_32_linear.pth"
        if not weight_path.exists():
            partial = weight_path.with_suffix(".partial")
            urlretrieve(self.aesthetic_weights_url, partial)
            partial.replace(weight_path)
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
        return CLIPQualityScore(
            clip_similarity=similarity,
            clip_score=clip_score_from_similarity(similarity),
            clip_aesthetic=aesthetic_score,
        )
