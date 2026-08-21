from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .config import Settings


def _huggingface_file_coordinates(model_url: str) -> tuple[str, str]:
    parsed = urlparse(model_url)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError("a pinned single-file model must use a huggingface.co URL")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] not in {"blob", "resolve"}:
        raise ValueError(
            "the Hugging Face single-file URL must contain /blob/<ref>/ or "
            "/resolve/<ref>/"
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[4:])


def resolve_single_file_sources(settings: Settings) -> tuple[str | Path, str | Path]:
    """Resolve checkpoint and Diffusers config independently at pinned commits."""

    checkpoint: str | Path = settings.base_model_id
    config: str | Path = settings.base_model_config_id
    if not settings.base_model_revision and not settings.base_model_config_revision:
        return checkpoint, config

    from huggingface_hub import hf_hub_download, snapshot_download

    if settings.base_model_revision:
        model_path = Path(settings.base_model_id)
        if model_path.is_file():
            checkpoint = model_path
        else:
            repo_id, filename = _huggingface_file_coordinates(settings.base_model_id)
            checkpoint = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=settings.base_model_revision,
                cache_dir=settings.model_cache_dir,
            )
    if settings.base_model_config_revision:
        config = snapshot_download(
            repo_id=settings.base_model_config_id,
            revision=settings.base_model_config_revision,
            cache_dir=settings.model_cache_dir,
            allow_patterns=[
                "*.json",
                "**/*.json",
                "*.txt",
                "**/*.txt",
                "*.model",
                "**/*.model",
            ],
        )
    return checkpoint, config
