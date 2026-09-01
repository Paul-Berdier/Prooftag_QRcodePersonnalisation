"""E040 model bridge: research-safe access to trained Prooftag advisor/surrogate artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

DEFAULT_ADVISOR_ROOT = Path("/data/e031-prospective-stage2-models")
SURROGATE_FILENAME = "scan-surrogate.research-only.torchscript.pt"
SURROGATE_CARD_FILENAME = "surrogate-card.json"


def _canonical_sha(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def discover_latest_advisor(root: Path | None = None) -> Path | None:
    explicit = os.environ.get("PROOFTAG_E040_ADVISOR_MODEL", "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    root = Path(root or os.environ.get("PROOFTAG_E040_ADVISOR_ROOT", DEFAULT_ADVISOR_ROOT))
    if not root.is_dir():
        return None
    candidates = [path for path in root.glob("*.joblib") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def advisor_preview(
    *,
    prompt: str,
    payload_length: int,
    error_correction: str = "M",
    qr_context: Mapping[str, Any] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Use the latest E026/E031 advisor as a *recommendation*, never as a delivery gate."""
    path = discover_latest_advisor()
    if path is None:
        return {"available": False, "reason": "no advisor joblib found"}

    from .lab import _legacy_laboratory_profiles
    from .parameter_advisor import E026ParameterAdvisor, RecipeCandidate

    advisor = E026ParameterAdvisor.load(path)
    candidates: list[RecipeCandidate] = []
    for profile in _legacy_laboratory_profiles():
        if profile.get("backend") != "controlnet":
            continue
        if profile.get("output_variant") not in {"srpg", "srmpgd"}:
            continue
        configuration = {
            key: value
            for key, value in profile.items()
            if key not in {"name", "description", "enabled"}
        }
        signature = _canonical_sha(configuration)
        candidates.append(
            RecipeCandidate(
                id=f"e040-{signature[:10]}",
                method_id=str(profile.get("id") or "unknown"),
                configuration=configuration,
                signature=signature,
                observations=0,
            )
        )
    if not candidates:
        return {"available": False, "path": str(path), "reason": "no candidate profile"}

    recommendations = advisor.recommend(
        prompt=prompt,
        candidates=candidates,
        payload_length=payload_length,
        error_correction=error_correction,
        qr_context=dict(qr_context or {}),
        scan_probability_threshold=0.80,
        limit=min(limit, len(candidates)),
    )
    return {
        "available": True,
        "path": str(path),
        "training_report": dict(advisor.training_report),
        "recommendations": [item.to_dict() for item in recommendations],
        "note": "advisor recommends parameters only; QR-Verify remains authoritative",
    }


def discover_surrogate() -> tuple[Path | None, Path | None]:
    explicit = os.environ.get("PROOFTAG_E016_SURROGATE_MODEL", "").strip()
    if explicit:
        model = Path(explicit)
        card = Path(os.environ.get("PROOFTAG_E016_SURROGATE_CARD", model.with_name(SURROGATE_CARD_FILENAME)))
        return (model if model.is_file() else None, card if card.is_file() else None)

    roots = [
        Path(os.environ.get("PROOFTAG_E016_SURROGATE_ROOT", "/data/notebook-runs")),
        Path("/data/e016"),
        Path("/data/e016-surrogate"),
    ]
    pairs: list[tuple[Path, Path | None]] = []
    for root in roots:
        if not root.exists():
            continue
        patterns = [SURROGATE_FILENAME, f"*/{SURROGATE_FILENAME}", f"*/*/{SURROGATE_FILENAME}"]
        for pattern in patterns:
            for model in root.glob(pattern):
                if not model.is_file():
                    continue
                card = model.with_name(SURROGATE_CARD_FILENAME)
                pairs.append((model, card if card.is_file() else None))
    if not pairs:
        return None, None
    return max(pairs, key=lambda pair: pair[0].stat().st_mtime_ns)


def surrogate_status() -> dict[str, Any]:
    model, card = discover_surrogate()
    if model is None:
        return {"available": False, "research_usable": False, "reason": "no E016 TorchScript found"}
    card_data: dict[str, Any] = {}
    if card is not None:
        try:
            card_data = json.loads(card.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - audit should survive malformed legacy cards
            card_data = {"card_error": repr(exc)}
    promotion = card_data.get("promotion") if isinstance(card_data, dict) else None
    research_usable = bool((promotion or {}).get("research_usable"))
    return {
        "available": True,
        "model_path": str(model),
        "card_path": str(card) if card else None,
        "research_usable": research_usable,
        "production_usable": bool((promotion or {}).get("production_usable")),
        "promotion": promotion,
        "note": "E016 is research-only unless its own card promotes it; it never replaces real decoders",
    }


def score_surrogate_images(images: Mapping[str, Image.Image]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score rasters with E016. Returns no score unless the E016 card says research_usable."""
    status = surrogate_status()
    if not status.get("research_usable"):
        return {}, status

    import numpy as np
    import torch
    import torch.nn.functional as F

    model = torch.jit.load(str(status["model_path"]), map_location="cpu").eval()
    results: dict[str, Any] = {}
    with torch.no_grad():
        for name, image in images.items():
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            # Match E016's 256x256 input contract while keeping this path deterministic.
            tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).reshape(-1).cpu().numpy().astype(float).tolist()
            results[name] = {
                "decoder_probabilities": probabilities,
                "mean_success_probability": float(sum(probabilities) / max(1, len(probabilities))),
                "min_success_probability": float(min(probabilities)) if probabilities else None,
            }
    return results, status
