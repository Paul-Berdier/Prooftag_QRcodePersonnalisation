from __future__ import annotations
from pathlib import Path
from prooftag_qr.e040_checkpoint_frontier import DEFAULT_RADII, DEFAULT_RECIPES, E039Config, recipe_catalog


def test_e040_contract() -> None:
    assert E039Config().gamma == 1000.0
    assert list(DEFAULT_RADII) == [0.150, 0.175, 0.200, 0.225, 0.250]
    assert len(DEFAULT_RECIPES) == 5
    assert all(item.max_iterations == 8 for item in DEFAULT_RECIPES)
    assert len(recipe_catalog()) == 5


def test_e040_source_is_checkpoint_aware() -> None:
    root = Path(__file__).parents[1]
    text = (root / "prooftag_qr/e040_checkpoint_frontier.py").read_text(encoding="utf-8")
    assert "iteration-{iteration:03d}.safetensors" in text
    assert "checkpoint-comparison.csv" in text
    assert "99-FINAL-QR.png" in text
    assert "score_surrogate_images" in text
    assert "advisor_preview" in text
    assert "E040 keeps gamma fixed at 1000" in text
