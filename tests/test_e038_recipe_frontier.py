from __future__ import annotations

from pathlib import Path

from prooftag_qr.e038_recipe_frontier import (
    DEFAULT_RECIPES,
    E038Config,
    OBJECTIVE_RECIPES,
    RADIUS_RECIPES,
    recipe_catalog,
)


def test_e038_gamma_and_iteration_contract_are_frozen() -> None:
    config = E038Config()
    assert config.gamma == 1000.0
    assert config.max_iterations == 4
    assert config.lpips_weight == 0.01


def test_e038_has_six_radius_recipes_and_four_objective_recipes() -> None:
    assert len(RADIUS_RECIPES) == 6
    assert len(OBJECTIVE_RECIPES) == 4
    assert len(DEFAULT_RECIPES) == 10
    assert [round(recipe.latent_radius_rms, 3) for recipe in RADIUS_RECIPES] == [
        0.075,
        0.100,
        0.125,
        0.150,
        0.200,
        0.300,
    ]


def test_e038_recipe_catalog_is_unique_and_contains_hybrids() -> None:
    catalog = recipe_catalog()
    names = [item["name"] for item in catalog]
    assert len(names) == len(set(names)) == 10
    assert "e038_hybrid_r100" in names
    assert "e038_hybrid_r150" in names
    assert "e038_full_r100" in names
    assert "e038_robust_r100" in names


def test_e038_runner_has_ssr_first_visual_gate_contract() -> None:
    root = Path(__file__).parents[1]
    text = (root / "prooftag_qr/e038_recipe_frontier.py").read_text(encoding="utf-8")
    assert '"visual_guard_pass"' in text
    assert '"qr_verify_exact_presets"' in text
    assert '"original_exact"' in text
    assert "project_latent_candidate" in text
    assert "config.gamma * gradient" in text
    assert "E038 keeps gamma fixed at 1000" in text
