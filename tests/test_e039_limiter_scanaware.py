from __future__ import annotations

from pathlib import Path

from prooftag_qr.e039_limiter_scanaware import DEFAULT_RECIPES, E039Config, recipe_catalog


def test_e039_gamma_and_lpips_contract_are_frozen() -> None:
    config = E039Config()
    assert config.gamma == 1000.0
    assert config.lpips_weight == 0.01
    assert config.max_lpips_for_ranking == 0.050


def test_e039_has_ten_preregistered_recipes() -> None:
    assert len(DEFAULT_RECIPES) == 10
    assert len(recipe_catalog()) == 10
    assert len({recipe.name for recipe in DEFAULT_RECIPES}) == 10


def test_e039_iteration_and_profile_grid() -> None:
    e038 = [r for r in DEFAULT_RECIPES if r.profile == "e038_hybrid"]
    scan = [r for r in DEFAULT_RECIPES if r.profile == "scanaware_v2"]
    assert [r.max_iterations for r in e038] == [4, 6, 8, 12]
    assert len(scan) == 6
    assert any(r.latent_radius_rms == 0.300 and r.max_iterations == 12 for r in scan)


def test_e039_logs_every_backtrack_and_blocker() -> None:
    root = Path(__file__).parents[1]
    text = (root / "prooftag_qr/e039_limiter_scanaware.py").read_text(encoding="utf-8")
    for token in (
        "rejection-log.json",
        "blocker-summary.json",
        "rejected_by_lpips_budget",
        "rejected_by_objective_nonincrease",
        "project_latent_candidate",
        "config.gamma * gradient",
        "E039 keeps gamma fixed at 1000",
        "_downscale_restore(images, 0.50)",
        "brightness_85",
        "contrast_85",
    ):
        assert token in text
