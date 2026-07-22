import numpy as np

from prooftag_qr.controlnet_benchmark import (
    aggregate_controlnet_benchmark,
    best_trial_per_model,
    controlnet_trials,
    select_promotable_controlnet,
)
from prooftag_qr.controlnet_models import (
    CONTROLNET_PROFILES,
    benchmark_profiles,
    control_image_for_profile,
)
from prooftag_qr.qr import generate_qr


def test_sd15_registry_contains_distinct_real_candidates():
    profiles = benchmark_profiles()

    assert {profile.name for profile in profiles} == {
        "dion_sd15",
        "monster_sd15_v1",
        "monster_sd15_v2",
        "nacholmo_sd15_v2",
    }
    assert all(profile.family == "sd15" for profile in profiles)
    assert CONTROLNET_PROFILES["monster_sd15_v2"].subfolder == "v2"
    assert CONTROLNET_PROFILES["monster_sd15_v2"].conditioning_profile == "gray_quiet_zone"
    assert CONTROLNET_PROFILES["monster_sd15_v1"].subfolder == ""
    assert CONTROLNET_PROFILES["monster_sd15_v1"].conditioning_profile == "binary"
    assert (
        CONTROLNET_PROFILES["nacholmo_sd15_v2"].conditioning_profile
        == "nacholmo_extremes_25"
    )
    assert CONTROLNET_PROFILES["nacholmo_sdxl"].status.startswith("work_in_progress")


def test_binary_conditioning_preserves_the_exact_qr():
    blueprint = generate_qr("https://example.prooftag.test/control/binary", "H")

    conditioned = control_image_for_profile(blueprint, "binary")

    assert conditioned is not blueprint.image
    assert np.array_equal(np.asarray(conditioned), np.asarray(blueprint.image))


def test_monster_conditioning_grays_only_the_quiet_zone():
    blueprint = generate_qr("https://example.prooftag.test/control/gray", "H")

    conditioned = np.asarray(control_image_for_profile(blueprint, "gray_quiet_zone"))
    original = np.asarray(blueprint.image)
    module_pixels = conditioned.shape[0] / blueprint.matrix.shape[0]
    core_start = round(blueprint.border * module_pixels)
    core_end = round((blueprint.matrix.shape[0] - blueprint.border) * module_pixels)

    assert np.all(conditioned[:core_start] == 128)
    assert np.all(conditioned[core_end:] == 128)
    assert np.all(conditioned[:, :core_start] == 128)
    assert np.all(conditioned[:, core_end:] == 128)
    assert np.array_equal(
        conditioned[core_start:core_end, core_start:core_end],
        original[core_start:core_end, core_start:core_end],
    )


def test_nacholmo_conditioning_keeps_sparse_extremes_on_neutral_gray():
    blueprint = generate_qr("https://example.prooftag.test/control/nacholmo", "H")

    conditioned = np.asarray(
        control_image_for_profile(blueprint, "nacholmo_extremes_25")
    )
    gray = conditioned[..., 0]
    values = set(np.unique(gray).tolist())
    module_pixels = conditioned.shape[0] / blueprint.matrix.shape[0]
    core_start = round(blueprint.border * module_pixels)
    core_end = round((blueprint.matrix.shape[0] - blueprint.border) * module_pixels)
    extreme_ratio = float(np.isin(gray, (0, 255)).mean())

    assert values == {0, 128, 255}
    assert np.all(gray[:core_start] == 128)
    assert np.all(gray[core_end:] == 128)
    assert np.all(gray[:, :core_start] == 128)
    assert np.all(gray[:, core_end:] == 128)
    assert 0.40 <= extreme_ratio <= 0.55


def test_controlnet_trials_vary_only_the_paired_scales():
    profile = CONTROLNET_PROFILES["nacholmo_sd15_v2"]

    trials = controlnet_trials(profile)

    assert len(trials) == 4
    assert {trial.base_controlnet_scale for trial in trials} == {0.9, 1.1, 1.35, 1.6}
    assert {trial.controlnet_scale for trial in trials} == {0.9, 1.1, 1.35, 1.6}
    assert all(trial.base_controlnet_scale == trial.controlnet_scale for trial in trials)
    assert all(trial.steps == 100 for trial in trials)


def _row(
    trial: str,
    context: str,
    *,
    strict: bool,
    pass_rate: float,
    aesthetic: float,
) -> dict:
    return {
        "status": "ok",
        "trial": trial,
        "context_id": context,
        "controlnet_model_id": trial,
        "controlnet_model_subfolder": "",
        "controlnet_conditioning_profile": "binary",
        "parameters": {"controlnet_scale": 1.35},
        "strict_all": strict,
        "pass_rate": pass_rate,
        "raw_strict_all": False,
        "raw_pass_rate": 0.5,
        "clip_aesthetic": aesthetic,
        "clip_score": aesthetic,
        "raw_clip_aesthetic": aesthetic + 0.1,
        "raw_clip_score": aesthetic + 0.1,
        "context_features": {"raw_module_error_rate": 0.03},
        "module_error_rate": 0.02,
        "duration_seconds": 10.0,
        "peak_gpu_memory_mib": 4096.0,
    }


def test_controlnet_ranking_prioritizes_complete_strict_scan_over_aesthetics():
    rows = [
        _row("strict", "c1", strict=True, pass_rate=1.0, aesthetic=0.2),
        _row("strict", "c2", strict=True, pass_rate=1.0, aesthetic=0.2),
        _row("pretty", "c1", strict=False, pass_rate=0.99, aesthetic=0.9),
        _row("pretty", "c2", strict=False, pass_rate=0.99, aesthetic=0.9),
    ]

    aggregates = aggregate_controlnet_benchmark(rows, expected_contexts=2)

    assert aggregates[0]["trial"] == "strict"
    assert select_promotable_controlnet(aggregates)["trial"] == "strict"


def test_incomplete_controlnet_group_cannot_be_promoted():
    rows = [_row("partial", "c1", strict=True, pass_rate=1.0, aesthetic=1.0)]

    aggregates = aggregate_controlnet_benchmark(rows, expected_contexts=2)

    assert aggregates[0]["complete"] is False
    assert select_promotable_controlnet(aggregates) is None


def test_physical_shortlist_keeps_only_the_best_trial_per_model():
    rows = [
        _row("model-a-scale-low", "c1", strict=True, pass_rate=1.0, aesthetic=0.4),
        _row("model-a-scale-high", "c1", strict=True, pass_rate=1.0, aesthetic=0.8),
        _row("model-b", "c1", strict=True, pass_rate=1.0, aesthetic=0.6),
    ]
    for row in rows[:2]:
        row["controlnet_model_id"] = "model-a"

    aggregates = aggregate_controlnet_benchmark(rows, expected_contexts=1)
    shortlist = best_trial_per_model(aggregates)

    assert [row["trial"] for row in shortlist] == ["model-a-scale-high", "model-b"]
