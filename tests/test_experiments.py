from prooftag_qr.experiments import (
    aggregate_confirmation,
    screening_trials,
    trial_rank_key,
)


def test_screening_plan_is_causal_and_contains_paper_ranges():
    trials = screening_trials()
    names = [trial.name for trial in trials]

    assert len(names) == len(set(names))
    assert {40, 60, 80, 100}.issubset({trial.steps for trial in trials})
    assert {400.0, 500.0, 600.0, 1000.0}.issubset({trial.qr_weight for trial in trials})
    official = next(trial for trial in trials if trial.name == "official_steps_100")
    assert official.guidance_scale == 7.5
    assert official.dark_threshold == 0.45
    assert official.light_threshold == 0.65


def test_trial_ranking_prioritizes_strict_scan_over_visual_change():
    strict = {
        "status": "ok",
        "strict_all": True,
        "pass_rate": 1.0,
        "original_pass_rate": 1.0,
        "module_error_rate": 0.1,
        "mean_absolute_change": 0.5,
        "duration_seconds": 100,
    }
    pretty_but_not_strict = {
        **strict,
        "strict_all": False,
        "pass_rate": 0.99,
        "mean_absolute_change": 0.01,
    }

    assert trial_rank_key(strict) < trial_rank_key(pretty_but_not_strict)


def test_confirmation_uses_worst_case_before_mean():
    rows = [
        {
            "trial": "stable",
            "status": "ok",
            "strict_all": False,
            "pass_rate": 0.95,
            "module_error_rate": 0.02,
            "mean_absolute_change": 0.10,
            "duration_seconds": 10,
        },
        {
            "trial": "stable",
            "status": "ok",
            "strict_all": True,
            "pass_rate": 1.0,
            "module_error_rate": 0.01,
            "mean_absolute_change": 0.10,
            "duration_seconds": 10,
        },
        {
            "trial": "fragile",
            "status": "ok",
            "strict_all": False,
            "pass_rate": 0.90,
            "module_error_rate": 0.01,
            "mean_absolute_change": 0.01,
            "duration_seconds": 5,
        },
        {
            "trial": "fragile",
            "status": "ok",
            "strict_all": True,
            "pass_rate": 1.0,
            "module_error_rate": 0.01,
            "mean_absolute_change": 0.01,
            "duration_seconds": 5,
        },
    ]

    aggregates = aggregate_confirmation(rows)

    assert aggregates[0]["trial"] == "stable"
    assert aggregates[0]["worst_pass_rate"] == 0.95
