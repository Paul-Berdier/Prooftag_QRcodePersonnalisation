from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .srpg import SRPGConfig


@dataclass(frozen=True, slots=True)
class SRPGTrial:
    name: str
    source: str
    steps: int = 100
    strength: float = 1.0
    guidance_scale: float = 12.0
    controlnet_scale: float = 1.35
    qr_weight: float = 500.0
    perceptual_weight: float = 3.0
    functional_weight: float = 4.0
    dark_threshold: float = 0.5
    light_threshold: float = 0.5
    max_noise_delta_rms: float = 2.0

    def to_srpg_config(self) -> SRPGConfig:
        return SRPGConfig(
            steps=self.steps,
            strength=self.strength,
            controlnet_scale=self.controlnet_scale,
            qr_weight=self.qr_weight,
            perceptual_weight=self.perceptual_weight,
            functional_weight=self.functional_weight,
            dark_threshold=self.dark_threshold,
            light_threshold=self.light_threshold,
            max_noise_delta_rms=self.max_noise_delta_rms,
            max_mean_absolute_change=0.40,
            min_relative_module_improvement=0.0,
            save_step_previews=False,
        )


def screening_trials() -> tuple[SRPGTrial, ...]:
    """Causal screening plan, not a wasteful Cartesian product.

    The first four points isolate the user's observed step-count effect. The next points
    reproduce the paper/repository ranges before testing one parameter at a time around the
    official profile. Every comparison keeps the Stage-1 image and Stage-2 seed fixed.
    """
    current = SRPGTrial(name="current_steps_100", source="prooftag")
    official = replace(
        current,
        name="official_steps_100",
        source="diffqrcoder",
        guidance_scale=7.5,
        functional_weight=1.0,
        dark_threshold=0.45,
        light_threshold=0.65,
    )
    return (
        replace(current, name="current_steps_40", steps=40),
        replace(current, name="current_steps_60", steps=60),
        replace(current, name="current_steps_80", steps=80),
        current,
        replace(official, name="official_steps_40", steps=40),
        official,
        replace(official, name="official_qr_400", qr_weight=400.0),
        replace(official, name="official_qr_600", qr_weight=600.0),
        replace(official, name="official_qr_1000", qr_weight=1000.0),
        replace(official, name="official_perceptual_0", perceptual_weight=0.0),
        replace(official, name="official_perceptual_2", perceptual_weight=2.0),
        replace(official, name="official_perceptual_5", perceptual_weight=5.0),
        replace(official, name="official_control_105", controlnet_scale=1.05),
        replace(official, name="official_control_150", controlnet_scale=1.50),
        replace(official, name="official_strength_085", strength=0.85),
        replace(official, name="official_noise_cap_1", max_noise_delta_rms=1.0),
        replace(official, name="official_noise_cap_4", max_noise_delta_rms=4.0),
    )


def trial_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Rank scannability before aesthetics and speed."""
    return (
        row.get("status") != "ok",
        -int(bool(row.get("strict_all"))),
        -float(row.get("pass_rate", 0.0)),
        -float(row.get("original_pass_rate", 0.0)),
        float(row.get("module_error_rate", 1.0)),
        float(row.get("mean_absolute_change", 1.0)),
        float(row.get("duration_seconds", float("inf"))),
    )


def aggregate_confirmation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(str(row["trial"]), []).append(row)
    aggregates = []
    for trial, items in grouped.items():
        pass_rates = [float(item["pass_rate"]) for item in items]
        aggregates.append(
            {
                "trial": trial,
                "cases": len(items),
                "all_strict": all(bool(item["strict_all"]) for item in items),
                "worst_pass_rate": min(pass_rates),
                "mean_pass_rate": sum(pass_rates) / len(pass_rates),
                "mean_module_error_rate": sum(float(item["module_error_rate"]) for item in items)
                / len(items),
                "mean_absolute_change": sum(float(item["mean_absolute_change"]) for item in items)
                / len(items),
                "mean_duration_seconds": sum(float(item["duration_seconds"]) for item in items)
                / len(items),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            -int(row["all_strict"]),
            -row["worst_pass_rate"],
            -row["mean_pass_rate"],
            row["mean_module_error_rate"],
            row["mean_absolute_change"],
            row["mean_duration_seconds"],
        ),
    )
