from __future__ import annotations

from dataclasses import replace
from typing import Any

from .controlnet_models import ControlNetProfile
from .experiments import SRPGTrial


def controlnet_trials(
    profile: ControlNetProfile,
    scales: tuple[float, ...] = (0.90, 1.10, 1.35, 1.60),
) -> tuple[SRPGTrial, ...]:
    """Create a controlled scale sweep for one SD1.5 ControlNet.

    Stage-1 and Stage-2 receive the same conditioning scale. All other parameters remain fixed so
    model identity and scale are the only manipulated variables.
    """
    reference = SRPGTrial(
        name="reference",
        source="e008-controlnet-bakeoff",
        base_steps=16,
        base_strength=0.90,
        base_guidance_scale=7.5,
        steps=100,
        strength=1.0,
        guidance_scale=7.5,
        qr_weight=500.0,
        perceptual_weight=3.0,
        functional_weight=4.0,
    )
    return tuple(
        replace(
            reference,
            name=f"{profile.name}-scale-{scale:.2f}".replace(".", "p"),
            base_controlnet_scale=scale,
            controlnet_scale=scale,
        )
        for scale in scales
    )


def aggregate_controlnet_benchmark(
    rows: list[dict[str, Any]],
    *,
    expected_contexts: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(str(row["trial"]), []).append(row)
    aggregates = []
    for trial, items in grouped.items():
        complete = len(items) == expected_contexts
        pass_rates = [float(item["pass_rate"]) for item in items]
        worst_decoder_rates = [
            float(item.get("worst_decoder_pass_rate", item["pass_rate"])) for item in items
        ]
        raw_pass_rates = [float(item["raw_pass_rate"]) for item in items]
        first = items[0]
        aggregates.append(
            {
                "trial": trial,
                "model_id": first["controlnet_model_id"],
                "model_subfolder": first["controlnet_model_subfolder"],
                "conditioning_profile": first["controlnet_conditioning_profile"],
                "controlnet_scale": first["parameters"]["controlnet_scale"],
                "contexts": len(items),
                "complete": complete,
                "all_strict": complete and all(bool(item["strict_all"]) for item in items),
                "raw_all_strict": complete
                and all(bool(item["raw_strict_all"]) for item in items),
                "worst_pass_rate": min(pass_rates),
                "mean_pass_rate": sum(pass_rates) / len(items),
                "worst_decoder_pass_rate": min(worst_decoder_rates),
                "mean_worst_decoder_pass_rate": sum(worst_decoder_rates) / len(items),
                "raw_worst_pass_rate": min(raw_pass_rates),
                "raw_mean_pass_rate": sum(raw_pass_rates) / len(items),
                "mean_clip_aesthetic": sum(float(item["clip_aesthetic"]) for item in items)
                / len(items),
                "mean_clip_score": sum(float(item["clip_score"]) for item in items)
                / len(items),
                "raw_mean_clip_aesthetic": sum(
                    float(item["raw_clip_aesthetic"]) for item in items
                )
                / len(items),
                "raw_mean_clip_score": sum(float(item["raw_clip_score"]) for item in items)
                / len(items),
                "raw_mean_module_error_rate": sum(
                    float(item["context_features"]["raw_module_error_rate"])
                    for item in items
                )
                / len(items),
                "mean_module_error_rate": sum(
                    float(item["module_error_rate"]) for item in items
                )
                / len(items),
                "mean_duration_seconds": sum(float(item["duration_seconds"]) for item in items)
                / len(items),
                "mean_peak_gpu_memory_mib": sum(
                    float(item["peak_gpu_memory_mib"]) for item in items
                )
                / len(items),
                "max_peak_gpu_memory_mib": max(
                    float(item["peak_gpu_memory_mib"]) for item in items
                ),
                "failed_contexts": [
                    str(item["context_id"]) for item in items if not bool(item["strict_all"])
                ],
            }
        )
    return sorted(aggregates, key=controlnet_rank_key)


def controlnet_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Rank complete full-pipeline scanning before aesthetics and speed."""
    return (
        -int(bool(row["complete"])),
        -int(bool(row["all_strict"])),
        -float(row.get("worst_decoder_pass_rate", row["worst_pass_rate"])),
        -float(row["worst_pass_rate"]),
        -float(row["mean_pass_rate"]),
        -float(row["mean_clip_aesthetic"]),
        -float(row["mean_clip_score"]),
        float(row["mean_duration_seconds"]),
    )


def select_promotable_controlnet(
    aggregates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [row for row in aggregates if row["complete"] and row["all_strict"]]
    return min(eligible, key=controlnet_rank_key) if eligible else None


def best_trial_per_model(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the best trial for each distinct weight/profile combination."""
    selected = []
    seen: set[tuple[str, str, str]] = set()
    for row in sorted(aggregates, key=controlnet_rank_key):
        identity = (
            str(row["model_id"]),
            str(row["model_subfolder"]),
            str(row["conditioning_profile"]),
        )
        if identity not in seen:
            selected.append(row)
            seen.add(identity)
    return selected
