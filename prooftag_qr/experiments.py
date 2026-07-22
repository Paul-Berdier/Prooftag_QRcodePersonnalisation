from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from PIL import Image

from .qr import QRBlueprint, functional_pattern_mask
from .srpg import SRPGConfig

NEGATIVE_PROMPT_PROFILES = {
    "minimal": "text, watermark, barcode",
    "standard": "easynegative, text, watermark, blurry, plain QR code, barcode",
    "structure_safe": (
        "easynegative, text, watermark, blurry, barcode, checkerboard, regular grid, "
        "repeating squares, tiny high-frequency details"
    ),
}


def image_context_features(image: Image.Image, blueprint: QRBlueprint) -> dict[str, float]:
    """Cheap prompt-result and QR-risk features available before Stage 2."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    rows, cols = blueprint.matrix.shape
    module_means = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        y0 = round(row * gray.shape[0] / rows)
        y1 = max(y0 + 1, round((row + 1) * gray.shape[0] / rows))
        for col in range(cols):
            x0 = round(col * gray.shape[1] / cols)
            x1 = max(x0 + 1, round((col + 1) * gray.shape[1] / cols))
            module_means[row, col] = float(gray[y0:y1, x0:x1].mean())
    target_dark = blueprint.matrix.astype(bool)
    predicted_dark = module_means < 0.5
    errors = predicted_dark != target_dark
    functional = functional_pattern_mask(blueprint)
    signed_margins = np.where(target_dark, 0.5 - module_means, module_means - 0.5)
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    histogram = np.histogram(gray, bins=32, range=(0, 1), density=False)[0].astype(np.float64)
    probabilities = histogram / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum() / 5.0)
    return {
        "qr_version": float(blueprint.version),
        "matrix_modules": float(rows),
        "qr_dark_ratio": float(target_dark.mean()),
        "raw_luminance_mean": float(gray.mean()),
        "raw_luminance_std": float(gray.std()),
        "raw_dark_pixel_ratio": float((gray < 0.25).mean()),
        "raw_light_pixel_ratio": float((gray > 0.75).mean()),
        "raw_edge_density": float((horizontal + vertical) / 2),
        "raw_entropy": entropy,
        "raw_module_error_rate": float(errors.mean()),
        "raw_functional_error_rate": float(errors[functional].mean()),
        "raw_data_error_rate": float(errors[~functional].mean()),
        "raw_module_margin_p10": float(np.quantile(signed_margins, 0.10)),
        "raw_module_margin_median": float(np.median(signed_margins)),
    }


@dataclass(frozen=True, slots=True)
class SRPGTrial:
    name: str
    source: str
    base_steps: int = 12
    base_strength: float = 0.90
    base_guidance_scale: float = 12.0
    base_controlnet_scale: float = 1.50
    steps: int = 100
    strength: float = 1.0
    guidance_scale: float = 12.0
    controlnet_scale: float = 1.35
    qr_weight: float = 500.0
    perceptual_weight: float = 3.0
    functional_weight: float = 4.0
    center_fraction: float = 1 / 3
    dark_threshold: float = 0.5
    light_threshold: float = 0.5
    robust_blur_weight: float = 0.0
    robust_blur_kernel: int = 3
    robust_downscale_weight: float = 0.0
    robust_downscale_factor: float = 0.75
    robust_brightness_weight: float = 0.0
    robust_brightness_low: float = 0.75
    robust_brightness_high: float = 1.25
    robust_contrast_weight: float = 0.0
    robust_contrast_factor: float = 0.70
    target_module_error_rate: float = 0.0
    max_noise_delta_rms: float = 2.0
    eta: float = 0.0
    stage2_seed_index: int = 0
    negative_prompt_profile: str = "standard"

    def to_srpg_config(self) -> SRPGConfig:
        return SRPGConfig(
            steps=self.steps,
            strength=self.strength,
            controlnet_scale=self.controlnet_scale,
            qr_weight=self.qr_weight,
            perceptual_weight=self.perceptual_weight,
            functional_weight=self.functional_weight,
            center_fraction=self.center_fraction,
            dark_threshold=self.dark_threshold,
            light_threshold=self.light_threshold,
            robust_blur_weight=self.robust_blur_weight,
            robust_blur_kernel=self.robust_blur_kernel,
            robust_downscale_weight=self.robust_downscale_weight,
            robust_downscale_factor=self.robust_downscale_factor,
            robust_brightness_weight=self.robust_brightness_weight,
            robust_brightness_low=self.robust_brightness_low,
            robust_brightness_high=self.robust_brightness_high,
            robust_contrast_weight=self.robust_contrast_weight,
            robust_contrast_factor=self.robust_contrast_factor,
            target_module_error_rate=self.target_module_error_rate,
            max_noise_delta_rms=self.max_noise_delta_rms,
            eta=self.eta,
            max_mean_absolute_change=0.40,
            min_relative_module_improvement=0.0,
            save_step_previews=False,
        )

    @property
    def negative_prompt(self) -> str:
        return NEGATIVE_PROMPT_PROFILES[self.negative_prompt_profile]

    def numeric_features(self) -> dict[str, float]:
        values = {
            key: float(value)
            for key, value in asdict(self).items()
            if key not in {"name", "source", "negative_prompt_profile"}
        }
        for profile in NEGATIVE_PROMPT_PROFILES:
            values[f"negative_prompt_{profile}"] = float(profile == self.negative_prompt_profile)
        return values


def sample_e007_trial(sampler: Any, *, name: str) -> SRPGTrial:
    """Sample every generation dimension that can affect quality or scan robustness.

    ``sampler`` follows Optuna's Trial interface but remains untyped so the production package
    does not require Optuna outside the experiment image.
    """
    dark_threshold = sampler.suggest_float("dark_threshold", 0.35, 0.52)
    threshold_margin = sampler.suggest_float("threshold_margin", 0.0, 0.28)
    return SRPGTrial(
        name=name,
        source="e007-tpe",
        base_steps=sampler.suggest_categorical("base_steps", [8, 12, 16, 20, 24]),
        base_strength=sampler.suggest_float("base_strength", 0.65, 1.0),
        base_guidance_scale=sampler.suggest_float("base_guidance_scale", 5.0, 15.0),
        base_controlnet_scale=sampler.suggest_float("base_controlnet_scale", 0.8, 2.0),
        steps=sampler.suggest_categorical("steps", [32, 40, 60, 80, 100, 120]),
        strength=sampler.suggest_float("strength", 0.65, 1.0),
        guidance_scale=sampler.suggest_float("guidance_scale", 5.0, 13.0),
        controlnet_scale=sampler.suggest_float("controlnet_scale", 0.9, 1.9),
        qr_weight=sampler.suggest_float("qr_weight", 300.0, 1800.0, log=True),
        perceptual_weight=sampler.suggest_float("perceptual_weight", 0.0, 8.0),
        functional_weight=sampler.suggest_float("functional_weight", 1.0, 16.0, log=True),
        center_fraction=sampler.suggest_categorical("center_fraction", [0.25, 1 / 3, 0.45, 0.60]),
        dark_threshold=dark_threshold,
        light_threshold=min(0.80, dark_threshold + threshold_margin),
        robust_blur_weight=sampler.suggest_float("robust_blur_weight", 0.0, 2.0),
        robust_blur_kernel=sampler.suggest_categorical("robust_blur_kernel", [3, 5, 7]),
        robust_downscale_weight=sampler.suggest_float("robust_downscale_weight", 0.0, 2.0),
        robust_downscale_factor=sampler.suggest_float("robust_downscale_factor", 0.50, 0.90),
        robust_brightness_weight=sampler.suggest_float("robust_brightness_weight", 0.0, 2.0),
        robust_brightness_low=sampler.suggest_float("robust_brightness_low", 0.60, 0.90),
        robust_brightness_high=sampler.suggest_float("robust_brightness_high", 1.10, 1.40),
        robust_contrast_weight=sampler.suggest_float("robust_contrast_weight", 0.0, 2.0),
        robust_contrast_factor=sampler.suggest_float("robust_contrast_factor", 0.50, 0.90),
        target_module_error_rate=sampler.suggest_float(
            "target_module_error_rate", 0.0, 0.08
        ),
        max_noise_delta_rms=sampler.suggest_float("max_noise_delta_rms", 0.5, 4.0, log=True),
        eta=sampler.suggest_float("eta", 0.0, 1.0),
        stage2_seed_index=sampler.suggest_int("stage2_seed_index", 0, 3),
        negative_prompt_profile=sampler.suggest_categorical(
            "negative_prompt_profile", list(NEGATIVE_PROMPT_PROFILES)
        ),
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
        -float(row.get("worst_decoder_pass_rate", row.get("pass_rate", 0.0))),
        -float(row.get("pass_rate", 0.0)),
        -float(row.get("original_pass_rate", 0.0)),
        -float(row.get("clip_aesthetic", float("-inf"))),
        -float(row.get("clip_score", float("-inf"))),
        float(row.get("module_error_rate", 1.0)),
        float(row.get("mean_absolute_change", 1.0)),
        float(row.get("duration_seconds", float("inf"))),
    )


def aggregate_confirmation(
    rows: list[dict[str, Any]], *, expected_cases: int | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(str(row["trial"]), []).append(row)
    aggregates = []
    for trial, items in grouped.items():
        pass_rates = [float(item["pass_rate"]) for item in items]
        worst_decoder_rates = [
            float(item.get("worst_decoder_pass_rate", item["pass_rate"])) for item in items
        ]
        complete = expected_cases is None or len(items) == expected_cases
        aggregates.append(
            {
                "trial": trial,
                "cases": len(items),
                "complete": complete,
                "all_strict": complete and all(bool(item["strict_all"]) for item in items),
                "worst_pass_rate": min(pass_rates),
                "mean_pass_rate": sum(pass_rates) / len(pass_rates),
                "worst_decoder_pass_rate": min(worst_decoder_rates),
                "mean_worst_decoder_pass_rate": sum(worst_decoder_rates)
                / len(worst_decoder_rates),
                "mean_module_error_rate": sum(float(item["module_error_rate"]) for item in items)
                / len(items),
                "mean_absolute_change": sum(float(item["mean_absolute_change"]) for item in items)
                / len(items),
                "mean_clip_aesthetic": sum(float(item.get("clip_aesthetic", 0.0)) for item in items)
                / len(items),
                "mean_clip_score": sum(float(item.get("clip_score", 0.0)) for item in items)
                / len(items),
                "mean_duration_seconds": sum(float(item["duration_seconds"]) for item in items)
                / len(items),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            -int(row["complete"]),
            -int(row["all_strict"]),
            -row["worst_decoder_pass_rate"],
            -row["worst_pass_rate"],
            -row["mean_pass_rate"],
            -row["mean_clip_aesthetic"],
            -row["mean_clip_score"],
            row["mean_absolute_change"],
            row["mean_duration_seconds"],
        ),
    )


def select_delivery_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the best strictly scannable image; never trade scan success for aesthetics."""
    eligible = [row for row in rows if row.get("status") == "ok" and bool(row.get("strict_all"))]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row.get("clip_aesthetic", float("-inf"))),
            -float(row.get("clip_score", float("-inf"))),
            float(row.get("mean_absolute_change", 1.0)),
            float(row.get("duration_seconds", float("inf"))),
        ),
    )
