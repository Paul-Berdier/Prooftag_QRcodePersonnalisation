from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experiments import NEGATIVE_PROMPT_PROFILES, SRPGTrial


@dataclass(frozen=True, slots=True)
class AdvisorPrediction:
    trial: SRPGTrial
    predicted_pass_rate: float
    pass_rate_uncertainty: float
    predicted_clip_aesthetic: float
    predicted_clip_score: float


def _parameter_features(parameters: dict[str, Any]) -> dict[str, float]:
    ignored = {"name", "source", "negative_prompt_profile"}
    features = {
        key: float(value)
        for key, value in parameters.items()
        if key not in ignored and isinstance(value, int | float | bool)
    }
    selected = str(parameters.get("negative_prompt_profile", "standard"))
    for profile in NEGATIVE_PROMPT_PROFILES:
        features[f"negative_prompt_{profile}"] = float(selected == profile)
    return features


def advisor_rank_key(prediction: AdvisorPrediction) -> tuple[float, ...]:
    """Predicted scanning lower confidence bound first, then aesthetic quality."""
    scan_lower_bound = prediction.predicted_pass_rate - 0.25 * prediction.pass_rate_uncertainty
    return (
        -scan_lower_bound,
        -prediction.predicted_pass_rate,
        -prediction.predicted_clip_aesthetic,
        -prediction.predicted_clip_score,
        prediction.pass_rate_uncertainty,
    )


class ContextualParameterAdvisor:
    """Small nonlinear surrogate that recommends parameters for a prompt and QR context.

    It predicts observed validation success and quality. It never certifies an image: the real
    QR validator remains the only delivery gate.
    """

    def __init__(self, *, random_state: int = 20260721, trees: int = 256) -> None:
        self.random_state = random_state
        self.trees = trees
        self.feature_names: list[str] = []
        self.models: dict[str, Any] = {}
        self.training_report: dict[str, Any] = {}

    @staticmethod
    def _usable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row.get("status") == "ok"
            and isinstance(row.get("context_features"), dict)
            and isinstance(row.get("parameters"), dict)
            and all(key in row for key in ("pass_rate", "clip_aesthetic", "clip_score"))
        ]

    def _row_features(self, row: dict[str, Any]) -> dict[str, float]:
        features = {
            f"context_{key}": float(value)
            for key, value in row["context_features"].items()
            if isinstance(value, int | float | bool)
        }
        features.update(
            {
                f"parameter_{key}": value
                for key, value in _parameter_features(row["parameters"]).items()
            }
        )
        return features

    def _matrix(self, feature_rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[features.get(name, 0.0) for name in self.feature_names] for features in feature_rows],
            dtype=np.float32,
        )

    def fit(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import GroupKFold

        usable = self._usable_rows(rows)
        if len(usable) < 12:
            raise ValueError("at least 12 successful scored trials are required")
        feature_rows = [self._row_features(row) for row in usable]
        self.feature_names = sorted({key for features in feature_rows for key in features})
        matrix = self._matrix(feature_rows)
        targets = {
            "pass_rate": np.asarray([row["pass_rate"] for row in usable], dtype=np.float32),
            "clip_aesthetic": np.asarray(
                [row["clip_aesthetic"] for row in usable], dtype=np.float32
            ),
            "clip_score": np.asarray([row["clip_score"] for row in usable], dtype=np.float32),
        }
        groups = np.asarray(
            [str(row.get("context_id", row.get("case", "unknown"))) for row in usable]
        )
        unique_groups = np.unique(groups)
        cross_validation: dict[str, float | None] = {}
        for name, target in targets.items():
            model = ExtraTreesRegressor(
                n_estimators=self.trees,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=self.random_state,
                n_jobs=-1,
            )
            if unique_groups.size >= 2:
                predictions = np.zeros_like(target)
                splitter = GroupKFold(n_splits=min(5, unique_groups.size))
                for train, test in splitter.split(matrix, target, groups):
                    fold = ExtraTreesRegressor(
                        n_estimators=max(64, self.trees // 2),
                        min_samples_leaf=2,
                        max_features=0.8,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                    fold.fit(matrix[train], target[train])
                    predictions[test] = fold.predict(matrix[test])
                cross_validation[f"{name}_group_mae"] = float(
                    mean_absolute_error(target, predictions)
                )
            else:
                cross_validation[f"{name}_group_mae"] = None
            model.fit(matrix, target)
            self.models[name] = model
        self.training_report = {
            "rows": len(usable),
            "contexts": int(unique_groups.size),
            "features": len(self.feature_names),
            **cross_validation,
        }
        return dict(self.training_report)

    def predict(
        self,
        context_features: dict[str, float],
        candidates: list[SRPGTrial],
    ) -> list[AdvisorPrediction]:
        if not self.models:
            raise RuntimeError("fit the advisor before prediction")
        feature_rows = []
        for candidate in candidates:
            row = {
                "context_features": context_features,
                "parameters": candidate.numeric_features()
                | {"negative_prompt_profile": candidate.negative_prompt_profile},
            }
            feature_rows.append(self._row_features(row))
        matrix = self._matrix(feature_rows)
        means = {name: model.predict(matrix) for name, model in self.models.items()}
        scan_tree_predictions = np.asarray(
            [tree.predict(matrix) for tree in self.models["pass_rate"].estimators_]
        )
        uncertainties = scan_tree_predictions.std(axis=0)
        predictions = [
            AdvisorPrediction(
                trial=candidate,
                predicted_pass_rate=float(np.clip(means["pass_rate"][index], 0.0, 1.0)),
                pass_rate_uncertainty=float(uncertainties[index]),
                predicted_clip_aesthetic=float(means["clip_aesthetic"][index]),
                predicted_clip_score=float(means["clip_score"][index]),
            )
            for index, candidate in enumerate(candidates)
        ]
        return sorted(predictions, key=advisor_rank_key)

    def recommend(
        self,
        context_features: dict[str, float],
        candidates: list[SRPGTrial],
        *,
        limit: int = 6,
    ) -> list[AdvisorPrediction]:
        if limit < 1:
            raise ValueError("limit must be positive")
        return self.predict(context_features, candidates)[:limit]

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
