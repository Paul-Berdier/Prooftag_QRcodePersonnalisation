from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

PromptEmbeddingProvider = Callable[[str], Sequence[float] | np.ndarray]


@dataclass(slots=True)
class AdvisorRecord:
    trial_id: str
    prompt_id: str
    prompt_text: str
    group_id: str
    context_features: dict[str, Any]
    parameters: dict[str, Any]
    targets: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecipeCandidate:
    id: str
    method_id: str
    configuration: dict[str, Any]
    signature: str
    observations: int = 0


@dataclass(frozen=True, slots=True)
class ParameterRecommendation:
    rank: int
    candidate: RecipeCandidate
    scan_safe: bool
    predicted_qr_success: float
    qr_success_uncertainty: float
    qr_success_lower_bound: float
    predicted_qr_tolerance: float | None
    predicted_clip_aesthetic: float | None
    predicted_clip_score: float | None
    predicted_hpsv2_1: float | None
    predicted_human_aesthetic: float | None
    predicted_human_prompt_fidelity: float | None
    predicted_human_qr_discretion: float | None
    predicted_human_overall: float | None
    predicted_duration_ms: float | None
    predicted_saturation_risk: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate"] = asdict(self.candidate)
        return result


@dataclass(slots=True)
class AdvisorDataset:
    records: list[AdvisorRecord]
    candidates: list[RecipeCandidate]
    audit: dict[str, Any]


TARGET_COLUMNS = {
    "qr_success": ("quality_qr_verify_any_exact", "exact_payload_match"),
    "qr_tolerance": ("quality_qr_verify_tolerance_score", "scan_pass_rate"),
    "clip_aesthetic": ("quality_clip_aesthetic",),
    "clip_score": ("quality_clip_score",),
    "hpsv2_1": ("quality_hpsv2_1",),
    "duration_ms": ("total_ms", "generation_ms"),
    "saturation_risk": (
        "quality_high_saturation_pixel_ratio",
        "quality_rgb_clipped_channel_ratio",
    ),
    "human_aesthetic": ("aesthetic_score",),
    "human_prompt_fidelity": ("prompt_fidelity_score",),
    "human_qr_discretion": ("qr_discretion_score",),
    "human_overall": ("overall_score",),
}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "oui"}:
        return 1.0
    if lowered in {"false", "no", "non"}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_float(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        result = _float(row.get(name))
        if result is not None:
            return result
    return None


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or not str(value).strip():
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prompt_features(prompt: str) -> dict[str, float]:
    words = re.findall(r"\w+", prompt, flags=re.UNICODE)
    return {
        "prompt_characters": float(len(prompt)),
        "prompt_words": float(len(words)),
        "prompt_unique_word_ratio": float(
            len({word.lower() for word in words}) / max(1, len(words))
        ),
        "prompt_commas": float(prompt.count(",")),
        "prompt_digits": float(sum(character.isdigit() for character in prompt)),
        "prompt_non_ascii": float(sum(ord(character) > 127 for character in prompt)),
    }


def _method_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _json(row.get("method_configuration_json"))
    if configuration:
        return configuration
    generation = _json(row.get("method_generation_json"))
    model = _json(row.get("method_model_json"))
    tools = _json(row.get("method_tools_json"))
    if generation or model or tools:
        return {
            "id": row.get("method_id", "unknown"),
            "backend": row.get("method_backend", "controlnet"),
            "output_variant": row.get("output_variant_requested")
            or row.get("selected_variant", "raw"),
            "reuse_stage1": row.get("reuse_stage1_requested", True),
            "generation": generation,
            "model": model,
            "tools": tools,
        }

    # Legacy exports did not contain the method JSON. They remain usable when
    # a prompt catalogue is supplied, but only with the parameters that were
    # explicitly echoed into quality diagnostics.
    requested = {
        name.removeprefix("quality_").removesuffix("_requested"): value
        for name, value in row.items()
        if name.startswith("quality_") and name.endswith("_requested") and _float(value) is not None
    }
    return {
        "id": row.get("method_id", "legacy_unknown"),
        "output_variant": row.get("selected_variant", "raw"),
        "legacy_requested_parameters": requested,
    }


def _target_values(row: Mapping[str, Any]) -> dict[str, float]:
    targets = {}
    for target, columns in TARGET_COLUMNS.items():
        value = _first_float(row, columns)
        if value is not None:
            targets[target] = value
    # Saturation is a risk target: use the worst of the available fractions.
    saturation_values = [
        _float(row.get("quality_high_saturation_pixel_ratio")),
        _float(row.get("quality_rgb_clipped_channel_ratio")),
    ]
    saturation_values = [value for value in saturation_values if value is not None]
    if saturation_values:
        targets["saturation_risk"] = max(saturation_values)
    return targets


def load_lab_exports(
    paths: Iterable[str | Path],
    *,
    prompt_catalog: Mapping[str, str] | None = None,
    embedding_provider: PromptEmbeddingProvider | None = None,
) -> AdvisorDataset:
    """Build a leakage-safe E026 dataset from one or more laboratory CSV exports.

    Output metrics are targets only. Model inputs contain the prompt, QR context and
    requested method configuration; no post-generation diagnostic is admitted as a feature.
    """

    prompt_catalog = dict(prompt_catalog or {})
    raw_rows: list[dict[str, Any]] = []
    source_files: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files.append(str(path))
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                row["_source_file"] = str(path)
                raw_rows.append(row)

    deduplicated: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        key = str(row.get("trial_id") or f"legacy-{index}-{row.get('_source_file')}")
        deduplicated[key] = row

    # A power failure or a retried campaign creates new trial IDs for the same
    # prompt/configuration/seed. Keep only the richest generated observation so
    # retries cannot overweight one recipe in the advisor.
    logical_rows: dict[str, tuple[tuple[int, int, int], str, dict[str, Any]]] = {}
    for trial_id, row in deduplicated.items():
        prompt = str(row.get("prompt_text") or prompt_catalog.get(row.get("prompt_id"), "")).strip()
        configuration = _method_configuration(row)
        seed = _float(row.get("seed"))
        if not prompt or not configuration or seed is None:
            logical_key = f"trial:{trial_id}"
        else:
            logical_key = hashlib.sha256(
                _canonical_json(
                    {
                        "payload_hash": row.get("payload_hash"),
                        "payload_length": row.get("payload_length"),
                        "prompt": prompt,
                        "configuration": configuration,
                        "seed": seed,
                        "error_correction": row.get("error_correction"),
                    }
                ).encode("utf-8")
            ).hexdigest()
        targets = _target_values(row)
        rank = (
            int("qr_success" in targets),
            int(bool(str(row.get("generation_run_id") or "").strip())),
            len(targets),
        )
        previous = logical_rows.get(logical_key)
        if previous is None or rank > previous[0]:
            logical_rows[logical_key] = (rank, trial_id, row)

    logical_deduplicated = {
        trial_id: row for _, trial_id, row in logical_rows.values()
    }

    embedding_cache: dict[str, list[float]] = {}
    records: list[AdvisorRecord] = []
    skipped = {"missing_prompt": 0, "missing_qr_target": 0, "non_generated": 0}
    legacy_configurations = 0
    for trial_id, row in logical_deduplicated.items():
        if str(row.get("status", "")).lower() in {"error", "queued", "running"}:
            skipped["non_generated"] += 1
            continue
        prompt_id = str(row.get("prompt_id") or "unknown")
        prompt = str(row.get("prompt_text") or prompt_catalog.get(prompt_id, "")).strip()
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        targets = _target_values(row)
        if "qr_success" not in targets:
            skipped["missing_qr_target"] += 1
            continue
        configuration = _method_configuration(row)
        if "legacy_requested_parameters" in configuration:
            legacy_configurations += 1
        context: dict[str, Any] = {
            **_prompt_features(prompt),
            "error_correction": row.get("error_correction") or "unknown",
        }
        for output_name, input_names in {
            "payload_length": ("payload_length",),
            "qr_version": ("quality_diffqrcoder_qr_version",),
            "qr_mask_pattern": ("quality_diffqrcoder_qr_mask_pattern",),
            "qr_module_size": ("quality_diffqrcoder_qr_module_size",),
            "qr_padding_px": ("quality_diffqrcoder_qr_padding_px",),
        }.items():
            value = _first_float(row, input_names)
            if value is not None:
                context[output_name] = value
        if embedding_provider is not None:
            if prompt not in embedding_cache:
                embedding = np.asarray(embedding_provider(prompt), dtype=np.float32).reshape(-1)
                embedding_cache[prompt] = embedding.astype(float).tolist()
            context.update(
                {
                    f"prompt_embedding_{index:03d}": value
                    for index, value in enumerate(embedding_cache[prompt])
                }
            )
        group_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        records.append(
            AdvisorRecord(
                trial_id=trial_id,
                prompt_id=prompt_id,
                prompt_text=prompt,
                group_id=group_id,
                context_features=context,
                parameters=configuration,
                targets=targets,
                metadata={
                    "campaign_id": row.get("campaign_id"),
                    "method_id": row.get("method_id"),
                    "seed": _float(row.get("seed")),
                    "generation_run_id": row.get("generation_run_id"),
                    "source_file": row.get("_source_file"),
                },
            )
        )

    candidate_counts: dict[str, tuple[dict[str, Any], str, int]] = {}
    for record in records:
        signature = hashlib.sha256(_canonical_json(record.parameters).encode("utf-8")).hexdigest()
        previous = candidate_counts.get(signature)
        candidate_counts[signature] = (
            record.parameters,
            str(record.metadata.get("method_id") or record.parameters.get("id") or "unknown"),
            1 if previous is None else previous[2] + 1,
        )
    candidates = [
        RecipeCandidate(
            id=f"recipe-{signature[:10]}",
            method_id=method_id,
            configuration=configuration,
            signature=signature,
            observations=count,
        )
        for signature, (configuration, method_id, count) in sorted(candidate_counts.items())
    ]
    qr_successes = sum(record.targets["qr_success"] >= 0.5 for record in records)
    audit = {
        "source_files": source_files,
        "raw_rows": len(raw_rows),
        "deduplicated_rows": len(deduplicated),
        "logical_deduplicated_rows": len(logical_deduplicated),
        "logical_duplicates_removed": len(deduplicated) - len(logical_deduplicated),
        "usable_rows": len(records),
        "prompt_groups": len({record.group_id for record in records}),
        "recipes": len(candidates),
        "qr_successes": qr_successes,
        "qr_failures": len(records) - qr_successes,
        "embedding_dimensions": max(
            (
                sum(name.startswith("prompt_embedding_") for name in record.context_features)
                for record in records
            ),
            default=0,
        ),
        "legacy_configurations": legacy_configurations,
        "skipped": skipped,
    }
    return AdvisorDataset(records=records, candidates=candidates, audit=audit)


def _category_token(value: Any) -> str:
    text = str(value)
    readable = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")[:32] or "empty"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def _flatten_features(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if name in {"name", "description", "enabled"}:
                continue
            _flatten_features(f"{prefix}__{name}" if prefix else str(name), nested, output)
    elif isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        output[prefix] = float(value)
    elif isinstance(value, str):
        output[f"{prefix}=={_category_token(value)}"] = 1.0
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output[f"{prefix}__length"] = float(len(value))
        numeric = [_float(item) for item in value]
        numeric = [item for item in numeric if item is not None]
        if numeric:
            output[f"{prefix}__mean"] = float(np.mean(numeric))
            output[f"{prefix}__min"] = float(np.min(numeric))
            output[f"{prefix}__max"] = float(np.max(numeric))
        else:
            output[f"{prefix}=={_category_token(_canonical_json({'items': list(value)}))}"] = 1.0


def _record_features(
    context_features: Mapping[str, Any], parameters: Mapping[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    _flatten_features("context", context_features, result)
    _flatten_features("parameter", parameters, result)
    return result


class E026ParameterAdvisor:
    """Multi-objective surrogate: scan gate first, visual quality second.

    The advisor recommends configurations; it never certifies an output image. The
    generated image must still pass the real qr-verify delivery gate.
    """

    def __init__(
        self,
        *,
        random_state: int = 20260805,
        trees: int = 384,
        uncertainty_penalty: float = 0.75,
    ) -> None:
        self.random_state = random_state
        self.trees = trees
        self.uncertainty_penalty = uncertainty_penalty
        self.feature_names: list[str] = []
        self.classifier: Any | None = None
        self.calibrator: Any | None = None
        self.regressors: dict[str, Any] = {}
        self.training_report: dict[str, Any] = {}
        self.validation_predictions: list[dict[str, Any]] = []
        self.feature_importances: list[tuple[str, float]] = []

    def _matrix(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[row.get(name, 0.0) for name in self.feature_names] for row in rows],
            dtype=np.float32,
        )

    @staticmethod
    def _positive_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
        probabilities = model.predict_proba(matrix)
        classes = list(model.classes_)
        if 1 not in classes:
            return np.zeros(len(matrix), dtype=np.float64)
        return np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)

    def fit(
        self,
        records: Sequence[AdvisorRecord],
        *,
        minimum_rows: int = 100,
        minimum_groups: int = 12,
        minimum_class_count: int = 12,
    ) -> dict[str, Any]:
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            mean_absolute_error,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold

        records = list(records)
        groups = np.asarray([record.group_id for record in records])
        unique_groups = np.unique(groups)
        labels = np.asarray(
            [int(record.targets.get("qr_success", 0.0) >= 0.5) for record in records]
        )
        class_counts = np.bincount(labels, minlength=2)
        problems = []
        if len(records) < minimum_rows:
            problems.append(f"{len(records)} usable rows < {minimum_rows}")
        if len(unique_groups) < minimum_groups:
            problems.append(f"{len(unique_groups)} prompt groups < {minimum_groups}")
        if int(class_counts.min()) < minimum_class_count:
            problems.append(
                f"minority qr-verify class {int(class_counts.min())} < {minimum_class_count}"
            )
        if problems:
            raise ValueError("insufficient E026 dataset: " + "; ".join(problems))

        feature_rows = [
            _record_features(record.context_features, record.parameters) for record in records
        ]
        self.feature_names = sorted({name for row in feature_rows for name in row})
        matrix = self._matrix(feature_rows)
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        oof_probability = np.zeros(len(records), dtype=np.float64)
        for fold_index, (train, test) in enumerate(splitter.split(matrix, labels, groups)):
            classifier = ExtraTreesClassifier(
                n_estimators=max(96, self.trees // 2),
                min_samples_leaf=2,
                max_features=0.8,
                class_weight="balanced",
                random_state=self.random_state + fold_index,
                n_jobs=-1,
            )
            classifier.fit(matrix[train], labels[train])
            oof_probability[test] = self._positive_probability(classifier, matrix[test])
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(oof_probability, labels)
        calibrated_oof = self.calibrator.predict(oof_probability)
        self.classifier = ExtraTreesClassifier(
            n_estimators=self.trees,
            min_samples_leaf=2,
            max_features=0.8,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.classifier.fit(matrix, labels)
        self.validation_predictions = [
            {
                "group_id": str(groups[index]),
                "observed": int(labels[index]),
                "raw_probability": float(oof_probability[index]),
                "calibrated_probability": float(calibrated_oof[index]),
            }
            for index in range(len(records))
        ]
        self.feature_importances = sorted(
            zip(self.feature_names, self.classifier.feature_importances_, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        report: dict[str, Any] = {
            "rows": len(records),
            "prompt_groups": len(unique_groups),
            "features": len(self.feature_names),
            "qr_successes": int(labels.sum()),
            "qr_failures": int(len(labels) - labels.sum()),
            "validation": "GroupKFold by SHA-256(prompt text); no prompt leakage",
            "qr_average_precision": float(average_precision_score(labels, calibrated_oof)),
            "qr_brier_raw": float(brier_score_loss(labels, oof_probability)),
            "qr_brier_calibrated": float(brier_score_loss(labels, calibrated_oof)),
            "qr_roc_auc": float(roc_auc_score(labels, calibrated_oof)),
        }

        regression_targets = [
            "qr_tolerance",
            "clip_aesthetic",
            "clip_score",
            "hpsv2_1",
            "duration_ms",
            "saturation_risk",
            "human_aesthetic",
            "human_prompt_fidelity",
            "human_qr_discretion",
            "human_overall",
        ]
        for target_name in regression_targets:
            selected = np.asarray([target_name in record.targets for record in records], dtype=bool)
            target = np.asarray(
                [record.targets.get(target_name, np.nan) for record in records],
                dtype=np.float64,
            )
            target_groups = np.unique(groups[selected])
            if int(selected.sum()) < max(20, minimum_class_count) or len(target_groups) < 3:
                report[f"{target_name}_status"] = "not_trained_insufficient_labels"
                continue
            predictions = np.zeros(int(selected.sum()), dtype=np.float64)
            selected_matrix = matrix[selected]
            selected_target = target[selected]
            selected_groups = groups[selected]
            target_splitter = GroupKFold(n_splits=min(5, len(target_groups)))
            for fold_index, (train, test) in enumerate(
                target_splitter.split(selected_matrix, selected_target, selected_groups)
            ):
                fold = ExtraTreesRegressor(
                    n_estimators=max(96, self.trees // 2),
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=self.random_state + 100 + fold_index,
                    n_jobs=-1,
                )
                fold.fit(selected_matrix[train], selected_target[train])
                predictions[test] = fold.predict(selected_matrix[test])
            model = ExtraTreesRegressor(
                n_estimators=self.trees,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=self.random_state + 100,
                n_jobs=-1,
            )
            model.fit(selected_matrix, selected_target)
            self.regressors[target_name] = model
            report[f"{target_name}_labels"] = int(selected.sum())
            report[f"{target_name}_group_mae"] = float(
                mean_absolute_error(selected_target, predictions)
            )
        self.training_report = report
        return dict(report)

    def recommend(
        self,
        *,
        prompt: str,
        candidates: Sequence[RecipeCandidate],
        prompt_embedding: Sequence[float] | np.ndarray | None = None,
        payload_length: int | None = None,
        error_correction: str = "M",
        qr_context: Mapping[str, Any] | None = None,
        scan_probability_threshold: float = 0.80,
        limit: int = 6,
    ) -> list[ParameterRecommendation]:
        if self.classifier is None:
            raise RuntimeError("fit the E026 advisor before requesting recommendations")
        if not 0.0 <= scan_probability_threshold <= 1.0:
            raise ValueError("scan_probability_threshold must be between zero and one")
        context: dict[str, Any] = {
            **_prompt_features(prompt),
            "error_correction": error_correction,
            **dict(qr_context or {}),
        }
        if payload_length is not None:
            context["payload_length"] = float(payload_length)
        if prompt_embedding is not None:
            embedding = np.asarray(prompt_embedding, dtype=np.float32).reshape(-1)
            context.update(
                {
                    f"prompt_embedding_{index:03d}": float(value)
                    for index, value in enumerate(embedding)
                }
            )
        feature_rows = [
            _record_features(context, candidate.configuration) for candidate in candidates
        ]
        matrix = self._matrix(feature_rows)
        raw_probability = self._positive_probability(self.classifier, matrix)
        probability = (
            np.asarray(self.calibrator.predict(raw_probability))
            if self.calibrator is not None
            else raw_probability
        )
        tree_probabilities = np.asarray(
            [self._positive_probability(tree, matrix) for tree in self.classifier.estimators_]
        )
        uncertainty = tree_probabilities.std(axis=0)
        lower_bound = np.clip(probability - self.uncertainty_penalty * uncertainty, 0.0, 1.0)
        regression_predictions = {
            name: model.predict(matrix) for name, model in self.regressors.items()
        }

        def predicted(name: str, index: int) -> float | None:
            values = regression_predictions.get(name)
            return float(values[index]) if values is not None else None

        unranked = [
            ParameterRecommendation(
                rank=0,
                candidate=candidate,
                scan_safe=bool(lower_bound[index] >= scan_probability_threshold),
                predicted_qr_success=float(probability[index]),
                qr_success_uncertainty=float(uncertainty[index]),
                qr_success_lower_bound=float(lower_bound[index]),
                predicted_qr_tolerance=predicted("qr_tolerance", index),
                predicted_clip_aesthetic=predicted("clip_aesthetic", index),
                predicted_clip_score=predicted("clip_score", index),
                predicted_hpsv2_1=predicted("hpsv2_1", index),
                predicted_human_aesthetic=predicted("human_aesthetic", index),
                predicted_human_prompt_fidelity=predicted("human_prompt_fidelity", index),
                predicted_human_qr_discretion=predicted("human_qr_discretion", index),
                predicted_human_overall=predicted("human_overall", index),
                predicted_duration_ms=predicted("duration_ms", index),
                predicted_saturation_risk=predicted("saturation_risk", index),
            )
            for index, candidate in enumerate(candidates)
        ]

        def optional(value: float | None, *, reverse: bool = False) -> float:
            if value is None:
                return float("-inf")
            return -value if reverse else value

        ranked = sorted(
            unranked,
            key=lambda item: (
                item.scan_safe,
                item.qr_success_lower_bound,
                item.predicted_qr_success,
                optional(item.predicted_qr_tolerance),
                optional(item.predicted_human_overall),
                optional(item.predicted_human_aesthetic),
                optional(item.predicted_human_prompt_fidelity),
                optional(item.predicted_human_qr_discretion),
                optional(item.predicted_hpsv2_1),
                optional(item.predicted_clip_aesthetic),
                optional(item.predicted_clip_score),
                optional(item.predicted_saturation_risk, reverse=True),
                optional(item.predicted_duration_ms, reverse=True),
            ),
            reverse=True,
        )
        return [replace(item, rank=index) for index, item in enumerate(ranked[:limit], start=1)]

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> E026ParameterAdvisor:
        import joblib

        advisor = joblib.load(Path(path))
        if not isinstance(advisor, cls):
            raise TypeError(f"unexpected advisor type: {type(advisor)!r}")
        return advisor
