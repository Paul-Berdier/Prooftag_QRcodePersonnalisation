"""Supervised parameter advisor. All feature fitting/selection stays inside training folds."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import time
from typing import Any
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
import joblib
from .common import read, write, sha, digest, eligibility, normalized_prompt, utc

NUMERIC = ("payload_length", "qr_version", "stage1_steps", "stage1_guidance_scale",
           "stage1_controlnet_scale", "control_guidance_start", "control_guidance_end",
           "stage2_strength_effective", "stage2_steps", "stage2_controlnet_scale",
           "stage2_qr_weight", "stage2_perceptual_weight")
CATEGORICAL = ("error_correction", "qr_mask_pattern", "stage2_initialization")
TARGETS = ("wechat_exact_presets", "clip_score", "clip_aesthetic", "hpsv2_1", "module_error_rate")


def number(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


class Encoder(BaseEstimator, TransformerMixin):
    """No ID, generated-image metric, prompt family label or arbitrary seed as feature."""
    def _tab(self, rows):
        out = []
        for r in rows:
            d = {}
            for k in NUMERIC:
                x = number(r.get(k))
                d[k] = 0.0 if x is None else x
                d[k + "__missing"] = float(x is None)
            for k in CATEGORICAL:
                d[k] = str(r.get(k, "unknown"))
            out.append(d)
        return out

    def fit(self, X, y=None):
        self.tab_ = DictVectorizer(sparse=True).fit(self._tab(X))
        self.text_ = TfidfVectorizer(max_features=1024, ngram_range=(1, 2), min_df=1,
                                   sublinear_tf=True)
        self.text_.fit([str(r.get("prompt") or "empty_prompt") for r in X])
        return self

    def transform(self, X):
        return sparse.hstack([self.tab_.transform(self._tab(X)),
                              self.text_.transform([str(r.get("prompt") or "empty_prompt") for r in X])],
                             format="csr", dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.concatenate([self.tab_.get_feature_names_out(),
                               ["text__" + x for x in self.text_.get_feature_names_out()]])


def prepare_rows(source):
    """Union groups BEFORE raster deduplication: no prompt/payload/raster crosses a split."""
    rows = [dict(x) for x in source if x.get("primary_training_observation") is True
            and x.get("raw") is True and x.get("source_kind") == "parent"
            and x.get("wechat_original_exact") in (True, False)
            and number(x.get("wechat_exact_presets")) is not None]
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(rows):
        for key, val in (("prompt", normalized_prompt(r.get("prompt"))),
                         ("payload", r.get("payload_sha256") or r.get("payload")),
                         ("generation", r.get("generation_group_id")),
                         ("raster", r.get("image_sha256"))):
            if val:
                t = (key, str(val))
                if t in seen:
                    parent[root(i)] = root(seen[t])
                else:
                    seen[t] = i
    raster = {}
    conflicts = set()
    for i, r in enumerate(rows):
        h = r.get("image_sha256") or f"unique-{i}"
        if h in raster and eligibility(rows[raster[h]]) != eligibility(r):
            conflicts.add(h)
        raster.setdefault(h, i)
    kept = [i for h, i in raster.items() if h not in conflicts]
    out = [rows[i] for i in kept]
    groups = [f"group-{root(i)}" for i in kept]
    return out, np.asarray(groups), {"input": len(source), "eligible_rows": len(rows),
           "unique_rows": len(out), "label_conflicting_rasters": len(conflicts),
           "duplicates_removed": len(rows) - len(raster), "groups": len(set(groups))}


def probability(model, rows):
    proba = model.predict_proba(rows)
    classes = list(model.classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(rows))


def metrics(y, p):
    if not len(y):
        return {"count": 0}
    both = len(np.unique(y)) == 2
    return {"count": int(len(y)), "positives": int(np.sum(y)),
            "brier": float(brier_score_loss(y, p)),
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "single_class": not both}


def logits(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p)).reshape(-1, 1)


def recommend(bundle, rows):
    p = probability(bundle["classifier"], rows)
    raw = p.copy()
    if bundle["calibrator"] is not None:
        p = bundle["calibrator"].predict_proba(logits(p))[:, 1]
    encoded = bundle["classifier"][:-1].transform(rows)
    values = {k: m.predict(encoded) for k, m in bundle["regressors"].items()}
    out = []
    for i, r in enumerate(rows):
        entry = {"candidate": r, "p_valid_raw": float(raw[i]), "p_valid": float(p[i])}
        entry.update({"pred_" + k: float(v[i]) for k, v in values.items()})
        out.append(entry)
    # Gate-first, never an arbitrary weighted score compensating unreadability.
    def key(r):
        gate = r["p_valid"] >= .8 and r.get("pred_wechat_exact_presets", 0) >= 34
        return (int(gate), r.get("pred_clip_aesthetic", -1e6) if gate else r["p_valid"],
                r.get("pred_clip_score", -1e6) if gate else r.get("pred_wechat_exact_presets", -1e6),
                r["p_valid"], r.get("pred_hpsv2_1", -1e6))
    return sorted(out, key=key, reverse=True)


def train(work: Path, cfg: dict, deadline: float) -> dict:
    source = read(work / "dataset/observations.json")
    rows, groups, audit = prepare_rows(source)
    out = work / "model"
    out.mkdir(parents=True, exist_ok=True)
    write(out / "dataset-audit.json", audit)
    y = np.array([int(eligibility(r)) for r in rows])
    if (len(rows) < cfg["min_training_rows"] or len(set(groups)) < cfg["min_training_groups"]
        or np.bincount(y, minlength=2).min() < cfg["minimum_class_count"]):
        result = {"status": "BLOCKED_INSUFFICIENT_DATA", "audit": audit,
                  "class_counts": np.bincount(y, minlength=2).tolist(), "trained": False}
        write(out / "training-report.json", result)
        return result
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=73017)
    development, test = next(splitter.split(rows, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=73018)
    a, b = next(inner.split(development, y[development], groups[development]))
    fit, cal = development[a], development[b]
    if len(set(y[fit])) != 2:
        raise ValueError("Partition entraînement à une seule classe : aucune validation inventée")
    # Freeze all partition IDs before fitting or searching hyperparameters.
    write(out / "splits.json", {k: [{"candidate_id": rows[i]["candidate_id"], "group": groups[i]}
                                   for i in ids] for k, ids in (("fit", fit), ("calibration", cal), ("test", test))})
    cv = GroupKFold(n_splits=min(3, len(set(groups[fit]))))
    choices = [(False, 2, 1.0), (True, 2, .8), (True, 4, .8)]
    search = []
    def pipeline(selected, leaf, fraction):
        selector = (SelectFromModel(ExtraTreesClassifier(n_estimators=64, random_state=17,
                      n_jobs=cfg["cpu_per_worker"], class_weight="balanced", min_samples_leaf=2),
                      threshold="median") if selected else "passthrough")
        return Pipeline([("encode", Encoder()), ("select", selector),
                ("model", ExtraTreesClassifier(n_estimators=cfg["trees"], random_state=18,
                        n_jobs=cfg["cpu_per_worker"], min_samples_leaf=leaf,
                        max_features=fraction, class_weight="balanced"))])
    for selected, leaf, fraction in choices:
        scores = []
        for tr, va in cv.split(fit, y[fit], groups[fit]):
            if time.time() > deadline - 150:
                break
            train_ids, val_ids = fit[tr], fit[va]
            m = pipeline(selected, leaf, fraction)
            m.fit([rows[i] for i in train_ids], y[train_ids])
            scores.append(metrics(y[val_ids], probability(m, [rows[i] for i in val_ids])))
        result = {"selection": selected, "min_samples_leaf": leaf, "max_features": fraction,
                  "folds": scores, "complete": len(scores) == cv.n_splits}
        if result["complete"]:
            result["mean_brier"] = float(np.mean([s["brier"] for s in scores]))
        search.append(result)
        write(out / "hyperparameter-search.json", search)
    complete = [s for s in search if s["complete"]]
    if not complete:
        raise TimeoutError("Pas de validation croisée terminée avant la limite d'entraînement")
    best = min(complete, key=lambda x: x["mean_brier"])
    model = pipeline(best["selection"], best["min_samples_leaf"], best["max_features"])
    Xfit = [rows[i] for i in fit]
    model.fit(Xfit, y[fit])
    raw_cal = probability(model, [rows[i] for i in cal])
    calibrator = None
    if np.bincount(y[cal], minlength=2).min() >= 5:
        calibrator = LogisticRegression(C=1, random_state=19).fit(logits(raw_cal), y[cal])
    raw_test = probability(model, [rows[i] for i in test])
    test_p = calibrator.predict_proba(logits(raw_test))[:, 1] if calibrator is not None else raw_test
    bundle = {"classifier": model, "calibrator": calibrator, "regressors": {},
              "schema": "qrnight-advisor-v1", "production_ready": False}
    xfit = model[:-1].transform(Xfit)
    xtest = model[:-1].transform([rows[i] for i in test])
    regressions = {}
    for target in TARGETS:
        labels = np.array([np.nan if number(r.get(target)) is None else number(r.get(target)) for r in rows])
        good = np.isfinite(labels[fit]); good_test = np.isfinite(labels[test])
        if good.sum() < 30 or time.time() > deadline - 30:
            regressions[target] = {"status": "not_trained"}
            continue
        reg = ExtraTreesRegressor(n_estimators=cfg["trees"], min_samples_leaf=best["min_samples_leaf"],
                                  random_state=20, n_jobs=cfg["cpu_per_worker"])
        reg.fit(xfit[good], labels[fit][good])
        bundle["regressors"][target] = reg
        pred = reg.predict(xtest)
        regressions[target] = {"labels_fit": int(good.sum()), "labels_test": int(good_test.sum()),
              "test_mae": float(mean_absolute_error(labels[test][good_test], pred[good_test])) if good_test.any() else None}
    features = model["encode"].get_feature_names_out()
    if best["selection"]:
        features = features[model["select"].get_support()]
    write(out / "features.json", {"numeric": NUMERIC, "categorical": CATEGORICAL,
            "text": "TF-IDF fitted within each training fold (not a semantic embedding model)",
            "selected": list(features), "selection_winner": best["selection"],
            "excluded": ["seed", "prompt_id", "payload_content", "post-generation measurements"]})
    joblib.dump(bundle, out / "advisor.joblib", compress=3)
    restored = joblib.load(out / "advisor.joblib")
    np.testing.assert_allclose(probability(restored["classifier"], [rows[i] for i in test]), raw_test)
    baseline = np.repeat(y[fit].mean(), len(test))
    result = {"status": "TRAINED_RESEARCH_ONLY", "trained": True, "production_ready": False,
              "dataset_sha256": sha(work / "dataset/observations.json"), "audit": audit,
              "selected_configuration": best, "calibrator": "sigmoid_on_separate_groups" if calibrator is not None else "none_insufficient_calibration_labels",
              "test_raw": metrics(y[test], raw_test), "test_calibrated": metrics(y[test], test_p),
              "constant_baseline_test": metrics(y[test], baseline), "regression": regressions,
              "serialization_roundtrip": True, "finished": utc()}
    write(out / "test-predictions.json", [{"candidate_id": rows[i]["candidate_id"], "group": groups[i],
          "observed": int(y[i]), "p_raw": float(p), "p": float(q)} for i, p, q in zip(test, raw_test, test_p)])
    write(out / "training-report.json", result)
    write(out / "model-manifest.json", {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "model-manifest.json"})
    return result
