"""Grouped feature-block selection using the repository's custom estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .booking_features import GROUPS, build_blocks, raw_groups
from .c45 import C45DecisionTree
from .knn import KNNClassifier
from .logreg import LogisticRegressionClassifier
from .naive_bayes import NaiveBayesClassifier
from .svm import SVM

MODEL_NAMES = ("c45", "knn", "naive_bayes", "logreg", "svm")
_GROUP_ORDER = tuple(GROUPS)
_CANDIDATES = tuple(
    dict.fromkeys(
        (
            (),
            *((name,) for name in _GROUP_ORDER),
            tuple(name for name in _GROUP_ORDER if name not in ("te", "knn")),
            ("calendar", "behavior"),
            ("knn", "calendar"),
            ("knn", "calendar", "behavior"),
            ("te", "knn"),
            _GROUP_ORDER,
        )
    )
)
_CANDIDATES = tuple(
    tuple(name for name in _GROUP_ORDER if name in candidate)
    for candidate in _CANDIDATES
)


def _ordered(groups):
    groups = set(groups)
    unknown = groups - set(_GROUP_ORDER)
    if unknown:
        raise ValueError(f"unknown feature group(s): {', '.join(sorted(unknown))}")
    return tuple(name for name in _GROUP_ORDER if name in groups)


def _labels(y, n):
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("y must be a finite one-dimensional array matching X")
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("booking feature search requires cancellation labels 0 and 1")
    return values


def _group_fold_positions(X, y, n_splits, seed):
    """Stratify folds by label while keeping identical predictor rows together."""
    groups = np.asarray(raw_groups(X))
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("raw_groups(X) must return one group per row")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"at least {n_splits} distinct raw groups are required")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return tuple(valid for _, valid in splitter.split(np.zeros(len(y)), y, groups))


def _folds(X, y, n_splits, seed):
    valid_folds = _group_fold_positions(X, y, n_splits, seed)
    all_positions = np.arange(len(X))
    return tuple(
        (np.setdiff1d(all_positions, valid, assume_unique=True), valid)
        for valid in valid_folds
    )


def feature_partitions(X, y, *, seed):
    """Return positional dev/holdout indices and four grouped inner folds."""
    y = _labels(y, len(X))
    outer = _group_fold_positions(X, y, 5, seed)
    holdout = np.sort(outer[0])
    dev = np.setdiff1d(np.arange(len(X)), holdout, assume_unique=True)
    inner_X, inner_y = X.iloc[dev], y[dev]
    folds = tuple(
        (dev[train], dev[valid])
        for train, valid in _folds(inner_X, inner_y, 4, seed + 1)
    )
    return dev, holdout, folds


def _cap_positions(indices, limit, seed, y):
    if limit is not None and (
        isinstance(limit, (bool, np.bool_))
        or not isinstance(limit, (int, np.integer))
        or limit < 1
    ):
        raise ValueError("train_limit must be a positive integer or None")
    if limit is None or len(indices) <= limit:
        return np.asarray(indices, dtype=int)
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=np.asarray(y)[indices],
    )
    return np.sort(selected)


def _combine(blocks, names, side):
    return pd.concat(
        [
            blocks[name][side].reset_index(drop=True).add_prefix(f"{name}:")
            for name in names
        ],
        axis=1,
    )


def _estimator(name, X, seed):
    if name == "c45":
        return C45DecisionTree()
    if name == "knn":
        return KNNClassifier(n_neighbors=min(15, len(X)), weights="distance")
    if name == "naive_bayes":
        # Gaussian density at a near-zero variance can overwhelm other evidence for a flag.
        types = [
            "categorical" if len(np.unique(column)) <= 2 else "gaussian"
            for column in X.T
        ]
        return NaiveBayesClassifier(alpha=2.0, feature_types=types)
    if name == "logreg":
        return LogisticRegressionClassifier(
            learning_rate=0.05,
            max_iter=1000,
            penalty="l2",
            reg_strength=0.01,
            tol=1e-3,
            random_state=seed,
        )
    variance = float(np.var(X))
    gamma = 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
    # The default capped fits fit in the dense kernel budget; larger fits use the row cache.
    return SVM(
        kernel="rbf",
        C=12,
        gamma=gamma,
        solver="smo",
        random_state=seed,
        max_iter=50000,
        max_passes=10,
        cache_size_mb=512,
    )


def _fit_predict_model(name, train, y, query, seed):
    train_values = train.to_numpy(dtype=float)
    model = _estimator(name, train_values, seed)
    model.fit(train if name == "c45" else train_values, y)
    if name == "c45":
        prediction = model.predict(query)
    elif name == "svm":
        query_values = query.to_numpy(dtype=float)
        prediction = np.concatenate(
            [
                model.predict(query_values[start : start + 256])
                for start in range(0, len(query_values), 256)
            ]
        )
    else:
        prediction = model.predict(query.to_numpy(dtype=float))
    return prediction, getattr(model, "converged_", None), model


def _fit_predict(name, train, y, query, seed):
    prediction, converged, _ = _fit_predict_model(name, train, y, query, seed)
    return prediction, converged


def _macro_f1(actual, predicted, classes=None):
    classes = (
        np.unique(np.concatenate((actual, predicted))) if classes is None else classes
    )
    scores = []
    for label in classes:
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def select_feature_groups(X_dev, y_dev, *, seed, train_limit=4000):
    """Choose one predeclared feature-group tuple per custom estimator."""
    y = _labels(y_dev, len(X_dev))
    folds = _folds(X_dev, y, 4, seed + 1)
    rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        capped = _cap_positions(train_idx, train_limit, seed + fold_id, y)
        train, valid = X_dev.iloc[capped], X_dev.iloc[valid_idx]
        train_y, valid_y = y[capped], y[valid_idx]
        block_map = build_blocks(
            train, train_y, valid, _GROUP_ORDER, seed=seed + fold_id
        )
        for groups in _CANDIDATES:
            names = ("base", *_ordered(groups))
            train_matrix, valid_matrix = (
                _combine(block_map, names, 0),
                _combine(block_map, names, 1),
            )
            for model_name in MODEL_NAMES:
                prediction, converged = _fit_predict(
                    model_name, train_matrix, train_y, valid_matrix, seed + fold_id
                )
                rows.append(
                    {
                        "model": model_name,
                        "candidate": "base" if not groups else "+".join(groups),
                        "groups": groups,
                        "fold": fold_id,
                        "macro_f1": _macro_f1(valid_y, prediction, np.unique(y)),
                        "converged": converged,
                    }
                )
        print(
            f"CV fitur: fold {fold_id + 1}/{len(folds)} selesai ({len(train_y)} baris latih).",
            flush=True,
        )
    scores = pd.DataFrame(rows)
    results = scores.groupby(
        ["model", "candidate", "groups"], sort=False, as_index=False
    ).agg(
        mean_macro_f1=("macro_f1", "mean"),
        fold_scores=("macro_f1", list),
        fold_converged=("converged", list),
    )
    canonical_rank = {name: i for i, name in enumerate(_GROUP_ORDER)}
    winners = {}
    for model_name in MODEL_NAMES:
        choices = results[results.model == model_name].to_dict("records")
        if model_name == "svm":
            choices = [row for row in choices if all(row["fold_converged"])]
            if not choices:
                raise ValueError("No SVM feature candidate converged across all folds")
        best = min(
            choices,
            key=lambda row: (
                -row["mean_macro_f1"],
                len(row["groups"]),
                tuple(canonical_rank[name] for name in row["groups"]),
            ),
        )
        winners[model_name] = _ordered(best["groups"])
    return results, winners


def _validated_winners(winners):
    if set(winners) != set(MODEL_NAMES):
        raise ValueError(f"winners must contain exactly: {', '.join(MODEL_NAMES)}")
    return {name: _ordered(winners[name]) for name in MODEL_NAMES}


def _evaluation_predictions(
    X_train, y_train, X_query, winners, *, seed, train_limit, include_base
):
    y = _labels(y_train, len(X_train))
    if len(X_query) == 0:
        raise ValueError("query data must not be empty")
    indices = _cap_positions(np.arange(len(X_train)), train_limit, seed, y)
    train, train_y = X_train.iloc[indices], y[indices]
    choices = _validated_winners(winners)
    all_groups = set().union(*(set(groups) for groups in choices.values()))
    blocks = build_blocks(train, train_y, X_query, all_groups, seed=seed)
    predictions = {}
    for model_name in MODEL_NAMES:
        feature_choices = (
            (("base", ()), ("selected", choices[model_name]))
            if include_base
            else (("selected", choices[model_name]),)
        )
        for choice, groups in feature_choices:
            names = ("base", *_ordered(groups))
            train_matrix, query_matrix = (
                _combine(blocks, names, 0),
                _combine(blocks, names, 1),
            )
            predictions[f"{model_name}_{choice}"], _ = _fit_predict(
                model_name,
                train_matrix,
                train_y,
                query_matrix,
                seed,
            )
    return predictions


def evaluate_feature_choices(
    X_dev, y_dev, X_holdout, y_holdout, winners, *, seed, train_limit=8000
):
    """Compare base-only and selected blocks on the untouched holdout set."""
    actual = _labels(y_holdout, len(X_holdout))
    predictions = _evaluation_predictions(
        X_dev,
        y_dev,
        X_holdout,
        winners,
        seed=seed,
        train_limit=train_limit,
        include_base=True,
    )
    return pd.DataFrame(
        [
            {
                "model": name,
                "base_holdout_f1": _macro_f1(
                    actual,
                    predictions[f"{name}_base"],
                    np.unique(np.concatenate((actual, np.asarray(y_dev)))),
                ),
                "selected_holdout_f1": _macro_f1(
                    actual,
                    predictions[f"{name}_selected"],
                    np.unique(np.concatenate((actual, np.asarray(y_dev)))),
                ),
            }
            for name in MODEL_NAMES
        ]
    )


def predict_feature_choices(X_train, y, X_test, winners, *, seed, train_limit=8000):
    """Return selected-model predictions without reading test labels."""
    values = _evaluation_predictions(
        X_train,
        y,
        X_test,
        winners,
        seed=seed,
        train_limit=train_limit,
        include_base=False,
    )
    return pd.DataFrame(
        {name: values[f"{name}_selected"] for name in MODEL_NAMES}, index=X_test.index
    )


def fit_feature_choices(X_train, y, X_test, winners, *, seed, train_limit=8000):
    """Fit selected models and return their predictions and preprocessing artifacts."""
    y = _labels(y, len(X_train))
    if len(X_test) == 0:
        raise ValueError("query data must not be empty")
    unknown = set(winners) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"unknown model key(s): {', '.join(sorted(unknown))}")
    choices = {name: _ordered(groups) for name, groups in winners.items()}
    indices = _cap_positions(np.arange(len(X_train)), train_limit, seed, y)
    reference_train = X_train.iloc[indices].copy()
    reference_labels = y[indices].copy()
    all_groups = set().union(*(set(groups) for groups in choices.values()))
    blocks = build_blocks(reference_train, reference_labels, X_test, all_groups, seed=seed)
    predictions, artifacts = {}, {}
    for name, groups in choices.items():
        names = ("base", *groups)
        train_matrix, query_matrix = (
            _combine(blocks, names, 0),
            _combine(blocks, names, 1),
        )
        prediction, _, model = _fit_predict_model(
            name, train_matrix, reference_labels, query_matrix, seed
        )
        predictions[name] = prediction
        artifacts[name] = {
            "model": model,
            "feature_groups": groups,
            "seed": seed,
            "reference_train": reference_train,
            "reference_labels": reference_labels,
            "feature_columns": tuple(train_matrix.columns),
        }
    return pd.DataFrame(predictions, index=X_test.index), artifacts
