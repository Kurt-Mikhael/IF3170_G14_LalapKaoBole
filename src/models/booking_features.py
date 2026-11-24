from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

GROUPS = ("calendar", "party", "price", "history", "behavior", "missing", "te", "knn")

_NUMERIC = (
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
)
_CATEGORICAL = (
    "needs_parking",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "meal_package",
    "room_category",
    "booking_channel",
)
_MISSING = (
    "special_requests",
    "nightly_rate_idr",
    "days_before_arrival",
    "num_children",
    "num_adults",
    "meal_package",
    "needs_parking",
    "room_category",
)
_ENCODINGS = (
    ("booking_channel",),
    ("room_category",),
    ("meal_package",),
    ("booking_channel", "room_category"),
    ("arrival_month", "booking_channel"),
    ("room_category", "meal_package"),
    ("arrival_year", "arrival_month"),
)


def raw_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return deterministic uint64 group IDs for identical predictor rows."""
    _validate_columns(frame.columns)
    return pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype=np.uint64
    )


def _is_id_or_target(column) -> bool:
    name = str(column).strip().lower()
    tokens = set(re.split(r"[^a-z0-9]+", name))
    return bool(
        tokens & {"id", "uuid", "guid", "identifier", "target", "label", "outcome"}
    ) or name in {
        "target",
        "label",
        "outcome",
        "is_canceled",
        "is_cancelled",
        "canceled",
        "cancelled",
        "booking_status",
        "cancellation",
        "cancellation_status",
        "is_cancellation",
    }


def _validate_columns(columns) -> None:
    if not columns.is_unique:
        raise ValueError("predictor columns must be unique")
    invalid = [col for col in columns if _is_id_or_target(col)]
    if invalid:
        raise ValueError(f"ID/target columns are not predictors: {invalid}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        values = (
            pd.to_numeric(frame[col], errors="coerce")
            if col in frame
            else pd.Series(np.nan, index=frame.index)
        )
        out[col] = values.replace([np.inf, -np.inf], np.nan)
    return out


def _raw(
    train: pd.DataFrame, query: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = list(
        dict.fromkeys([*_NUMERIC, *_CATEGORICAL, *train.columns, *query.columns])
    )
    num_cols = [
        c
        for c in columns
        if c in _NUMERIC
        or (c in train and pd.api.types.is_numeric_dtype(train[c].dtype))
    ]
    cat_cols = [c for c in columns if c not in num_cols]
    trn = _numeric(train, tuple(num_cols))
    qun = _numeric(query, tuple(num_cols))
    for col in num_cols:
        fill = trn[col].median()
        fill = float(fill) if pd.notna(fill) and np.isfinite(fill) else 0.0
        trn[col] = trn[col].fillna(fill)
        qun[col] = qun[col].fillna(fill)
    trc, quc = pd.DataFrame(index=train.index), pd.DataFrame(index=query.index)
    for col in cat_cols:
        a = train[col] if col in train else pd.Series(np.nan, index=train.index)
        b = query[col] if col in query else pd.Series(np.nan, index=query.index)
        mode = a.dropna().astype(str).mode()
        fill = mode.iloc[0] if len(mode) else "__missing__"
        trc[col] = a.astype("string").fillna(fill)
        quc[col] = b.astype("string").fillna(fill)
    return trn, qun, trc, quc


def _base(trn: pd.DataFrame, qun: pd.DataFrame, trc: pd.DataFrame, quc: pd.DataFrame):
    if len(trn.columns):
        mean = trn.to_numpy(float).mean(axis=0)
        scale = trn.to_numpy(float).std(axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        a = (trn.to_numpy(float) - mean) / scale
        b = (qun.to_numpy(float) - mean) / scale
    else:
        a, b = np.empty((len(trc), 0)), np.empty((len(quc), 0))
    names = [f"base__{c}" for c in trn.columns]
    for col in trc.columns:
        categories = sorted(trc[col].unique())
        for category in categories:
            names.append(f"base__{col}={category}")
            a = np.column_stack((a, (trc[col].to_numpy() == category).astype(float)))
            b = np.column_stack((b, (quc[col].to_numpy() == category).astype(float)))
    return pd.DataFrame(a, columns=names), pd.DataFrame(b, columns=names)


def _scale(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    if not a.shape[1]:
        return a, b
    med = np.array(
        [
            np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
            for col in a.T
        ]
    )
    filled = np.where(np.isfinite(a), a, med)
    q1, q3 = np.percentile(filled, [25, 75], axis=0)
    width = q3 - q1
    width = np.where(np.isfinite(width) & (width > 1e-12), width, 1.0)
    query_values = np.nan_to_num(b, nan=med, posinf=med, neginf=med)
    return (
        np.clip((filled - med) / width, -8, 8),
        np.clip((query_values - med) / width, -8, 8),
    )


def _engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = _numeric(frame, _NUMERIC)
    for col in _NUMERIC:
        n[col] = n[col].fillna(0.0)
    dates = pd.to_datetime(
        pd.DataFrame(
            {
                "year": frame.get("arrival_year", 2000),
                "month": frame.get("arrival_month", 1),
                "day": frame.get("arrival_day", 1),
            },
            index=frame.index,
        ),
        errors="coerce",
    )
    dow = dates.dt.day_of_week.fillna(0).to_numpy(float)
    doy = dates.dt.day_of_year.fillna(1).to_numpy(float)
    week = dates.dt.isocalendar().week.fillna(1).to_numpy(float)
    month = (
        pd.to_numeric(
            frame.get("arrival_month", pd.Series(1, index=frame.index)), errors="coerce"
        )
        .fillna(1)
        .to_numpy(float)
    )
    nights = n.weekend_nights + n.weekday_nights
    guests = n.num_adults + n.num_children
    out = pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365),
            "doy_cos": np.cos(2 * np.pi * doy / 365),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "week_sin": np.sin(2 * np.pi * week / 52),
            "week_cos": np.cos(2 * np.pi * week / 52),
            "week_of_year": week,
            "quarter": dates.dt.quarter.fillna(1).to_numpy(float),
            "is_weekend": (dow >= 5).astype(float),
            "is_high_month": np.isin(month, [6, 7, 8]).astype(float),
            "is_low_month": np.isin(month, [1, 12]).astype(float),
            "total_guests": guests,
            "total_nights": nights,
            "adults_children_ratio": n.num_adults / (n.num_children + 1),
            "is_family": (n.num_children > 0).astype(float),
            "log_rate": np.log1p(n.nightly_rate_idr.clip(lower=0)),
            "rate_per_stay_night": n.nightly_rate_idr / nights.replace(0, np.nan),
            "rate_per_guest": n.nightly_rate_idr / guests.replace(0, np.nan),
            "rate_per_stay_night_guest": n.nightly_rate_idr
            / (nights * guests).replace(0, np.nan),
            "prior_total": n.prior_cancellations + n.prior_completed_bookings,
            "prior_cancel_ratio": n.prior_cancellations
            / (n.prior_cancellations + n.prior_completed_bookings + 1),
            "any_prior_cancel": (n.prior_cancellations > 0).astype(float),
            "any_prior_completed": (n.prior_completed_bookings > 0).astype(float),
            "log_days": np.log1p(n.days_before_arrival.clip(lower=0)),
            "is_last_minute": (n.days_before_arrival <= 7).astype(float),
            "is_long_lead": (n.days_before_arrival > 180).astype(float),
            "has_special_requests": (n.special_requests > 0).astype(float),
            "special_requests_log": np.log1p(n.special_requests.clip(lower=0)),
        },
        index=frame.index,
    )
    cats = pd.DataFrame(index=frame.index)
    for col in ("booking_channel", "room_category", "meal_package"):
        cats[col] = (
            frame[col].map(lambda x: str(x) if pd.notna(x) else "__missing__")
            if col in frame
            else "__missing__"
        )
    cats["arrival_month"] = month.astype(int).astype(str)
    cats["arrival_year"] = (
        pd.to_numeric(
            frame.get("arrival_year", pd.Series(2000, index=frame.index)),
            errors="coerce",
        )
        .fillna(2000)
        .astype(int)
        .astype(str)
    )
    cats["dow"] = dow.astype(int).astype(str)
    return out.replace([np.inf, -np.inf], np.nan), cats


def _folds(groups: np.ndarray, seed: int):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("supervised features require at least two distinct raw groups")
    # urutan digest ber-seed membuat pembagian bisa diulang tanpa memakai label
    ordered = sorted(
        unique.tolist(),
        key=lambda value: hashlib.sha256(f"{seed}:{int(value)}".encode()).digest(),
    )
    rank = {group: i for i, group in enumerate(ordered)}
    labels = np.array([rank[g] for g in groups])
    return list(
        GroupKFold(n_splits=min(5, len(unique))).split(
            np.zeros(len(groups)), groups=labels
        )
    )


def _te(train_cat, query_cat, y, groups, seed):
    a = np.empty((len(y), len(_ENCODINGS)))
    b = np.empty((len(query_cat), len(_ENCODINGS)))
    for j, cols in enumerate(_ENCODINGS):
        tk = train_cat[list(cols)].astype(str).agg("|".join, axis=1)
        qk = query_cat[list(cols)].astype(str).agg("|".join, axis=1)
        for fit, valid in _folds(groups, seed):
            prior = float(y[fit].mean())
            stats = (
                pd.DataFrame({"key": tk.iloc[fit].to_numpy(), "y": y[fit]})
                .groupby("key")
                .y.agg(["sum", "count"])
            )
            means = (stats["sum"] + 20 * prior) / (stats["count"] + 20)
            a[valid, j] = tk.iloc[valid].map(means).fillna(prior).to_numpy()
        prior = float(y.mean())
        stats = (
            pd.DataFrame({"key": tk.to_numpy(), "y": y})
            .groupby("key")
            .y.agg(["sum", "count"])
        )
        b[:, j] = (
            qk.map((stats["sum"] + 20 * prior) / (stats["count"] + 20))
            .fillna(prior)
            .to_numpy()
        )
    # rata-rata sudah terbatas, jadi tidak diskalakan antar-baris OOF agar label tidak bocor lewat fold lain
    return a, b


def _knn(base_train, base_query, y, groups, seed):
    folds = _folds(groups, seed)
    output = np.empty((len(y), 8))

    def features(q, ref, labels):
        if len(ref) == 0:
            return np.zeros((len(q), 8))
        # dimensi dipilih hanya dari label di fold referensi ini
        centered = ref - ref.mean(axis=0)
        target = labels - labels.mean()
        denom = np.linalg.norm(centered, axis=0) * np.linalg.norm(target)
        score = np.divide(
            np.abs(centered.T @ target),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        dims = np.argsort(-score, kind="stable")[: min(25, ref.shape[1])]
        r, v = ref[:, dims], q[:, dims]
        center, scale = r.mean(axis=0), r.std(axis=0)
        scale[scale == 0] = 1
        r, v = (
            np.clip((r - center) / scale, -1e6, 1e6),
            np.clip((v - center) / scale, -1e6, 1e6),
        )
        r_norm = np.einsum("ij,ij->i", r, r)
        result = np.empty((len(v), 8))
        for start in range(0, len(v), 256):
            chunk = v[start : start + 256]
            # identitas dot product membatasi memori ke satu matriks chunk kali referensi
            d2 = np.maximum(
                np.einsum("ij,ij->i", chunk, chunk)[:, None]
                + r_norm[None, :]
                - 2 * chunk @ r.T,
                0.0,
            )
            order = np.argsort(d2, axis=1, kind="stable")
            for row, ids in enumerate(order):
                distances = d2[row]
                sorted_ids = ids
                sorted_d2 = distances[sorted_ids]
                for col, k in enumerate((5, 15, 50)):
                    count = min(k, len(sorted_ids))
                    result[start + row, col] = labels[sorted_ids[:count]].mean()
                nearest50 = sorted_ids[: min(50, len(sorted_ids))]
                weights = 1.0 / np.maximum(distances[nearest50], 1e-12)
                result[start + row, 3] = np.average(labels[nearest50], weights=weights)
                class0, class1 = labels == 0, labels == 1
                fallback = distances.max() + 1.0
                d0 = distances[class0].min() if class0.any() else fallback
                d1 = distances[class1].min() if class1.any() else fallback
                result[start + row, 4:7] = (d0, d1, d0 - d1)
                result[start + row, 7] = sorted_d2[: min(15, len(sorted_ids))].mean()
        return result

    for fit, valid in folds:
        output[valid] = features(base_train[valid], base_train[fit], y[fit])
    external = features(base_query, base_train, y)
    # rata-rata target sudah terbatas dan jarak memakai fitur dasar yang distandardisasi
    # penskala global yang memakai label dihindari karena bocor secara tidak langsung
    return output, external


def build_blocks(
    train: pd.DataFrame,
    y: np.ndarray,
    query: pd.DataFrame,
    needed: set[str],
    *,
    seed: int = 824,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build train/query blocks; ``base`` is always present and unweighted.

    Every block has aligned descriptive columns. Robust scaling for engineered
    blocks and standardization for base numeric columns are fit on train only.
    """
    y = np.asarray(y).ravel()
    needed = set(needed)
    if len(train) != len(y):
        raise ValueError("train and y must have equal lengths")
    _validate_columns(train.columns)
    _validate_columns(query.columns)
    if set(train.columns) != set(query.columns):
        raise ValueError("train and query must have the same predictor columns")
    query = query.loc[:, train.columns]
    unknown = set(needed) - set(GROUPS) - {"base"}
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    groups = raw_groups(train)
    trn, qun, trc, quc = _raw(train, query)
    base_tr, base_qu = _base(trn, qun, trc, quc)
    result = {"base": (base_tr, base_qu)}
    if not needed - {"base"}:
        return result
    eng_tr, cat_tr = _engineer(train)
    eng_qu, cat_qu = _engineer(query)
    fields = {
        "calendar": [
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "month_sin",
            "month_cos",
            "week_sin",
            "week_cos",
            "week_of_year",
            "quarter",
            "is_weekend",
            "is_high_month",
            "is_low_month",
        ],
        "party": ["total_guests", "total_nights", "adults_children_ratio", "is_family"],
        "price": [
            "log_rate",
            "rate_per_stay_night",
            "rate_per_guest",
            "rate_per_stay_night_guest",
        ],
        "history": [
            "prior_total",
            "prior_cancel_ratio",
            "any_prior_cancel",
            "any_prior_completed",
        ],
        "behavior": [
            "log_days",
            "is_last_minute",
            "is_long_lead",
            "has_special_requests",
            "special_requests_log",
        ],
    }
    for name, cols in fields.items():
        if name in needed:
            a, b = _scale(eng_tr[cols].to_numpy(), eng_qu[cols].to_numpy())
            result[name] = (
                pd.DataFrame(a, columns=cols),
                pd.DataFrame(b, columns=cols),
            )
    if "missing" in needed:
        cols = [f"missing__{c}" for c in _MISSING]
        result["missing"] = (
            pd.DataFrame(
                {
                    c: train[src].isna().to_numpy(float)
                    if src in train
                    else np.ones(len(train))
                    for c, src in zip(cols, _MISSING)
                }
            ),
            pd.DataFrame(
                {
                    c: query[src].isna().to_numpy(float)
                    if src in query
                    else np.ones(len(query))
                    for c, src in zip(cols, _MISSING)
                }
            ),
        )
    if "te" in needed:
        a, b = _te(cat_tr, cat_qu, y, groups, seed)
        cols = ["te__" + "_".join(keys) for keys in _ENCODINGS]
        result["te"] = (pd.DataFrame(a, columns=cols), pd.DataFrame(b, columns=cols))
    if "knn" in needed:
        a, b = _knn(
            base_tr.to_numpy(), base_qu.to_numpy(), y.astype(float), groups, seed
        )
        cols = [
            "knn__cancel_mean_k5",
            "knn__cancel_mean_k15",
            "knn__cancel_mean_k50",
            "knn__inverse_d2_weighted_cancel_mean_k50",
            "knn__nearest_class0_d2",
            "knn__nearest_class1_d2",
            "knn__class0_minus_class1_nearest_d2",
            "knn__mean_d2_k15",
        ]
        result["knn"] = (pd.DataFrame(a, columns=cols), pd.DataFrame(b, columns=cols))
    return result
