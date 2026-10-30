"""The single preprocessing transform used for training, evaluation and inference.

Pipeline version 2 (fixes a leakage issue in version 1, which fitted the transform on
the server-held validation split):

* Numeric/binary features: mean imputation -> standardisation.
* Categorical features: most-frequent imputation -> one-hot encoding with the category
  vocabulary fixed by the dataset schema.

The learned parameters (means, standard deviations, most-frequent categories) are
estimated from **training rows only**. In federated runs every hospital computes
*aggregate* statistics of its own local training rows (counts, means, sums of squared
deviations, category counts); the server combines them exactly (Chan et al. parallel
variance) and sends the fitted parameters back. The result is numerically identical to
fitting scikit-learn's ``SimpleImputer(strategy="mean") + StandardScaler`` and
``SimpleImputer(strategy="most_frequent") + OneHotEncoder`` on the pooled training rows,
which ``tests/test_leakage.py`` verifies. Mean (not median) imputation is used because
the median cannot be combined exactly from per-hospital aggregates.

Validation and test rows are only ever *transformed*, never used for fitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml.datasets.registry import DatasetRegistry
from ml.datasets.schema import DatasetSpec

PIPELINE_VERSION = 2
MISSING = "__missing__"


@dataclass(frozen=True)
class TransformedColumn:
    name: str  # e.g. "ChestPainType=ASY"
    feature: str  # original feature name, e.g. "ChestPainType"
    category: str | None  # category for one-hot columns


def transformed_columns(spec: DatasetSpec) -> list[TransformedColumn]:
    """Ordered description of the columns produced by the preprocessor."""
    cols = [TransformedColumn(n, n, None) for n in spec.numeric_features]
    for c in spec.categorical_features:
        for cat in spec.feature(c).categories:
            cols.append(TransformedColumn(f"{c}={cat}", c, cat))
    return cols


# ----------------------------------------------------------------- statistics
def local_statistics(spec: DatasetSpec, X: pd.DataFrame) -> dict[str, Any]:
    """Aggregate, non-row-level statistics of one data holder's rows."""
    X = X[spec.feature_names]
    stats: dict[str, Any] = {"n_rows": int(len(X)), "numeric": {}, "categorical": {}}
    for name in spec.numeric_features:
        col = pd.to_numeric(X[name], errors="coerce").to_numpy(dtype=float)
        obs = col[np.isfinite(col)]
        n = int(len(obs))
        mean = float(obs.mean()) if n else 0.0
        m2 = float(((obs - mean) ** 2).sum()) if n else 0.0
        stats["numeric"][name] = {"n_obs": n, "mean": mean, "m2": m2}
    for name in spec.categorical_features:
        col = X[name]
        counts = {cat: int((col == cat).sum()) for cat in spec.feature(name).categories}
        counts[MISSING] = int(col.isna().sum())
        stats["categorical"][name] = counts
    return stats


def combine_statistics(spec: DatasetSpec, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Exactly combine per-holder statistics (parallel mean/variance algorithm)."""
    if not parts:
        raise ValueError("No statistics to combine")
    total = {"n_rows": sum(int(p["n_rows"]) for p in parts), "numeric": {}, "categorical": {}}
    for name in spec.numeric_features:
        n, mean, m2 = 0, 0.0, 0.0
        for p in parts:
            s = p["numeric"][name]
            nb, mb, m2b = int(s["n_obs"]), float(s["mean"]), float(s["m2"])
            if nb == 0:
                continue
            delta = mb - mean
            nt = n + nb
            mean = mean + delta * nb / nt
            m2 = m2 + m2b + delta * delta * n * nb / nt
            n = nt
        total["numeric"][name] = {"n_obs": n, "mean": mean, "m2": m2}
    for name in spec.categorical_features:
        keys = list(spec.feature(name).categories) + [MISSING]
        total["categorical"][name] = {k: sum(int(p["categorical"][name].get(k, 0)) for p in parts)
                                      for k in keys}
    return total


# ----------------------------------------------------------------- transformer
class SchemaPreprocessor(TransformerMixin, BaseEstimator):
    """Schema-driven transform with parameters learned from training rows only."""

    def __init__(self, dataset_key: str) -> None:
        self.dataset_key = dataset_key

    @property
    def spec(self) -> DatasetSpec:
        return DatasetRegistry.get(self.dataset_key)

    # ------------------------------------------------------------- fitting
    def fit(self, X: pd.DataFrame, y: Any = None) -> "SchemaPreprocessor":
        """Central fit on training rows (used for pooled baselines and tests)."""
        return self.fit_statistics(local_statistics(self.spec, X), fitted_on="pooled training rows")

    def fit_statistics(self, stats: dict[str, Any], fitted_on: str) -> "SchemaPreprocessor":
        spec = self.spec
        n_rows = int(stats["n_rows"])
        if n_rows <= 0:
            raise ValueError("Cannot fit preprocessing on zero training rows")
        means, scales = [], []
        for name in spec.numeric_features:
            s = stats["numeric"][name]
            n_obs, mean, m2 = int(s["n_obs"]), float(s["mean"]), float(s["m2"])
            if n_obs == 0:  # column entirely missing in training data: impute 0, unit scale
                mean, var = 0.0, 0.0
            else:
                # variance of the mean-imputed column (imputed entries add zero deviation)
                var = m2 / n_rows
            means.append(mean)
            scales.append(float(np.sqrt(var)) if var > 1e-24 else 1.0)
        modes = []
        for name in spec.categorical_features:
            counts = stats["categorical"][name]
            cats = sorted(spec.feature(name).categories)  # sklearn tie-break: smallest value
            best = max(cats, key=lambda c: (counts.get(c, 0), -cats.index(c)))
            if counts.get(best, 0) == 0:
                best = spec.feature(name).categories[0]
            modes.append(best)
        self.means_ = np.asarray(means, dtype=float)
        self.scales_ = np.asarray(scales, dtype=float)
        self.modes_ = modes
        self.n_fit_rows_ = n_rows
        self.fitted_on_ = fitted_on
        self.pipeline_version_ = PIPELINE_VERSION
        self.n_features_out_ = len(transformed_columns(spec))
        return self

    # ------------------------------------------------------------- params I/O
    def to_params(self) -> dict[str, Any]:
        self._check()
        return {"dataset_key": self.dataset_key, "version": PIPELINE_VERSION,
                "means": self.means_.tolist(), "scales": self.scales_.tolist(),
                "modes": list(self.modes_), "n_fit_rows": int(self.n_fit_rows_),
                "fitted_on": self.fitted_on_}

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_params(), sort_keys=True).encode("utf-8")

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "SchemaPreprocessor":
        if int(params.get("version", 0)) != PIPELINE_VERSION:
            raise ValueError("Preprocessing parameters come from an incompatible pipeline version")
        obj = cls(params["dataset_key"])
        spec = obj.spec
        if (len(params["means"]) != len(spec.numeric_features)
                or len(params["modes"]) != len(spec.categorical_features)):
            raise ValueError("Preprocessing parameters do not match the dataset schema")
        obj.means_ = np.asarray(params["means"], dtype=float)
        obj.scales_ = np.asarray(params["scales"], dtype=float)
        obj.modes_ = list(params["modes"])
        obj.n_fit_rows_ = int(params["n_fit_rows"])
        obj.fitted_on_ = str(params["fitted_on"])
        obj.pipeline_version_ = PIPELINE_VERSION
        obj.n_features_out_ = len(transformed_columns(spec))
        return obj

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SchemaPreprocessor":
        return cls.from_params(json.loads(raw.decode("utf-8")))

    # ------------------------------------------------------------- transform
    def _check(self) -> None:
        if not hasattr(self, "means_"):
            raise RuntimeError("SchemaPreprocessor is not fitted")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check()
        spec = self.spec
        missing_cols = [c for c in spec.feature_names if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input is missing columns {missing_cols}")
        n = len(X)
        parts = []
        if spec.numeric_features:
            num = X[spec.numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            num = np.where(np.isfinite(num), num, self.means_)
            parts.append((num - self.means_) / self.scales_)
        for name, mode in zip(spec.categorical_features, self.modes_):
            cats = list(spec.feature(name).categories)
            col = X[name].astype(object).where(X[name].notna(), mode).astype(str).to_numpy()
            unknown = sorted(set(col) - set(cats))
            if unknown:
                raise ValueError(f"Unknown categories for {name}: {unknown}")
            onehot = np.zeros((n, len(cats)))
            onehot[np.arange(n), [cats.index(v) for v in col]] = 1.0
            parts.append(onehot)
        return np.hstack(parts) if parts else np.zeros((n, 0))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array([c.name for c in transformed_columns(self.spec)], dtype=object)

    def numeric_reference(self, transformed: np.ndarray) -> dict[str, float]:
        """Mean of each numeric feature of ``transformed`` rows, on the original scale."""
        k = len(self.spec.numeric_features)
        raw = self.means_ + self.scales_ * transformed[:, :k].mean(axis=0)
        return dict(zip(self.spec.numeric_features, raw.tolist()))


def fit_preprocessor(spec: DatasetSpec, X_train: pd.DataFrame) -> SchemaPreprocessor:
    """Central fit on TRAINING rows only (pooled baselines, unit tests)."""
    return SchemaPreprocessor(spec.key).fit(X_train)


def transform(pre: SchemaPreprocessor, spec: DatasetSpec, X: pd.DataFrame) -> np.ndarray:
    if pre.dataset_key != spec.key:
        raise ValueError(f"Preprocessor belongs to '{pre.dataset_key}', not '{spec.key}'")
    return np.asarray(pre.transform(X[spec.feature_names]), dtype=np.float64)
