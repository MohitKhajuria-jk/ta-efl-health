"""Dataset specification shared by every clinical dataset in TA-EFL.

Every dataset exposes the same contract so the federated pipeline, the GUI
and the experiment scripts are dataset-agnostic:

    spec.load()  ->  (DataFrame with feature columns + target, source_name)

If a real CSV is present in ``data/`` it is used; otherwise a statistically
plausible synthetic cohort is generated so the app always runs offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
)


@dataclass
class FeatureSpec:
    """Display metadata for one input feature (drives the GUI prediction form)."""
    key: str
    label: str
    lo: float
    hi: float
    default: float
    decimals: int = 0
    choices: dict[int, str] | None = None   # for categorical encodings


@dataclass
class DatasetSpec:
    key: str                      # registry key, e.g. "heart"
    name: str                     # human-readable name
    task: str                     # short description of the prediction task
    target: str                   # target column name
    positive_label: str           # meaning of y = 1
    features: list[FeatureSpec]
    csv_names: list[str]          # candidate filenames under data/
    synth: Callable[[int, int], pd.DataFrame]      # (n, seed) -> DataFrame
    csv_adapter: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    builtin: Callable[[], "tuple[pd.DataFrame, str] | None"] | None = None
    n_synth: int = 1500
    citation: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- convenience views -------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return [f.key for f in self.features]

    @property
    def labels(self) -> dict[str, str]:
        return {f.key: f.label for f in self.features}

    def feature(self, key: str) -> FeatureSpec:
        return next(f for f in self.features if f.key == key)

    # ---- loading -----------------------------------------------------------
    def _find_csv(self, data_dir: str) -> str | None:
        for name in self.csv_names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                return path
        return None

    def load(self, data_dir: str | None = None,
             use_cache: bool = True) -> tuple[pd.DataFrame, str]:
        """Return ``(dataframe, source_name)``; prefers a real CSV if present."""
        data_dir = data_dir or DATA_DIR
        if use_cache and "df" in self._cache:
            return self._cache["df"].copy(), self._cache["src"]

        df = src = None
        path = self._find_csv(data_dir)
        if path is not None:
            try:
                raw = pd.read_csv(path)
                if self.csv_adapter is not None:
                    raw = self.csv_adapter(raw)
                cols = [c for c in self.feature_names if c in raw.columns]
                if self.target in raw.columns and len(cols) >= max(4, len(self.feature_names) // 2):
                    raw = raw[cols + [self.target]].apply(
                        pd.to_numeric, errors="coerce").dropna()
                    raw[self.target] = (raw[self.target] > 0).astype(int)
                    if raw[self.target].nunique() == 2 and len(raw) >= 100:
                        df, src = raw, f"Real CSV ({os.path.basename(path)})"
            except Exception as exc:                     # pragma: no cover
                print(f"[datasets] could not read {path}: {exc}; using synthetic")

        if df is None and self.builtin is not None:
            got = self.builtin()
            if got is not None:
                df, src = got

        if df is None:
            df = self.synth(self.n_synth, 42)
            src = f"Synthetic ({self.name}-like)"

        df = df.reset_index(drop=True)
        self._cache.update(df=df.copy(), src=src)
        return df, src


def logits_to_labels(logit: np.ndarray, rng, prevalence: float = 0.5,
                     sharpness: float = 1.6) -> np.ndarray:
    """Sample labels from a synthetic risk score, calibrated so that the
    expected positive rate matches ``prevalence`` (bisection on the intercept
    — a plain mean-shift misses the target once the score is skewed)."""
    z = sharpness * (logit - logit.mean())
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = (1.0 / (1.0 + np.exp(-np.clip(z + mid, -30, 30)))).mean()
        if rate < prevalence:
            lo = mid
        else:
            hi = mid
    p = 1.0 / (1.0 + np.exp(-np.clip(z + 0.5 * (lo + hi), -30, 30)))
    return rng.binomial(1, p)
