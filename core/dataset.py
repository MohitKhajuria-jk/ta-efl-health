"""Backward-compatible facade over the multi-dataset registry.

Historically this module exposed a single heart-disease dataset. It now
delegates to :mod:`core.datasets`, so existing imports keep working while
new code should prefer ``from core.datasets import load, get_spec``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .datasets import (                                   # noqa: F401
    DATASETS, DEFAULT_DATASET, ALL_FEATURE_LABELS, DatasetSpec, FeatureSpec,
    get_spec, list_datasets,
)
from .datasets import load as load_registry

TARGET = "target"
FEATURES = DATASETS["heart"].feature_names
FEATURE_LABELS = dict(ALL_FEATURE_LABELS)      # union across all datasets


def load_dataset(data_dir: str | None = None,
                 dataset: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source_name)`` for ``dataset`` (default: heart)."""
    df, _spec, src = load_registry(dataset or DEFAULT_DATASET, data_dir)
    return df, src


def standardize(train_X: np.ndarray, *others: np.ndarray):
    """Z-score using training statistics; returns (mean, std, scaled arrays...)."""
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0) + 1e-8
    out = [(train_X - mu) / sd] + [(o - mu) / sd for o in others]
    return mu, sd, out
