"""Registry of every clinical dataset available to TA-EFL."""
from __future__ import annotations

import pandas as pd

from .base import DatasetSpec, FeatureSpec, DATA_DIR
from . import heart, diabetes, breast_cancer, framingham, stroke

DATASETS: dict[str, DatasetSpec] = {
    s.key: s for s in (
        heart.SPEC, diabetes.SPEC, breast_cancer.SPEC,
        framingham.SPEC, stroke.SPEC,
    )
}
DEFAULT_DATASET = "heart"


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())


def get_spec(key: str | DatasetSpec | None) -> DatasetSpec:
    if isinstance(key, DatasetSpec):
        return key
    if key is None:
        key = DEFAULT_DATASET
    key = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    if key in DATASETS:
        return DATASETS[key]
    for spec in DATASETS.values():                 # allow human-readable names
        if spec.name.lower().replace(" ", "_") == key:
            return spec
    raise KeyError(f"Unknown dataset '{key}'. Available: {sorted(DATASETS)}")


def load(key: str | None = None,
         data_dir: str | None = None) -> tuple[pd.DataFrame, DatasetSpec, str]:
    """Return ``(dataframe, spec, source_name)`` for the requested dataset."""
    spec = get_spec(key)
    df, src = spec.load(data_dir)
    return df, spec, src


# Merged label map, handy for generic UI code.
ALL_FEATURE_LABELS: dict[str, str] = {}
for _s in DATASETS.values():
    ALL_FEATURE_LABELS.update(_s.labels)
