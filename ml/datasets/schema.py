"""Typed dataset/feature schema definitions.

A schema is the single source of truth for a dataset's columns, their types,
the values the application accepts for prediction, and how known data-quality
issues (e.g. ``0`` used as a missing-value code) are handled.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

FeatureKind = Literal["numeric", "binary", "categorical"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one input feature."""

    name: str  # column name in the source file
    label: str  # human-readable label shown in the GUI and explanations
    kind: FeatureKind
    # numeric / binary: supported input range for *prediction* validation
    min_value: float | None = None
    max_value: float | None = None
    integer: bool = False
    unit: str = ""
    # categorical / binary: allowed raw values and their display labels
    categories: tuple[str, ...] = ()
    category_labels: tuple[str, ...] = ()
    # raw values that encode "missing" in the source data (converted to NaN)
    missing_codes: tuple[float, ...] = ()
    help: str = ""

    def display_value(self, raw: object) -> str:
        """Return a human-readable rendering of a raw value."""
        if raw is None:
            return "missing"
        if self.kind == "categorical" and self.category_labels:
            for code, lab in zip(self.categories, self.category_labels):
                if str(raw) == code:
                    return lab
        if self.kind == "binary" and self.category_labels:
            try:
                return self.category_labels[int(float(raw))]
            except (ValueError, IndexError):
                return str(raw)
        if isinstance(raw, float):
            txt = f"{raw:g}"
        else:
            txt = str(raw)
        return f"{txt} {self.unit}".strip()


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one supported healthcare dataset."""

    key: str
    name: str
    description: str
    source: str  # where the genuine data comes from
    source_url: str
    file_name: str | None  # expected CSV file name in the data directory; None = bundled
    target: str
    positive_label: str  # meaning of target == 1
    negative_label: str  # meaning of target == 0
    features: tuple[FeatureSpec, ...]
    drop_columns: tuple[str, ...] = ()
    expected_rows: int | None = None  # documented size of the genuine file (for warnings)
    expected_positives: int | None = None  # documented number of target == 1 rows
    licence: str = ""  # licence / terms note for the README and dataset panel
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(f"Unknown feature '{name}' for dataset '{self.key}'")

    @property
    def numeric_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind in ("numeric", "binary")]

    @property
    def categorical_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "categorical"]

    def schema_hash(self) -> str:
        """Stable hash of the schema; stored in artifacts to block cross-dataset use."""
        payload = {
            "key": self.key,
            "target": self.target,
            "features": [asdict(f) for f in self.features],
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]
