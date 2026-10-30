"""Validated predictions from a saved model artifact."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.schema import DatasetSpec, FeatureSpec
from ml.models.artifact import ModelArtifact
from ml.preprocessing.pipeline import transform


class InputValidationError(ValueError):
    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def _parse_feature(f: FeatureSpec, raw: Any) -> Any:
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        raise ValueError("required")
    if f.kind == "categorical":
        s = str(raw).strip()
        if s in f.categories:
            return s
        for code, lab in zip(f.categories, f.category_labels):
            if s.lower() == lab.lower():
                return code
        raise ValueError(f"must be one of {', '.join(f.category_labels or f.categories)}")
    if f.kind == "binary":
        s = str(raw).strip()
        for i, lab in enumerate(f.category_labels):
            if s.lower() == lab.lower():
                return float(i)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError("must be a number") from None
    if not math.isfinite(v):
        raise ValueError("must be a finite number")
    if f.kind == "binary" and v not in (0.0, 1.0):
        raise ValueError("must be 0 or 1")
    if f.integer and abs(v - round(v)) > 1e-9:
        raise ValueError("must be a whole number")
    if f.min_value is not None and v < f.min_value:
        raise ValueError(f"must be ≥ {f.min_value:g}")
    if f.max_value is not None and v > f.max_value:
        raise ValueError(f"must be ≤ {f.max_value:g}")
    return v


def validate_inputs(spec: DatasetSpec, values: dict[str, Any]) -> dict[str, Any]:
    """Return cleaned values for every schema feature or raise InputValidationError."""
    errors: dict[str, str] = {}
    clean: dict[str, Any] = {}
    unknown = set(values) - set(spec.feature_names)
    for name in sorted(unknown):
        errors[name] = "is not a feature of this dataset"
    for f in spec.features:
        try:
            clean[f.name] = _parse_feature(f, values.get(f.name))
        except ValueError as exc:
            errors[f.name] = f"{f.label} {exc}"
    if errors:
        raise InputValidationError(errors)
    return clean


@dataclass
class PredictionResult:
    dataset_key: str
    dataset_name: str
    experiment_id: str
    model_description: str
    inputs: dict[str, Any]
    x_transformed: np.ndarray  # shape (d,)
    decision_value: float  # model log-odds for the positive class
    probability_positive: float  # model probability (not a calibrated clinical risk)
    threshold: float
    predicted_class: int
    predicted_label: str
    class_labels: dict[int, str]
    notes: list[str] = field(default_factory=list)

    @property
    def headline(self) -> str:
        return "High Risk" if self.predicted_class == 1 else "Low Risk"


def predict(artifact: ModelArtifact, values: dict[str, Any]) -> PredictionResult:
    spec = artifact.spec
    clean = validate_inputs(spec, values)
    row = pd.DataFrame([{k: (np.nan if v is None else v) for k, v in clean.items()}],
                       columns=spec.feature_names)
    for f in spec.features:  # numeric dtype for numeric columns, object for categorical
        if f.kind in ("numeric", "binary"):
            row[f.name] = row[f.name].astype(float)
    x = transform(artifact.preprocessor, spec, row)
    model = artifact.model
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("This model does not provide probabilities")
    z = float(model.decision_function(x)[0])
    p = float(model.predict_proba(x)[0, 1])
    cls = int(p >= artifact.threshold)
    notes = [
        "The probability is the model's estimate from its training data; it is not a "
        "clinically calibrated risk and this tool is not a diagnostic device.",
    ]
    return PredictionResult(
        dataset_key=spec.key, dataset_name=spec.name, experiment_id=artifact.experiment_id,
        model_description=artifact.model_description, inputs=clean, x_transformed=x[0],
        decision_value=z, probability_positive=p, threshold=artifact.threshold,
        predicted_class=cls, predicted_label=artifact.class_labels[cls],
        class_labels=artifact.class_labels, notes=notes,
    )
