"""Trained model artifacts: model + preprocessing + schema + threshold, saved together.

Artifacts are written with joblib. joblib files are pickles: only load artifacts
this application created (the loader verifies a metadata sidecar and the schema
hash, but that is an integrity check, not a security boundary).
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from ml.datasets.registry import DatasetRegistry
from ml.datasets.schema import DatasetSpec
from ml.models.logistic import MiniBatchLogisticRegression
from ml.preprocessing.pipeline import PIPELINE_VERSION, SchemaPreprocessor

ARTIFACT_FILE = "model.joblib"
META_FILE = "model_meta.json"


class ArtifactError(Exception):
    pass


@dataclass
class ModelArtifact:
    dataset_key: str
    schema_hash: str
    feature_names: list[str]
    transformed_names: list[str]
    target: str
    class_labels: dict[int, str]  # {0: negative label, 1: positive label}
    threshold: float
    threshold_policy: str
    model: MiniBatchLogisticRegression
    preprocessor: SchemaPreprocessor
    background: np.ndarray  # transformed server-validation rows (SHAP background)
    experiment_id: str
    model_description: str
    metrics: dict[str, Any] = field(default_factory=dict)  # test metrics at creation
    evaluation_split: str = "test"
    pipeline_version: int = PIPELINE_VERSION
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    versions: dict[str, str] = field(default_factory=lambda: {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    })

    @property
    def spec(self) -> DatasetSpec:
        return DatasetRegistry.get(self.dataset_key)

    def metadata(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("model", "preprocessor", "background")}
        d["class_labels"] = {str(k): v for k, v in self.class_labels.items()}
        d["n_background"] = int(len(self.background))
        return d


def save_artifact(artifact: ModelArtifact, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ARTIFACT_FILE
    joblib.dump(artifact, path)
    (directory / META_FILE).write_text(json.dumps(artifact.metadata(), indent=2, default=str),
                                       encoding="utf-8")
    return path


def load_artifact(directory: Path, expected_dataset: str | None = None) -> ModelArtifact:
    path = directory / ARTIFACT_FILE
    meta_path = directory / META_FILE
    if not path.is_file() or not meta_path.is_file():
        raise ArtifactError(f"No trained model artifact in {directory}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Model metadata is unreadable: {exc}") from exc
    if int(meta.get("pipeline_version", 1)) != PIPELINE_VERSION:
        raise ArtifactError(
            "This model was trained with an earlier preprocessing pipeline (version "
            f"{meta.get('pipeline_version', 1)}, fitted on validation data) and is not used. "
            "Retrain it with the current version.")
    try:
        obj = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - corrupted / incompatible pickle
        raise ArtifactError(f"Model artifact could not be read ({type(exc).__name__}: {exc}); "
                            "retrain the model") from exc
    if not isinstance(obj, ModelArtifact):
        raise ArtifactError("File is not a TA-EFL Health model artifact")
    if getattr(obj, "pipeline_version", 1) != PIPELINE_VERSION or not isinstance(
            obj.preprocessor, SchemaPreprocessor):
        raise ArtifactError("Artifact uses an incompatible preprocessing pipeline; retrain it")
    if obj.experiment_id != meta.get("experiment_id"):
        raise ArtifactError("Artifact and metadata sidecar do not match")
    spec = DatasetRegistry.get(obj.dataset_key)
    if obj.schema_hash != spec.schema_hash():
        raise ArtifactError(
            "Artifact was trained on a different version of the dataset schema; retrain it.")
    if expected_dataset is not None and obj.dataset_key != expected_dataset:
        raise ArtifactError(
            f"Model was trained on '{obj.dataset_key}', not '{expected_dataset}'. "
            "A model is never used to predict another dataset's outcome.")
    return obj
