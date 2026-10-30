"""Application state shared by all pages (no GUI code)."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from app.configuration.settings import AppSettings, SettingsManager
from app.services.auth import LocalAuth
from experiments.store import ExperimentStore
from ml.datasets.registry import DatasetRegistry
from ml.explainability.shap_service import LocalExplanation
from ml.models.artifact import ArtifactError, ModelArtifact, load_artifact
from ml.prediction.service import PredictionResult

log = logging.getLogger("ta_efl")

EVENTS = ("settings_changed", "experiments_changed", "prediction_changed",
          "dataset_changed", "reference_mode_changed", "experiment_selected")


class AppContext:
    def __init__(self, settings_manager: SettingsManager | None = None) -> None:
        self.settings_manager = settings_manager or SettingsManager()
        self.settings: AppSettings = self.settings_manager.load()
        self.current_user: str | None = None
        self.dataset_key: str = self.settings.default_dataset
        self.selected_experiment_id: str | None = None
        self.last_prediction: PredictionResult | None = None
        self.last_explanation: LocalExplanation | None = None
        self.last_explanation_error: str | None = None
        self.training_running = False
        self._artifacts: dict[str, ModelArtifact] = {}
        self._listeners: dict[str, list[Callable[..., None]]] = defaultdict(list)

    # ---------------------------------------------------------------- events
    def on(self, event: str, cb: Callable[..., None]) -> None:
        if event not in EVENTS:
            raise KeyError(event)
        self._listeners[event].append(cb)

    def emit(self, event: str, *args: Any) -> None:
        for cb in list(self._listeners[event]):
            try:
                cb(*args)
            except Exception:  # noqa: BLE001 - one broken view must not break others
                log.exception("Listener for %s failed", event)

    # --------------------------------------------------------------- services
    @property
    def registry(self) -> DatasetRegistry:
        return DatasetRegistry(self.settings.data_path)

    @property
    def store(self) -> ExperimentStore:
        return ExperimentStore(self.settings.storage_path)

    @property
    def auth(self) -> LocalAuth:
        return LocalAuth(self.settings.storage_path)

    @property
    def reference_mode(self) -> bool:
        return self.settings.show_reference_values

    def set_reference_mode(self, on: bool) -> None:
        self.settings.show_reference_values = bool(on)
        try:
            self.settings_manager.save(self.settings)
        except Exception:  # noqa: BLE001
            log.exception("Could not persist reference-mode flag")
        self.emit("reference_mode_changed", on)

    def apply_settings(self, new: AppSettings) -> None:
        self.settings_manager.save(new)  # validates; raises SettingsError
        self.settings = new
        self._artifacts.clear()
        self.emit("settings_changed")

    def set_dataset(self, key: str) -> None:
        DatasetRegistry.get(key)
        if key != self.dataset_key:
            self.dataset_key = key
            self.last_prediction = None
            self.last_explanation = None
            self.emit("dataset_changed", key)

    def select_experiment(self, exp_id: str | None) -> None:
        self.selected_experiment_id = exp_id
        self.emit("experiment_selected", exp_id)

    # ------------------------------------------------------------ experiments
    def selected_record(self) -> dict[str, Any] | None:
        store = self.store
        if self.selected_experiment_id:
            try:
                return store.load(self.selected_experiment_id)
            except FileNotFoundError:
                self.selected_experiment_id = None
        return store.latest(None, with_model=False)

    def artifact_for(self, dataset_key: str, experiment_id: str | None = None) -> ModelArtifact:
        """Model for ``dataset_key``: the given experiment's, or the most recent one."""
        store = self.store
        rec = None
        if experiment_id:
            rec = store.load(experiment_id)
            if rec["config"]["dataset_key"] != dataset_key:
                raise ArtifactError("Selected experiment was trained on another dataset")
        else:
            rec = store.latest(dataset_key, with_model=True)
        if rec is None or not rec.get("artifact_dir"):
            raise ArtifactError(f"No trained model for '{DatasetRegistry.get(dataset_key).name}'. "
                                "Run a federated training experiment first.")
        if rec["id"] not in self._artifacts:
            self._artifacts[rec["id"]] = load_artifact(
                self.settings.storage_path / rec["artifact_dir"], expected_dataset=dataset_key)
        return self._artifacts[rec["id"]]

    def notify_experiments_changed(self) -> None:
        self.emit("experiments_changed")
