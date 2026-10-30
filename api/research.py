"""Read-only adapters and caches for existing research artifacts."""
import copy
import json
import re
import threading
from dataclasses import asdict
from pathlib import Path
from fastapi import HTTPException
from experiments.store import ExperimentStore, is_current
from ml.datasets.registry import DatasetRegistry
from ml.datasets.loader import dataset_summary
from ml.models.artifact import load_artifact

ID = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


def safe_id(value):
    if not ID.fullmatch(value):
        raise HTTPException(400, "Invalid resource identifier")
    return value


def contained(root: Path, path: Path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise HTTPException(400, "Resource is outside the configured storage")
    return resolved


def summary(rec):
    return {"id":rec["id"], "created":rec.get("created"), "status":rec.get("status"),
            "pipeline_version":rec.get("pipeline_version", 1), "current":is_current(rec),
            "config":rec.get("config", {}), "dataset":rec.get("dataset", {}),
            "test_metrics":rec.get("test_metrics", {}), "rounds_completed":rec.get("rounds_completed", 0),
            "has_model":bool(rec.get("artifact_dir")), "origin":rec.get("origin", "historical"),
            "split_config":rec.get("split_config", {}), "preprocessing":{
                "fitted_on":rec.get("preprocessing", {}).get("fitted_on")}}


class Research:
    def __init__(self, settings):
        self.settings = settings
        self.data = settings.data_path
        self.storage = settings.storage_path
        self.store = ExperimentStore(self.storage)
        self.lock = threading.RLock()
        self.prediction_lock = threading.Lock()
        self.report_lock = threading.Lock()
        self._records, self._signature = [], None
        self._datasets, self._data_signature = [], None

    def records(self):
        with self.lock:
            paths = sorted(self.store.dir.glob("*.json"))
            signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
            if signature != self._signature:
                self._records = [summary(r) for r in self.store.list()]
                self._signature = signature
            return copy.deepcopy(self._records)

    def datasets(self):
        with self.lock:
            signature = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(self.data.glob("*.csv")))
            if self._data_signature != signature:
                rows = []
                for key in DatasetRegistry.keys():
                    item = dataset_summary(key, self.data)
                    item["message"] = "Available locally" if item["available"] else "Import the documented dataset locally"
                    if item.get("error"):
                        item["error"] = "Dataset validation failed. Check the local data and documented schema."
                    item["features"] = [asdict(f) for f in DatasetRegistry.get(key).features]
                    rows.append(item)
                self._datasets, self._data_signature = rows, signature
            return copy.deepcopy(self._datasets)

    def record(self, exp_id):
        safe_id(exp_id)
        path = contained(self.store.dir, self.store.dir / f"{exp_id}.json")
        if not path.is_file():
            raise HTTPException(404, "Experiment not found")
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise HTTPException(409, "Experiment record is unreadable") from None
        if rec.get("id") != exp_id:
            raise HTTPException(409, "Experiment identity mismatch")
        return rec

    def artifact(self, exp_id):
        rec = self.record(exp_id)
        if not is_current(rec) or rec.get("status") not in {"completed", "stopped"} or not rec.get("artifact_dir"):
            raise HTTPException(409, "No usable current-pipeline model for this experiment")
        directory = contained(self.storage / "models", self.storage / rec["artifact_dir"])
        artifact = load_artifact(directory, rec["config"]["dataset_key"])
        if artifact.experiment_id != exp_id:
            raise HTTPException(409, "Model identity mismatch")
        return artifact


def protocol(rec, include_seed=True):
    """Comparisons require identical data and held-out split, not just matching labels."""
    cfg = rec["config"]
    split = rec.get("split_config", {})
    data = rec.get("dataset", {})
    if not is_current(rec) or not split or not data or rec.get("status") != "completed":
        raise HTTPException(409, "Only completed runs with current provenance can be compared")
    result = [cfg["dataset_key"], data.get("file_sha256"), data.get("rows"),
              rec.get("pipeline_version"), split.get("test_fraction"), split.get("validation_fraction"),
              split.get("stratified"), rec.get("test_metrics", {}).get("split"), cfg.get("decision_threshold", .5)]
    if include_seed:
        result.append(split.get("seed", cfg.get("seed")))
    return result
