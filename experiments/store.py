"""Persistence of experiment records (JSON, one file per experiment)."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def is_current(record: dict[str, Any]) -> bool:
    """True if the record was produced by the current preprocessing pipeline."""
    from ml.preprocessing.pipeline import PIPELINE_VERSION
    return int(record.get("pipeline_version", 1)) == PIPELINE_VERSION


STALE_NOTE = ("Produced by preprocessing pipeline v1, which fitted the preprocessing on the "
              "validation split. Its metrics are superseded; rerun the experiment.")


class ExperimentStore:
    def __init__(self, storage_dir: Path) -> None:
        self.dir = Path(storage_dir) / "experiments"

    def save(self, record: dict[str, Any]) -> Path:
        path = self.dir / f"{record['id']}.json"
        atomic_write_text(path, json.dumps(record, indent=1, default=_json_default))
        return path

    def load(self, exp_id: str) -> dict[str, Any]:
        path = self.dir / f"{exp_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Experiment '{exp_id}' not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, dataset_key: str | None = None) -> list[dict[str, Any]]:
        """Return experiment records, newest first. Unreadable files are skipped."""
        out = []
        if not self.dir.is_dir():
            return out
        for p in self.dir.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if dataset_key and rec.get("config", {}).get("dataset_key") != dataset_key:
                continue
            out.append(rec)
        out.sort(key=lambda r: r.get("created", ""), reverse=True)
        return out

    def latest(self, dataset_key: str | None = None, with_model: bool = True) -> dict[str, Any] | None:
        """Newest completed/stopped experiment produced by the CURRENT pipeline."""
        for rec in self.list(dataset_key):
            if not is_current(rec):
                continue
            if rec.get("status") in ("completed", "stopped") and (not with_model or rec.get("artifact_dir")):
                return rec
        return None

    def list_current(self, dataset_key: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.list(dataset_key) if is_current(r)]

    def delete(self, exp_id: str) -> None:
        path = self.dir / f"{exp_id}.json"
        if path.is_file():
            path.unlink()
