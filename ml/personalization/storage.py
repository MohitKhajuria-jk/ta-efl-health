"""Save and load personalisation runs and hospital calibrators.

Layout (separate from historical ``storage/experiments`` and ``storage/models``):

    storage/personalization/<run id>/record.json
    storage/personalization/<run id>/calibrators/<hospital id>.json
    storage/personalization/reports/<report id>.{json,md}

Calibrators are JSON (two numbers plus provenance), not pickles. A run directory is
created with ``exist_ok=False`` so an existing run can never be overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.store import atomic_write_text
from ml.datasets.registry import DatasetRegistry
from ml.personalization.calibration import CalibrationError, HospitalCalibrator
from ml.preprocessing.pipeline import PIPELINE_VERSION

DIR_NAME = "personalization"
RECORD_FILE = "record.json"
RECORD_FORMAT_VERSION = 1
SAFE = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


class PersonalizationStorageError(Exception):
    pass


def _safe(value: str, what: str) -> str:
    if not isinstance(value, str) or not SAFE.fullmatch(value):
        raise PersonalizationStorageError(f"Invalid {what} identifier")
    return value


def new_run_id(dataset_key: str, strategy: str, seed: int) -> str:
    return (f"pers-{dataset_key}-{strategy}-s{seed}-{datetime.now():%Y%m%d-%H%M%S}-"
            f"{secrets.token_hex(3)}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def indices_sha256(*arrays: list[int]) -> str:
    return hashlib.sha256(json.dumps([list(map(int, a)) for a in arrays]).encode()).hexdigest()


class PersonalizationStore:
    def __init__(self, storage_dir: Path) -> None:
        self.root = Path(storage_dir) / DIR_NAME

    def run_dir(self, run_id: str) -> Path:
        path = (self.root / _safe(run_id, "run")).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise PersonalizationStorageError("Run path is outside personalisation storage")
        return path

    # ----------------------------------------------------------------- write
    def save_run(self, record: dict[str, Any], calibrators: dict[str, dict[str, Any]]) -> Path:
        run_dir = self.run_dir(record["id"])
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)  # never overwrite an existing run
        cal_dir = run_dir / "calibrators"
        cal_dir.mkdir()
        files = {}
        for hid, payload in calibrators.items():
            _safe(hid, "hospital")
            if payload["calibrator"]["hospital_id"] != hid:
                raise PersonalizationStorageError("Calibrator file name and hospital id differ")
            path = cal_dir / f"{hid}.json"
            atomic_write_text(path, json.dumps(payload, indent=1, sort_keys=True))
            files[hid] = {"file": f"calibrators/{hid}.json", "sha256": file_sha256(path)}
        record["calibrator_files"] = files
        atomic_write_text(run_dir / RECORD_FILE, json.dumps(record, indent=1))
        return run_dir

    def save_report(self, name: str, payload: dict[str, Any], markdown: str) -> tuple[Path, Path]:
        _safe(name, "report")
        out = self.root / "reports"
        out.mkdir(parents=True, exist_ok=True)
        j, m = out / f"{name}.json", out / f"{name}.md"
        if j.exists() or m.exists():
            raise PersonalizationStorageError(f"Report {name} already exists")
        atomic_write_text(j, json.dumps(payload, indent=1))
        atomic_write_text(m, markdown)
        return j, m

    # ----------------------------------------------------------------- read
    def load_record(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / RECORD_FILE
        if not path.is_file():
            raise PersonalizationStorageError("Personalisation run not found")
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PersonalizationStorageError(f"Personalisation record is unreadable: {exc}") from None
        if rec.get("id") != run_id or rec.get("kind") != "personalization":
            raise PersonalizationStorageError("Personalisation record identity mismatch")
        return rec

    def list_records(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and SAFE.fullmatch(d.name) and (d / RECORD_FILE).is_file():
                try:
                    out.append(self.load_record(d.name))
                except PersonalizationStorageError:
                    continue
        return sorted(out, key=lambda r: r.get("created", ""), reverse=True)

    def load_calibrator(self, run_id: str, hospital_id: str, *, expected_global_id: str | None = None,
                        expected_artifact_sha256: str | None = None) -> tuple[HospitalCalibrator, dict]:
        """Load and validate one hospital's calibrator; raises on any provenance mismatch."""
        rec = self.load_record(run_id)
        _safe(hospital_id, "hospital")
        entry = rec.get("calibrator_files", {}).get(hospital_id)
        if entry is None:
            raise PersonalizationStorageError(f"{hospital_id} has no calibrator in this run")
        path = self.run_dir(run_id) / entry["file"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise PersonalizationStorageError("Calibrator file is missing or was modified")
        payload = json.loads(path.read_text(encoding="utf-8"))
        prov = payload.get("provenance", {})
        if payload.get("format_version") != RECORD_FORMAT_VERSION:
            raise PersonalizationStorageError("Unsupported calibrator file format")
        if prov.get("personalization_run_id") != run_id or prov.get("hospital_id") != hospital_id:
            raise PersonalizationStorageError("Calibrator provenance does not match this run/hospital")
        if int(prov.get("pipeline_version", -1)) != PIPELINE_VERSION:
            raise PersonalizationStorageError("Calibrator belongs to another preprocessing pipeline")
        if prov.get("schema_hash") != DatasetRegistry.get(prov["dataset_key"]).schema_hash():
            raise PersonalizationStorageError("Calibrator belongs to another dataset schema")
        if expected_global_id is not None and prov.get("global_experiment_id") != expected_global_id:
            raise PersonalizationStorageError("Calibrator was fitted for a different global model")
        if expected_artifact_sha256 is not None and prov.get("global_artifact_sha256") != expected_artifact_sha256:
            raise PersonalizationStorageError("Global model file changed since calibration")
        try:
            cal = HospitalCalibrator.from_dict(payload["calibrator"])
        except (CalibrationError, KeyError) as exc:
            raise PersonalizationStorageError(f"Calibrator is invalid: {exc}") from None
        return cal, prov
