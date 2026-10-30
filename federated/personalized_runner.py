"""Optional personalisation workflow: frozen global model + per-hospital local calibration.

This runner never trains or modifies a global model and never writes to
``storage/experiments`` or ``storage/models``. It:

1. loads an existing, current-pipeline global model and its experiment record;
2. regenerates that experiment's exact data splits (same seed and fractions) and checks
   them against the record (dataset SHA-256, split sizes and the saved test ROC-AUC);
3. takes the experiment's held-out **test split** - rows never used for federated
   training, preprocessing, trust scoring or the decision threshold - and partitions it
   into simulated hospitals with the existing partitioner (default: Dirichlet label
   skew, i.e. non-IID prevalence across hospitals);
4. splits every hospital's rows (stratified) into disjoint local calibration and local
   test subsets;
5. fits one calibrator per hospital on its own calibration rows only;
6. evaluates the global model and the personalised model on identical local test rows;
7. saves the run and calibrators under a new identifier in ``storage/personalization``,
   after confirming the global model file is byte-for-byte unchanged.
"""
from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from experiments.store import ExperimentStore, is_current
from federated.simulation.partition import partition_indices
from ml.datasets.loader import load_dataset, make_splits
from ml.evaluation.metrics import auc_or_none
from ml.models.artifact import ARTIFACT_FILE, META_FILE, load_artifact
from ml.personalization.calibration import (
    METHOD,
    CalibrationConfig,
    CalibrationError,
    InsufficientCalibrationData,
    fit_calibrator,
)
from ml.personalization.evaluator import aggregate, compare_hospital
from ml.personalization.storage import (
    RECORD_FORMAT_VERSION,
    PersonalizationStore,
    file_sha256,
    indices_sha256,
    new_run_id,
)
from ml.preprocessing.pipeline import PIPELINE_VERSION, transform


class PersonalizationError(Exception):
    pass


@dataclass
class PersonalizationConfig:
    global_experiment_id: str
    num_hospitals: int = 3
    partition: str = "dirichlet"  # "dirichlet" (non-IID label skew) | "iid"
    dirichlet_alpha: float = 0.5
    calibration_fraction: float = 0.5
    partition_seed: int | None = None  # default: the global experiment's seed
    min_hospital_rows: int = 20
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    label: str = ""

    def validate(self) -> None:
        if not 2 <= self.num_hospitals <= 10:
            raise PersonalizationError("Number of hospitals must be between 2 and 10")
        if self.partition not in ("dirichlet", "iid"):
            raise PersonalizationError("Partition must be 'dirichlet' or 'iid'")
        if self.dirichlet_alpha <= 0:
            raise PersonalizationError("Dirichlet alpha must be > 0")
        if not 0.2 <= self.calibration_fraction <= 0.8:
            raise PersonalizationError("Calibration fraction must be in [0.2, 0.8]")
        if self.min_hospital_rows < 4:
            raise PersonalizationError("Each hospital needs at least 4 rows")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["calibration"] = self.calibration.to_dict()
        return d


def _artifact_fingerprint(directory: Path) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in (ARTIFACT_FILE, META_FILE)}


def _split_hospital(idx: np.ndarray, y: np.ndarray, fraction: float, seed: int, k: int):
    y_h = y[idx]
    strat = y_h if min(np.bincount(y_h, minlength=2)) >= 2 else None
    cal, test = train_test_split(idx, train_size=fraction, stratify=strat,
                                 random_state=int(seed) * 1000 + k, shuffle=True)
    return np.sort(cal), np.sort(test), strat is not None


def run_personalization(cfg: PersonalizationConfig, data_dir: Path, storage_dir: Path,
                        save: bool = True) -> dict[str, Any]:
    cfg.validate()
    started = time.time()
    store = ExperimentStore(storage_dir)
    try:
        global_rec = store.load(cfg.global_experiment_id)
    except FileNotFoundError:
        raise PersonalizationError(f"Global experiment '{cfg.global_experiment_id}' not found") from None
    if not is_current(global_rec) or global_rec.get("status") != "completed" or not global_rec.get("artifact_dir"):
        raise PersonalizationError("Personalisation needs a completed current-pipeline experiment with a saved model")
    gcfg = global_rec["config"]
    art_dir = Path(storage_dir) / str(global_rec["artifact_dir"]).replace("\\", "/")
    fingerprint_before = _artifact_fingerprint(art_dir)
    artifact = load_artifact(art_dir, gcfg["dataset_key"])
    if artifact.experiment_id != global_rec["id"]:
        raise PersonalizationError("Global model identity mismatch")
    params_before = [a.copy() for a in artifact.model.get_params_arrays()]

    # ------------------------------------------------ regenerate and verify splits
    data = load_dataset(gcfg["dataset_key"], Path(data_dir))
    spec = data.spec
    saved_sha = global_rec.get("dataset", {}).get("file_sha256")
    if saved_sha and saved_sha != data.report.file_sha256:
        raise PersonalizationError("Dataset file differs from the one the global model used")
    split_cfg = global_rec["split_config"]
    splits = make_splits(data, int(split_cfg["seed"]), float(split_cfg["test_fraction"]),
                         float(split_cfg["validation_fraction"]))
    if splits.sizes() != global_rec["splits"]:
        raise PersonalizationError("Regenerated splits do not match the experiment record")
    X_test_t = transform(artifact.preprocessor, spec, splits.X_test)
    z_test_all = artifact.model.decision_function(X_test_t)
    saved_auc = global_rec["test_metrics"]["roc_auc"]
    check_auc = auc_or_none(splits.y_test, z_test_all)
    if saved_auc is None or check_auc is None or abs(check_auc - saved_auc) > 1e-9:
        raise PersonalizationError("Global model does not reproduce its saved test ROC-AUC; split identity "
                                   "cannot be confirmed")
    y = np.asarray(splits.y_test).astype(int)

    # ------------------------------------------------ simulated hospitals (test split only)
    seed = int(gcfg["seed"] if cfg.partition_seed is None else cfg.partition_seed)
    try:
        parts = partition_indices(y, cfg.num_hospitals, cfg.partition, seed, cfg.dirichlet_alpha,
                                  min_size=cfg.min_hospital_rows)  # type: ignore[arg-type]
    except ValueError as exc:
        raise PersonalizationError(f"Hospital partition failed: {exc}") from None
    run_id = new_run_id(spec.key, gcfg["strategy"], seed)
    threshold = float(artifact.threshold)
    hospitals, calibrators, provenance_rows = [], {}, []
    used_cal, used_test = set(), set()
    for k, idx in enumerate(parts, start=1):
        hid = f"hospital_{k}"
        cal_idx, test_idx, stratified = _split_hospital(idx, y, cfg.calibration_fraction, seed, k)
        if set(cal_idx) & set(test_idx):
            raise PersonalizationError(f"{hid}: calibration and test rows overlap")  # pragma: no cover
        used_cal |= set(cal_idx.tolist())
        used_test |= set(test_idx.tolist())
        z_cal, y_cal = z_test_all[cal_idx], y[cal_idx]
        calibrator, status, reason = None, "personalized", ""
        try:
            calibrator = fit_calibrator(hid, z_cal, y_cal, cfg.calibration)
        except InsufficientCalibrationData as exc:
            status, reason = "insufficient_calibration_data", str(exc)
        except CalibrationError as exc:
            status, reason = "calibration_rejected", str(exc)
        result = compare_hospital(hid, y[test_idx], z_test_all[test_idx], threshold, calibrator, status, reason)
        rows = {"calibration_indices": cal_idx.tolist(), "test_indices": test_idx.tolist()}
        result.update(
            n_rows=int(len(idx)), n_calibration=int(len(cal_idx)), n_test=int(len(test_idx)),
            calibration_positives=int(y_cal.sum()), test_positives=int(y[test_idx].sum()),
            prevalence=float(y[idx].mean()), stratified_split=stratified,
            calibrator=calibrator.to_dict() if calibrator else None,
            rows=rows, rows_sha256=indices_sha256(cal_idx, test_idx))
        hospitals.append(result)
        if calibrator is not None:
            calibrators[hid] = {
                "format_version": RECORD_FORMAT_VERSION,
                "calibrator": calibrator.to_dict(),
                "provenance": {
                    "personalization_run_id": run_id, "hospital_id": hid,
                    "global_experiment_id": global_rec["id"],
                    "global_artifact_sha256": fingerprint_before[ARTIFACT_FILE],
                    "dataset_key": spec.key, "schema_hash": spec.schema_hash(),
                    "dataset_file_sha256": data.report.file_sha256,
                    "pipeline_version": PIPELINE_VERSION,
                    "calibration_rows": "held-out test split of the global experiment, "
                                        f"{hid} local calibration subset",
                    "split_seed": int(split_cfg["seed"]), "partition_seed": seed,
                    "partition": cfg.partition, "dirichlet_alpha": cfg.dirichlet_alpha,
                    "calibration_fraction": cfg.calibration_fraction,
                    "n_calibration": int(len(cal_idx)), "rows_sha256": result["rows_sha256"],
                    "calibration_config": cfg.calibration.to_dict(),
                }}
    if used_cal & used_test:
        raise PersonalizationError("A row appears in one hospital's calibration and another's test set")

    # ------------------------------------------------ immutability of the global model
    params_after = artifact.model.get_params_arrays()
    fingerprint_after = _artifact_fingerprint(art_dir)
    unchanged = (fingerprint_before == fingerprint_after and
                 all(np.array_equal(a, b) for a, b in zip(params_before, params_after)))
    if not unchanged:
        raise PersonalizationError("Global model changed during personalisation; nothing saved")  # pragma: no cover

    record: dict[str, Any] = {
        "id": run_id, "kind": "personalization", "format_version": RECORD_FORMAT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"), "status": "completed",
        "method": METHOD,
        "formula": "P_personalized = sigmoid(a_h * z_global + b_h); z_global = global model log-odds",
        "description": ("Local probability calibration of a frozen global model. Not local fine-tuning "
                        "and not personalised feature learning."),
        "config": cfg.to_dict(),
        "global_experiment": {
            "id": global_rec["id"], "strategy": gcfg["strategy"],
            "strategy_label": global_rec.get("strategy_label"), "attack_type": gcfg.get("attack_type"),
            "malicious_clients": gcfg.get("malicious_clients", []), "seed": int(gcfg["seed"]),
            "label": gcfg.get("label", ""), "artifact_dir": str(global_rec["artifact_dir"]).replace("\\", "/"),
            "artifact_sha256": fingerprint_before, "saved_test_roc_auc": saved_auc,
            "decision_threshold": threshold},
        "dataset": {"key": spec.key, "name": spec.name, "file_sha256": data.report.file_sha256,
                    "schema_hash": spec.schema_hash()},
        "pipeline_version": PIPELINE_VERSION,
        "data_protocol": {
            "source_split": "held-out test split of the global experiment (never used for federated "
                            "training, preprocessing, trust scoring or threshold selection)",
            "split_seed": int(split_cfg["seed"]), "test_fraction": split_cfg["test_fraction"],
            "validation_fraction": split_cfg["validation_fraction"],
            "n_source_rows": int(len(y)), "partition": cfg.partition, "partition_seed": seed,
            "dirichlet_alpha": cfg.dirichlet_alpha, "num_hospitals": cfg.num_hospitals,
            "calibration_fraction": cfg.calibration_fraction,
            "server_validation_split_used": False, "threshold_tuned": False},
        "hospitals": hospitals,
        "aggregate": aggregate(hospitals, threshold),
        "integrity": {
            "global_model_unchanged": unchanged,
            "global_artifact_sha256_before": fingerprint_before,
            "global_artifact_sha256_after": fingerprint_after,
            "test_split_reproduces_saved_auc": True, "recomputed_test_roc_auc": check_auc,
            "calibration_test_disjoint": True,
            "rows_used": int(len(used_cal) + len(used_test)),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    record["integrity"]["partition_sha256"] = hashlib.sha256(
        "|".join(h["rows_sha256"] for h in hospitals).encode()).hexdigest()
    if save:
        PersonalizationStore(storage_dir).save_run(record, calibrators)
    return record
