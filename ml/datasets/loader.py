"""Loading, schema validation, cleaning and splitting of genuine datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.datasets.registry import DatasetRegistry
from ml.datasets.schema import DatasetSpec


class DatasetError(Exception):
    """Raised when a dataset is missing or does not match its documented schema."""


@dataclass
class DatasetReport:
    dataset_key: str
    source: str
    n_rows_raw: int
    n_rows: int
    n_dropped_missing_target: int
    positives: int
    negatives: int
    missing_per_feature: dict[str, int]
    missing_codes_converted: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    n_duplicates: int = 0
    file_sha256: str | None = None

    @property
    def positive_rate(self) -> float:
        return self.positives / self.n_rows if self.n_rows else float("nan")


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    X: pd.DataFrame  # columns exactly spec.feature_names, cleaned (NaN = missing)
    y: np.ndarray  # int 0/1
    report: DatasetReport


@dataclass
class DataSplits:
    """Disjoint splits. Test data is used ONLY for final evaluation."""

    X_train: pd.DataFrame  # pooled client data (partitioned across hospitals)
    y_train: np.ndarray
    X_val: pd.DataFrame  # server-held validation data (preprocessor fit, trust signal)
    y_val: np.ndarray
    X_test: pd.DataFrame  # held-out test data (final evaluation only)
    y_test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.y_train), "validation": len(self.y_val), "test": len(self.y_test)}


def _read_raw(spec: DatasetSpec, registry: DatasetRegistry) -> pd.DataFrame:
    if spec.file_name is None:
        if spec.key != "breast_cancer":  # pragma: no cover - only bundled dataset
            raise DatasetError(f"No bundled source for {spec.key}")
        from sklearn.datasets import load_breast_cancer

        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        # sklearn: 0 = malignant, 1 = benign  ->  recode so 1 = malignant (risk class)
        df[spec.target] = 1 - df.pop("target").astype(int)
        return df
    avail = registry.availability(spec.key)
    if not avail.available or avail.path is None:
        raise DatasetError(avail.message)
    try:
        return pd.read_csv(avail.path, na_values=["N/A", "NA", "", "?"], keep_default_na=True)
    except Exception as exc:  # noqa: BLE001 - surface any parser problem to the user
        raise DatasetError(f"Could not read {avail.path.name}: {exc}") from exc


def validate_and_clean(spec: DatasetSpec, raw: pd.DataFrame, source: str = "") -> LoadedDataset:
    """Check the raw frame against the schema and return a cleaned dataset.

    Raises DatasetError for structural problems (missing columns, wrong target
    coding, unknown categories). Known missing-value codes become NaN; missing
    values are *not* imputed here (that is the preprocessing pipeline's job).
    """
    required = spec.feature_names + [spec.target]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise DatasetError(
            f"File does not match the {spec.name} schema. Missing columns: {missing_cols}. "
            f"Expected columns: {required}"
        )
    warnings: list[str] = []
    extra = [c for c in raw.columns if c not in required and c not in spec.drop_columns]
    if extra:
        warnings.append(f"Ignoring unexpected columns: {extra}")

    df = raw[required].copy()
    n_raw = len(df)

    # ---- target -------------------------------------------------------------
    target = pd.to_numeric(df[spec.target], errors="coerce")
    n_missing_target = int(target.isna().sum())
    df = df.loc[target.notna()].copy()
    target = target.loc[df.index]
    bad_target = sorted(set(target.unique()) - {0, 1})
    if bad_target:
        raise DatasetError(f"Target '{spec.target}' must be 0/1; found values {bad_target}")
    if n_missing_target:
        warnings.append(f"Dropped {n_missing_target} rows with missing target")

    # ---- features -----------------------------------------------------------
    missing_codes_converted: dict[str, int] = {}
    for f in spec.features:
        col = df[f.name]
        if f.kind in ("numeric", "binary"):
            num = pd.to_numeric(col, errors="coerce")
            n_unparsable = int(num.isna().sum() - col.isna().sum())
            if n_unparsable > 0:
                warnings.append(f"{f.name}: {n_unparsable} non-numeric values treated as missing")
            if f.missing_codes:
                mask = num.isin(list(f.missing_codes))
                if mask.any():
                    missing_codes_converted[f.name] = int(mask.sum())
                    num = num.mask(mask)
            if f.kind == "binary":
                bad = sorted(set(num.dropna().unique()) - {0, 1})
                if bad:
                    raise DatasetError(f"Binary feature '{f.name}' has non 0/1 values {bad}")
            df[f.name] = num.astype(float)
        else:
            s = col.astype("string").str.strip()
            allowed = set(f.categories)
            unknown = sorted(set(s.dropna().unique()) - allowed)
            if unknown:
                raise DatasetError(
                    f"Categorical feature '{f.name}' has values outside the documented "
                    f"categories {list(f.categories)}: {unknown}"
                )
            df[f.name] = s.astype(object).where(s.notna(), np.nan)

    missing = {f.name: int(df[f.name].isna().sum()) for f in spec.features}
    y = target.astype(int).to_numpy()
    if (spec.expected_positives is not None and spec.expected_rows == n_raw
            and int(y.sum()) != spec.expected_positives):
        warnings.append(f"{int(y.sum())} positive records; the documented source has "
                        f"{spec.expected_positives}. Verify you are using the genuine file.")
    if spec.expected_rows is not None and n_raw != spec.expected_rows:
        warnings.append(
            f"File has {n_raw} rows; the documented source has {spec.expected_rows}. "
            "Verify you are using the genuine file."
        )
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        raise DatasetError("Target has a single class; cannot train a classifier.")
    if len(y) < 50:
        raise DatasetError(f"Only {len(y)} usable records; at least 50 are required.")
    full = df[spec.feature_names + [spec.target]]
    n_dup = int(full.duplicated().sum())
    if n_dup:
        warnings.append(f"{n_dup} exact duplicate records found; they are kept as in the source "
                        "file (duplicates may land in different splits).")
    rate = pos / len(y)
    if rate < 0.15 or rate > 0.85:
        warnings.append(f"Class imbalance: {100 * rate:.1f}% positive. Accuracy is misleading; "
                        "consider class weighting 'balanced' and focus on ROC-AUC/recall.")
    report = DatasetReport(
        dataset_key=spec.key,
        source=source or spec.source,
        n_rows_raw=n_raw,
        n_rows=len(y),
        n_dropped_missing_target=n_missing_target,
        positives=pos,
        negatives=int(len(y) - pos),
        missing_per_feature=missing,
        missing_codes_converted=missing_codes_converted,
        warnings=warnings,
        n_duplicates=n_dup,
    )
    X = df[spec.feature_names].reset_index(drop=True)
    return LoadedDataset(spec=spec, X=X, y=y, report=report)


def load_dataset(key: str, data_dir: Path) -> LoadedDataset:
    registry = DatasetRegistry(data_dir)
    spec = registry.get(key)
    raw = _read_raw(spec, registry)
    data = validate_and_clean(spec, raw)
    av = registry.availability(key)
    if av.path is not None:
        import hashlib
        data.report.file_sha256 = hashlib.sha256(av.path.read_bytes()).hexdigest()
    return data


def make_splits(
    data: LoadedDataset, seed: int, test_fraction: float = 0.20, val_fraction: float = 0.15
) -> DataSplits:
    """Stratified train / server-validation / test split."""
    if not (0.05 <= test_fraction <= 0.5) or not (0.05 <= val_fraction <= 0.5):
        raise ValueError("test/validation fractions must be within [0.05, 0.5]")
    if test_fraction + val_fraction >= 0.8:
        raise ValueError("test + validation fractions leave too little training data")
    X, y = data.X, data.y
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=seed
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=rel_val, stratify=y_rest, random_state=seed + 1
    )
    return DataSplits(
        X_train.reset_index(drop=True), np.asarray(y_train),
        X_val.reset_index(drop=True), np.asarray(y_val),
        X_test.reset_index(drop=True), np.asarray(y_test),
    )


def dataset_summary(key: str, data_dir: Path) -> dict:
    """Everything the GUI shows about a dataset before training (no rows are returned)."""
    registry = DatasetRegistry(data_dir)
    spec = registry.get(key)
    av = registry.availability(key)
    out = {
        "key": key, "name": spec.name, "description": spec.description, "source": spec.source,
        "source_url": spec.source_url, "licence": spec.licence, "target": spec.target,
        "positive_label": spec.positive_label, "negative_label": spec.negative_label,
        "n_features": len(spec.features), "available": av.available, "message": av.message,
        "notes": list(spec.notes), "status": "missing", "error": None,
    }
    if not av.available:
        return out
    try:
        r = load_dataset(key, data_dir).report
    except DatasetError as exc:
        out.update(status="invalid", error=str(exc))
        return out
    documented = (spec.expected_rows is None
                  or (r.n_rows_raw == spec.expected_rows and r.positives == spec.expected_positives))
    out.update(
        status="valid" if documented else "valid (differs from documented counts)",
        rows=r.n_rows, rows_raw=r.n_rows_raw, positives=r.positives, negatives=r.negatives,
        positive_rate=r.positive_rate, duplicates=r.n_duplicates,
        missing={k: v for k, v in r.missing_per_feature.items() if v},
        missing_codes_converted=r.missing_codes_converted, warnings=r.warnings,
        sha256=r.file_sha256,
    )
    return out
