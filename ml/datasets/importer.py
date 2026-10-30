"""Import a user-downloaded dataset file (CSV or the Kaggle .zip) into the data folder.

The file is validated against the dataset's schema *before* it is copied, under the
exact expected file name. Nothing is renamed into place unless its contents match.
"""
from __future__ import annotations

import io
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml.datasets.loader import DatasetError, validate_and_clean
from ml.datasets.registry import DatasetRegistry


@dataclass
class ImportResult:
    dataset_key: str
    path: Path
    rows: int
    positives: int
    matches_documented_counts: bool
    warnings: list[str]


def _read_source(src: Path, expected_name: str) -> tuple[bytes, str]:
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
            exact = [n for n in csvs if Path(n).name == expected_name]
            if exact:
                name = exact[0]
            elif len(csvs) == 1:
                name = csvs[0]
            else:
                raise DatasetError(f"The zip does not contain '{expected_name}' "
                                   f"(found: {', '.join(csvs) or 'no CSV files'})")
            return zf.read(name), Path(name).name
    if src.suffix.lower() != ".csv":
        raise DatasetError("Choose the downloaded .csv file or the Kaggle .zip archive")
    return src.read_bytes(), src.name


def import_dataset_file(key: str, src: Path, data_dir: Path, overwrite: bool = False) -> ImportResult:
    spec = DatasetRegistry.get(key)
    if spec.file_name is None:
        raise DatasetError(f"{spec.name} is bundled with scikit-learn; nothing to import")
    src = Path(src)
    if not src.is_file():
        raise DatasetError(f"File not found: {src}")
    raw_bytes, _ = _read_source(src, spec.file_name)
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), na_values=["N/A", "NA", "", "?"])
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read the file as CSV: {exc}") from exc
    data = validate_and_clean(spec, df, source=str(src))  # raises DatasetError on mismatch
    r = data.report
    matches = (spec.expected_rows is None
               or (r.n_rows_raw == spec.expected_rows and r.positives == spec.expected_positives))
    dest = Path(data_dir) / spec.file_name
    if dest.exists() and not overwrite:
        raise DatasetError(f"{dest} already exists; choose overwrite to replace it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".importing")
    tmp.write_bytes(raw_bytes)
    shutil.move(str(tmp), str(dest))
    return ImportResult(key, dest, r.n_rows, r.positives, matches, list(r.warnings))
