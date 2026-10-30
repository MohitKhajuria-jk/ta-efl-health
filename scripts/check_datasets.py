"""Validate the dataset files in the data directory against their documented schemas.

Usage (PowerShell, from the project folder):
    python scripts/check_datasets.py
    python scripts/check_datasets.py --data-dir D:\\datasets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configuration.settings import SettingsManager  # noqa: E402
from ml.datasets.loader import DatasetError, load_dataset  # noqa: E402
from ml.datasets.registry import DatasetRegistry  # noqa: E402


def main() -> int:
    settings = SettingsManager().load()
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=settings.data_path)
    a = p.parse_args()
    reg = DatasetRegistry(a.data_dir)
    print(f"Data directory: {a.data_dir.resolve()}\n")
    ok = True
    for av in reg.all_availability():
        s = av.spec
        if not av.available:
            print(f"[MISSING] {s.name}\n          expected file: {s.file_name}\n"
                  f"          source: {s.source_url}\n")
            continue
        try:
            d = load_dataset(s.key, a.data_dir)
            r = d.report
            print(f"[OK]      {s.name}: {r.n_rows} rows, {r.positives} positive "
                  f"({100 * r.positive_rate:.1f}%)")
            for k, v in r.missing_codes_converted.items():
                print(f"          {k}: {v} documented missing-value codes -> missing")
            for w in r.warnings:
                print(f"          warning: {w}")
            print()
        except DatasetError as exc:
            ok = False
            print(f"[INVALID] {s.name}: {exc}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
