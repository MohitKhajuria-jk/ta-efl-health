"""Obtain and verify the genuine dataset files (run on your own computer).

    python scripts\\fetch_datasets.py              # all missing datasets
    python scripts\\fetch_datasets.py --only diabetes

* Pima Indians Diabetes: downloaded from a pinned commit of the public GitHub mirror of
  the original UCI distribution (jbrownlee/Datasets), verified by SHA-256, and the
  documented column names (identical to Kaggle's diabetes.csv) are added.
* Heart, Framingham and Stroke are hosted on Kaggle and need a (free) Kaggle account.
  If the Kaggle CLI is installed and configured (~/.kaggle/kaggle.json) it is used;
  otherwise exact manual steps are printed.
Every file is then validated against its schema and the documented row/class counts.
Nothing is invented or substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configuration.settings import SettingsManager  # noqa: E402
from ml.datasets.loader import DatasetError, load_dataset  # noqa: E402
from ml.datasets.registry import DatasetRegistry  # noqa: E402

PIMA_URL = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
            "d20fcb6402ae34e653d4513b00f39257bb37ed7f/pima-indians-diabetes.data.csv")
PIMA_SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
PIMA_HEADER = ("Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,"
               "DiabetesPedigreeFunction,Age,Outcome")

KAGGLE = {
    "heart": ("fedesoriano/heart-failure-prediction", "heart.csv"),
    "diabetes": ("uciml/pima-indians-diabetes-database", "diabetes.csv"),
    "framingham": ("aasheesh200/framingham-heart-study-dataset", "framingham.csv"),
    "stroke": ("fedesoriano/stroke-prediction-dataset", "healthcare-dataset-stroke-data.csv"),
}


def convert_pima(raw_bytes: bytes, out: Path) -> None:
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != PIMA_SHA256:
        raise SystemExit(f"Pima file checksum mismatch ({digest}); refusing to use it.")
    text = raw_bytes.decode("utf-8").strip().replace("\r\n", "\n")
    out.write_text(PIMA_HEADER + "\n" + text + "\n", encoding="utf-8")


def fetch_pima(data_dir: Path) -> bool:
    out = data_dir / "diabetes.csv"
    print(f"Downloading Pima Indians Diabetes from\n  {PIMA_URL}")
    try:
        with urllib.request.urlopen(PIMA_URL, timeout=60) as resp:  # noqa: S310 - fixed https URL
            convert_pima(resp.read(), out)
    except OSError as exc:
        print(f"  download failed: {exc}")
        return False
    print(f"  saved {out}")
    return True


def fetch_kaggle(key: str, data_dir: Path) -> bool:
    slug, fname = KAGGLE[key]
    if shutil.which("kaggle") is None:
        print(f"[manual] {DatasetRegistry.get(key).name}:\n"
              f"  1. Sign in at https://www.kaggle.com/datasets/{slug}\n"
              f"  2. Click Download, unzip, and copy '{fname}' to {data_dir}\n"
              f"  or install the Kaggle CLI (pip install kaggle, add kaggle.json) and run:\n"
              f"     kaggle datasets download -d {slug} -p \"{data_dir}\" --unzip")
        return False
    print(f"kaggle datasets download -d {slug} --unzip")
    res = subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(data_dir),
                          "--unzip"], check=False)
    return res.returncode == 0 and (data_dir / fname).is_file()


def main() -> int:
    settings = SettingsManager().load()
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=settings.data_path)
    p.add_argument("--only", choices=list(KAGGLE), default=None)
    p.add_argument("--pima-from-kaggle", action="store_true",
                   help="use Kaggle instead of the pinned GitHub mirror for diabetes.csv")
    a = p.parse_args()
    a.data_dir.mkdir(parents=True, exist_ok=True)
    reg = DatasetRegistry(a.data_dir)
    keys = [a.only] if a.only else list(KAGGLE)
    for key in keys:
        if reg.availability(key).available:
            print(f"[present] {DatasetRegistry.get(key).name}")
            continue
        if key == "diabetes" and not a.pima_from_kaggle:
            fetch_pima(a.data_dir)
        else:
            fetch_kaggle(key, a.data_dir)
    print("\nValidation:")
    ok = True
    for key in DatasetRegistry.keys():
        spec = DatasetRegistry.get(key)
        if not reg.availability(key).available:
            print(f"  [MISSING] {spec.name}")
            continue
        try:
            r = load_dataset(key, a.data_dir).report
        except DatasetError as exc:
            ok = False
            print(f"  [INVALID] {spec.name}: {exc}")
            continue
        exact = (r.n_rows_raw == spec.expected_rows and r.positives == spec.expected_positives)
        print(f"  [{'OK' if exact else 'CHECK'}] {spec.name}: {r.n_rows} rows, {r.positives} positive"
              + ("" if exact else f" (documented: {spec.expected_rows} rows, "
                 f"{spec.expected_positives} positive)"))
        if r.file_sha256:
            print(f"        sha256 {r.file_sha256}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
