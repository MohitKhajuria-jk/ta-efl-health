"""Serve a disposable copy of research history for browser tests; never copy real accounts.

Used by Playwright. Requires `npm run build` in web/. All writes stay in a new
test-results/web-qa-* directory. Original datasets are read-only inputs.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from app.configuration.settings import SettingsManager
    original = SettingsManager().load()
    output = ROOT / "test-results"
    output.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="web-qa-", dir=output))
    for name in ("experiments", "models", "personalization"):
        source = original.storage_path / name
        if source.is_dir():
            shutil.copytree(source, work / "storage" / name)
    settings = original.to_dict()
    settings.update(storage_dir=str(work / "storage"), data_dir=str(original.data_path))
    path = work / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    os.environ["TA_EFL_SETTINGS"] = str(path)
    os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
    os.environ["MPLBACKEND"] = "Agg"
    (output / "latest-web-qa.txt").write_text(str(work))
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8781, access_log=False)


if __name__ == "__main__":
    main()
