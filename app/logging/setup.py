"""Application logging. Patient-level inputs are never written to the log."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(storage_dir: Path, level: str = "INFO", to_file: bool = True) -> Path | None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in list(root.handlers):
        if getattr(h, "_ta_efl", False):
            root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console._ta_efl = True  # type: ignore[attr-defined]
    root.addHandler(console)
    if not to_file:
        return None
    log_dir = Path(storage_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "ta_efl_health.log"
    fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh._ta_efl = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    return path
