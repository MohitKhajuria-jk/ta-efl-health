"""GUI actions for obtaining dataset files (shared by the Training and Settings pages)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from app.services.context import AppContext
from app.ui.widgets.common import confirm, show_error
from ml.datasets.importer import import_dataset_file
from ml.datasets.loader import DatasetError
from ml.datasets.registry import DatasetRegistry


def open_download_page(key: str) -> None:
    QDesktopServices.openUrl(QUrl(DatasetRegistry.get(key).source_url))


def import_dataset_dialog(parent: QWidget, ctx: AppContext, key: str) -> bool:
    """Ask for the downloaded CSV/zip, validate it and copy it into the data folder."""
    spec = DatasetRegistry.get(key)
    start = Path.home() / "Downloads"
    path, _ = QFileDialog.getOpenFileName(
        parent, f"Import {spec.name} ({spec.file_name})",
        str(start if start.is_dir() else Path.home()), "Dataset file (*.csv *.zip)")
    if not path:
        return False
    data_dir = ctx.settings.data_path
    overwrite = False
    if (data_dir / spec.file_name).exists():
        if not confirm(parent, "Replace dataset file",
                       f"{spec.file_name} already exists in {data_dir}. Replace it?"):
            return False
        overwrite = True
    try:
        res = import_dataset_file(key, Path(path), data_dir, overwrite=overwrite)
    except DatasetError as exc:
        show_error(parent, "Dataset not imported",
                   f"The file was not imported because it does not match the {spec.name} "
                   f"schema.\n\n{exc}")
        return False
    msg = (f"Imported {res.rows} records ({res.positives} positive) to\n{res.path}")
    if not res.matches_documented_counts:
        msg += (f"\n\nNote: the documented genuine file has {spec.expected_rows} rows and "
                f"{spec.expected_positives} positive records. Check that this is the original "
                f"download from {spec.source_url}.")
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information if res.matches_documented_counts
                else QMessageBox.Icon.Warning)
    box.setWindowTitle("Dataset imported")
    box.setText(msg)
    box.exec()
    ctx.emit("dataset_changed", key)
    return True
