"""TA-EFL Health desktop application entry point.

    python main.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyqt6")  # make Matplotlib use the PyQt6 binding

from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.logging.setup import configure_logging  # noqa: E402
from app.services.context import AppContext  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TA-EFL Health")
    app.setOrganizationName("TA-EFL")
    theme.apply_theme(app)  # light palette + stylesheet, independent of Windows dark mode
    ctx = AppContext()
    ctx.settings.storage_path.mkdir(parents=True, exist_ok=True)
    configure_logging(ctx.settings.storage_path, ctx.settings.log_level,
                      to_file=ctx.settings.save_experiment_logs)
    window = MainWindow(ctx)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
