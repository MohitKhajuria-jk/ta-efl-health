"""GUI tests (pytest-qt). Skipped automatically when PyQt6/pytest-qt are not installed.

On Windows they run against a real Qt event loop; on a headless Linux CI set
QT_QPA_PLATFORM=offscreen.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")
os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtCore import Qt  # noqa: E402

from app.configuration.settings import AppSettings, SettingsManager  # noqa: E402
from app.services.context import AppContext  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.widgets.sidebar import NAV_ITEMS  # noqa: E402
from app.ui.workers import ExperimentWorker  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path):
    m = SettingsManager(tmp_path / "settings.json")
    # Isolated, empty data folder: tests must not depend on which genuine dataset files the
    # developer has imported (breast cancer is bundled, so it is always available).
    (tmp_path / "data").mkdir(exist_ok=True)
    m.save(AppSettings(storage_dir=str(tmp_path / "storage"), data_dir=str(tmp_path / "data"),
                       default_dataset="breast_cancer",
                       num_rounds=3, malicious_simulation_enabled=False))
    ctx = AppContext(m)
    from PyQt6.QtWidgets import QApplication
    theme.apply_theme(QApplication.instance())  # exactly what main.py applies
    w = MainWindow(ctx)
    qtbot.addWidget(w)
    w.show()
    return w


def _login(qtbot, w):
    w.login.username.setText("tester")
    w.login.password.setText("password-123")
    w.login.confirm.setText("password-123")
    qtbot.mouseClick(w.login.submit, Qt.MouseButton.LeftButton)
    assert w.root.currentIndex() == 1


def test_first_run_account_and_navigation(qtbot, window):
    assert window.login.first_run
    _login(qtbot, window)
    for key, _text, _icon in NAV_ITEMS:
        qtbot.mouseClick(window.sidebar.buttons[key], Qt.MouseButton.LeftButton)
        assert window.pages.currentWidget() is window.page_map[key]
        assert window.sidebar.buttons[key].isChecked()


def test_wrong_password_rejected(qtbot, window):
    _login(qtbot, window)
    window._logout()
    window.login.username.setText("tester")
    window.login.password.setText("nope-nope")
    qtbot.mouseClick(window.login.submit, Qt.MouseButton.LeftButton)
    assert window.root.currentIndex() == 0
    assert "Invalid" in window.login.error.text.text()


def test_reference_mode_is_labelled(qtbot, window):
    _login(qtbot, window)
    window.sidebar.reference_box.setChecked(True)
    dash = window.page_map["dashboard"]
    assert dash.reference_banner.isVisible()
    assert dash.m_auc.value.text() == "0.805"
    window.sidebar.reference_box.setChecked(False)
    assert not dash.reference_banner.isVisible()
    assert dash.m_auc.value.text() == "Not run"  # no experiment has run: no number is shown


def test_background_worker_reports_errors(qtbot):
    """Regression for 'TypeError: native Qt signal is not callable' (signal named 'event')."""
    def failing(on_event, stop_event):
        on_event({"type": "status", "message": "about to fail"})
        raise RuntimeError("simulated failure")

    w = ExperimentWorker(failing)
    events, failures = [], []
    w.progress.connect(events.append)
    w.failed.connect(lambda m, d: failures.append((m, d)))
    with qtbot.waitSignal(w.done, timeout=10000):
        w.start()
    assert w.wait(5000)
    qtbot.waitUntil(lambda: bool(failures), timeout=5000)
    assert "simulated failure" in failures[0][0] and "Traceback" in failures[0][1]
    assert events and events[0]["message"] == "about to fail"
    assert callable(w.event)  # QObject.event() is not shadowed


def test_background_worker_success_and_cancellation(qtbot):
    def job(on_event, stop_event):
        for i in range(200):
            if stop_event.is_set():
                return "stopped"
            on_event({"i": i})
            stop_event.wait(0.01)
        return "finished"

    w = ExperimentWorker(job)
    results = []
    w.succeeded.connect(results.append)
    w.progress.connect(lambda e: e.get("i") == 3 and w.stop())
    with qtbot.waitSignal(w.done, timeout=20000):
        w.start()
    w.wait(5000)
    qtbot.waitUntil(lambda: bool(results), timeout=5000)
    assert results == ["stopped"]
    assert not w.is_running()


def test_thread_pool_job_reports_success_and_error(qtbot):
    from app.ui.workers import run_in_background

    done, errors = [], []
    run_in_background(lambda: 41 + 1, on_done=done.append)
    run_in_background(lambda: 1 / 0, on_error=lambda m, d: errors.append(m))
    qtbot.waitUntil(lambda: done == [42] and len(errors) == 1, timeout=10000)
    assert "ZeroDivisionError" in errors[0]


def test_gui_training_prediction_and_explanation(qtbot, window):
    _login(qtbot, window)
    window.navigate("training")
    page = window.page_map["training"]
    page.dataset.setCurrentIndex(page.dataset.findData("breast_cancer"))
    page.rounds.setValue(3)
    page.malicious.setCurrentIndex(0)
    qtbot.mouseClick(page.start_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page.worker is None and page.start_btn.isEnabled(), timeout=120000)
    assert page.status_badge.text() == "completed"
    assert "Round 3 / 3" in page.round_label.text()

    window.navigate("prediction")
    pred = window.page_map["prediction"]
    from ml.datasets.loader import load_dataset, make_splits
    d = load_dataset("breast_cancer", Path("data"))
    row = make_splits(d, 42).X_test.iloc[0].to_dict()
    pred._apply_values(row)
    qtbot.mouseClick(pred.predict_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.ctx.last_prediction is not None, timeout=20000)
    assert pred.explain_btn.isEnabled()
    qtbot.mouseClick(pred.explain_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.ctx.last_explanation is not None, timeout=30000)
    assert window.pages.currentWidget() is window.page_map["explain"]


def test_invalid_prediction_input_marked(qtbot, window):
    _login(qtbot, window)
    window.navigate("prediction")
    pred = window.page_map["prediction"]
    pred._spec = None
    pred._build_form(window.ctx.registry.get("breast_cancer"))
    pred._predict()
    assert pred.form_errors.text()  # every field is required


def test_training_page_shows_dataset_status_and_blocks_missing(qtbot, window):
    _login(qtbot, window)
    window.navigate("training")
    page = window.page_map["training"]
    page.dataset.setCurrentIndex(page.dataset.findData("breast_cancer"))
    qtbot.waitUntil(lambda: page.ds_status.text() == "valid", timeout=20000)
    assert "569" in page.ds_text.text() and page.start_btn.isEnabled()
    page.dataset.setCurrentIndex(page.dataset.findData("stroke"))
    qtbot.waitUntil(lambda: page.ds_status.text() == "file missing", timeout=20000)
    assert not page.start_btn.isEnabled()  # never trains on another dataset instead


def test_missing_model_state_on_prediction_page(qtbot, window):
    _login(qtbot, window)
    window.ctx.set_dataset("breast_cancer")
    window.navigate("prediction")
    pred = window.page_map["prediction"]
    assert "No trained model" in pred.message.text.text()
    assert not pred.predict_btn.isEnabled()


def test_import_missing_dataset_from_training_page(qtbot, window, tmp_path, monkeypatch):
    """Missing dataset -> 'Import downloaded file…' validates and copies the file."""
    import app.ui.dataset_actions as actions
    from ml.datasets.registry import DatasetRegistry
    from tests.conftest import make_schema_fixture

    src = make_schema_fixture("stroke", tmp_path / "Downloads" / "healthcare-dataset-stroke-data.csv")

    class FakeDialog:
        @staticmethod
        def getOpenFileName(*_a, **_k):
            return str(src), ""

    class FakeBox:
        class Icon:
            Information = Warning = None

        def __init__(self, *_a):
            pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(actions, "QFileDialog", FakeDialog)
    monkeypatch.setattr(actions, "QMessageBox", FakeBox)
    window.ctx.settings.data_dir = str(tmp_path / "data")
    _login(qtbot, window)
    window.navigate("training")
    page = window.page_map["training"]
    page.dataset.setCurrentIndex(page.dataset.findData("stroke"))
    qtbot.waitUntil(lambda: page.ds_status.text() == "file missing", timeout=20000)
    assert page.ds_actions.isVisible()
    page._import_dataset()
    assert DatasetRegistry(tmp_path / "data").availability("stroke").available
    qtbot.waitUntil(lambda: page.ds_status.text().startswith("valid"), timeout=20000)
    assert page.start_btn.isEnabled()
