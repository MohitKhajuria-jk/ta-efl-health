"""Visual regression tests: buttons and text must actually be visible on screen.

Each check renders a widget to an image (``QWidget.grab``) and measures luminance
contrast between its pixels. They run with a *dark* system palette installed first, to
reproduce Windows dark mode, and then the application's theme exactly as ``main.py``
applies it. Skipped when PyQt6/pytest-qt are not installed.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")
os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtGui import QColor, QPalette  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app.configuration.settings import AppSettings, SettingsManager  # noqa: E402
from app.services.context import AppContext  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


def _lum(c: QColor) -> float:
    return (0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()) / 255.0


def contrast(widget, step: int = 1) -> float:
    """How far the most extreme pixels (text, icons) stand out from the background.

    Background = median pixel luminance. Text often covers < 1 % of a widget, so the
    extremes are the mean of the 0.3 % brightest / darkest pixels (averaging a larger
    share dilutes short labels such as "Login" with background pixels).
    """
    img = widget.grab().toImage()
    vals = sorted(_lum(img.pixelColor(x, y))
                  for x in range(0, img.width(), step) for y in range(0, img.height(), step))
    assert vals, "widget rendered empty"
    n = len(vals)
    median = vals[n // 2]
    k = max(1, int(n * 0.003))
    hi, lo = sum(vals[-k:]) / k, sum(vals[:k]) / k
    return max(hi - median, median - lo)


def _dark_system_palette() -> QPalette:
    pal = QPalette()
    for role, c in ((QPalette.ColorRole.Window, "#202020"), (QPalette.ColorRole.Base, "#1E1E1E"),
                    (QPalette.ColorRole.Text, "#FFFFFF"), (QPalette.ColorRole.WindowText, "#FFFFFF"),
                    (QPalette.ColorRole.Button, "#2D2D2D"), (QPalette.ColorRole.ButtonText, "#FFFFFF")):
        pal.setColor(role, QColor(c))
    return pal


@pytest.fixture
def themed_window(qtbot, tmp_path):
    app = QApplication.instance()
    app.setPalette(_dark_system_palette())  # simulate Windows dark mode
    theme.apply_theme(app)
    (tmp_path / "data").mkdir()
    m = SettingsManager(tmp_path / "settings.json")
    m.save(AppSettings(storage_dir=str(tmp_path / "storage"), data_dir=str(tmp_path / "data"),
                       default_dataset="breast_cancer", malicious_simulation_enabled=False))
    w = MainWindow(AppContext(m))
    qtbot.addWidget(w)
    w.resize(1280, 800)
    w.show()
    qtbot.waitExposed(w)
    return w


def test_contrast_metric_is_calibrated(qtbot, themed_window):
    """The metric must reject invisible text and accept normal text, or the tests are void."""
    from PyQt6.QtWidgets import QLabel, QPushButton
    for css, visible in (("color: white; background: white;", False),
                         ("color: #1D4ED8; background: #1D4ED8;", False),
                         ("color: #1F2937; background: white;", True),
                         ("color: white; background: #1D4ED8;", True)):
        for widget in (QLabel("Settings"), QPushButton("Login")):
            widget.setStyleSheet(css + " border: none; padding: 8px;")
            widget.resize(560, 42)
            qtbot.addWidget(widget)
            widget.show()
            qtbot.waitExposed(widget)
            c = contrast(widget)
            assert (c > 0.4) == visible, f"{css} {type(widget).__name__}: contrast {c:.2f}"


def test_palette_is_light_even_in_dark_mode(themed_window):
    pal = QApplication.instance().palette()
    assert _lum(pal.color(QPalette.ColorRole.Window)) > 0.8
    assert _lum(pal.color(QPalette.ColorRole.Text)) < 0.3


def test_login_button_and_fields_visible(qtbot, themed_window):
    login = themed_window.login
    btn = login.submit
    assert btn.isVisible() and btn.width() > 100 and btn.height() >= 36
    img = btn.grab().toImage()
    bg = img.pixelColor(6, img.height() // 2)  # inside the button, left of the text
    assert bg.blue() > bg.red() + 60, f"Login button is not blue (got {bg.name()})"
    assert contrast(btn) > 0.4, "Login button text is not readable"
    for w in (login.title, login.subtitle):
        assert w.isVisible() and contrast(w) > 0.4, f"{w.text()} not readable"
    # empty inputs show light-grey placeholder text and an icon by design: lower bar
    for w in (login.username, login.password):
        assert w.isVisible() and contrast(w) > 0.25, f"{w.placeholderText()} field not visible"


def _login(qtbot, w):
    w.login.username.setText("tester")
    w.login.password.setText("password-123")
    w.login.confirm.setText("password-123")
    w.login.submit.click()
    assert w.root.currentIndex() == 1


def test_every_page_title_and_nav_button_readable(qtbot, themed_window):
    w = themed_window
    _login(qtbot, w)
    for key, page in w.page_map.items():
        w.navigate(key)
        qtbot.wait(30)
        assert contrast(page.header.title) > 0.4, f"{key}: title not readable"
        assert contrast(page.header.subtitle) > 0.4, f"{key}: subtitle not readable"
        assert contrast(w.sidebar.buttons[key]) > 0.4, f"{key}: sidebar item not readable"


def test_tables_tabs_and_dialogs_readable(qtbot, themed_window):
    w = themed_window
    _login(qtbot, w)
    w.navigate("dashboard")
    qtbot.wait(30)
    dash = w.page_map["dashboard"]
    assert contrast(dash.datasets.horizontalHeader()) > 0.4
    assert contrast(dash.datasets.viewport()) > 0.4
    w.navigate("results")
    qtbot.wait(30)
    assert contrast(w.page_map["results"].tabs.tabBar()) > 0.4
    box = QMessageBox(w)
    box.setText("A readable message")
    box.show()
    qtbot.waitExposed(box)
    assert contrast(box) > 0.4
    box.close()
