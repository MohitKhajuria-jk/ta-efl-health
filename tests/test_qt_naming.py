"""Regression tests for the 'native Qt signal is not callable' failure.

Root cause (fixed): ``ExperimentWorker`` and the thread-pool signal holder declared a
signal named ``event``. ``QObject.event(QEvent)`` is the virtual method Qt calls to
deliver every event, so the signal shadowed it and Qt's event loop tried to *call* the
signal. These tests fail if a GUI class shadows a Qt method name again.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1] / "app" / "ui"

# QObject / QWidget / QThread members that must never be replaced by a signal or an
# instance attribute (static list so this test also runs without PyQt6 installed).
RESERVED = {
    "event", "eventFilter", "timerEvent", "childEvent", "customEvent", "connectNotify",
    "disconnectNotify", "thread", "moveToThread", "parent", "setParent", "children",
    "deleteLater", "destroyed", "objectName", "objectNameChanged", "signalsBlocked",
    "blockSignals", "sender", "property", "setProperty", "metaObject", "inherits",
    "show", "hide", "close", "update", "repaint", "resize", "move", "size", "width",
    "height", "font", "style", "layout", "palette", "rect", "pos", "geometry", "cursor",
    "isVisible", "isEnabled", "setEnabled", "setVisible", "windowTitle", "focusWidget",
    "started", "finished", "start", "quit", "wait", "exit", "run",
}
# methods the GUI intentionally overrides
INTENTIONAL_OVERRIDES = {"paintEvent", "showEvent", "closeEvent", "run", "start", "wait"}


def _ui_sources():
    return sorted(UI_DIR.rglob("*.py"))


def test_no_signal_shadows_a_qt_method_static():
    bad = []
    for path in _ui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                name = getattr(fn, "id", getattr(fn, "attr", ""))
                if name == "pyqtSignal":
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id in RESERVED:
                            bad.append(f"{path.name}:{node.lineno} signal '{t.id}'")
    assert not bad, f"Signals shadow Qt methods: {bad}"


def test_no_instance_attribute_shadows_a_qt_method_static():
    bad = []
    for path in _ui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self" and t.attr in RESERVED):
                        bad.append(f"{path.name}:{node.lineno} self.{t.attr}")
    assert not bad, f"Instance attributes shadow Qt methods: {bad}"


def test_live_qt_classes_do_not_shadow_base_members():
    """With PyQt6 installed: introspect every QObject subclass defined in app.ui."""
    QtCore = pytest.importorskip("PyQt6.QtCore")
    pytest.importorskip("PyQt6.QtWidgets")
    import app.ui as ui_pkg

    offenders = []
    for mod_info in pkgutil.walk_packages(ui_pkg.__path__, "app.ui."):
        mod = importlib.import_module(mod_info.name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__ or not issubclass(cls, QtCore.QObject):
                continue
            qt_bases = [b for b in cls.__mro__[1:] if b.__module__.startswith("PyQt6")]
            for name, value in vars(cls).items():
                if name.startswith("_") or name in INTENTIONAL_OVERRIDES:
                    continue
                for base in qt_bases:
                    if hasattr(base, name):
                        offenders.append(f"{cls.__name__}.{name} shadows {base.__name__}.{name}")
                        break
    assert not offenders, offenders
