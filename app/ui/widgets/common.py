"""Reusable widgets: cards, metric cards, banners, badges, tables, headers."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.reference_data import REFERENCE_BANNER
from app.ui import icons, theme


def label(text: str = "", obj: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if obj:
        lab.setObjectName(obj)
    lab.setWordWrap(wrap)
    return lab


def button(text: str, kind: str | None = None, icon_name: str | None = None,
           on_click: Callable[[], None] | None = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if icon_name:
        color = "#FFFFFF" if kind in ("Primary", "Success") else theme.BLUE
        b.setIcon(icons.icon(icon_name, color, 32))
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if on_click:
        b.clicked.connect(lambda _checked=False: on_click())
    return b


class Card(QFrame):
    """White rounded panel with an optional blue title."""

    def __init__(self, title: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 14)
        self.body.setSpacing(10)
        self.header = QHBoxLayout()
        self.title_label = label(title or "", "CardTitle")
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        if title is not None:
            self.body.addLayout(self.header)

    def set_title(self, text: str) -> None:
        self.title_label.setText(text)

    def add(self, w: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(w, stretch)
        return w


class MetricCard(QFrame):
    """Dashboard KPI tile: icon, large value, caption (tinted background)."""

    def __init__(self, icon_name: str, caption: str, tint: str = "blue") -> None:
        super().__init__()
        self.setObjectName("MetricCard")
        bg, fg = theme.TINTS[tint]
        self.setStyleSheet(f"#MetricCard {{ background: {bg}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(2)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, fg, 34))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value = label("—", "MetricValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value.setStyleSheet(f"color: {fg if tint == 'red' else '#0F172A'};")
        self.caption = label(caption, "MetricCaption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ic)
        lay.addWidget(self.value)
        lay.addWidget(self.caption)
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, value: str, caption: str | None = None, tooltip: str = "") -> None:
        self.value.setText(value)
        if caption is not None:
            self.caption.setText(caption)
        self.setToolTip(tooltip)


class StatCard(QFrame):
    """Horizontal stat tile used on the clients page (icon left, value + caption right)."""

    def __init__(self, icon_name: str, caption: str, color: str = theme.BLUE) -> None:
        super().__init__()
        self.setObjectName("Card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        ic = QLabel()
        ic.setPixmap(icons.pixmap(icon_name, color, 30))
        col = QVBoxLayout()
        col.setSpacing(0)
        self.value = label("—", "MetricValue")
        self.value.setStyleSheet("font-size: 16pt;")
        self.caption = label(caption, "Muted")
        col.addWidget(self.value)
        col.addWidget(self.caption)
        lay.addWidget(ic)
        lay.addLayout(col)
        lay.addStretch(1)

    def set(self, value: str) -> None:
        self.value.setText(value)


class Banner(QFrame):
    KINDS = {"reference": "BannerReference", "info": "BannerInfo", "warning": "BannerWarning",
             "error": "BannerError", "success": "BannerSuccess"}

    def __init__(self, kind: str = "info", text: str = "") -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.icon = QLabel()
        self.text = label(text, wrap=True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.text, 1)
        self.set(kind, text)
        if not text:
            self.hide()

    def set(self, kind: str, text: str) -> None:
        self.setObjectName(self.KINDS[kind])
        color = {"reference": "#B45309", "info": theme.BLUE, "warning": "#B45309",
                 "error": theme.RED, "success": theme.GREEN}[kind]
        name = {"reference": "warning", "info": "about", "warning": "warning",
                "error": "warning", "success": "check"}[kind]
        self.icon.setPixmap(icons.pixmap(name, color, 18))
        self.text.setText(text)
        self.style().unpolish(self)
        self.style().polish(self)
        self.setVisible(bool(text))


class ReferenceBanner(Banner):
    def __init__(self) -> None:
        super().__init__("reference", REFERENCE_BANNER)
        self.hide()


def badge(text: str, kind: str) -> QLabel:
    colors = {"ok": ("#DCFCE7", "#15803D"), "bad": ("#FEE2E2", "#B91C1C"),
              "warn": ("#FEF3C7", "#92400E"), "info": ("#DBEAFE", "#1E40AF"),
              "neutral": ("#F1F5F9", "#334155")}
    bg, fg = colors[kind]
    lab = QLabel(text)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab.setStyleSheet(f"background: {bg}; color: {fg}; border-radius: 4px; padding: 2px 8px;"
                      f"font-weight: 600; font-size: 9pt;")
    return lab


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.title = label(title, "PageTitle")
        self.subtitle = label(subtitle, "PageSubtitle", wrap=True)  # never clipped
        col.addWidget(self.title)
        col.addWidget(self.subtitle)
        lay.addLayout(col, 1)
        self.right = QHBoxLayout()
        self.right.setSpacing(8)
        lay.addLayout(self.right)

    def add_right(self, w: QWidget) -> QWidget:
        self.right.addWidget(w)
        return w


class Page(QWidget):
    """Base page: scrollable column with header, reference banner and message banner."""

    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner.setObjectName("Workspace")
        self.root = QVBoxLayout(inner)
        self.root.setContentsMargins(24, 18, 24, 18)
        self.root.setSpacing(12)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.header = PageHeader(title, subtitle)
        self.root.addWidget(self.header)
        self.reference_banner = ReferenceBanner()
        self.root.addWidget(self.reference_banner)
        self.message = Banner()
        self.root.addWidget(self.message)

    def show_message(self, kind: str, text: str) -> None:
        self.message.set(kind, text)

    def clear_message(self) -> None:
        self.message.set("info", "")

    def refresh(self) -> None:  # pragma: no cover - overridden
        pass

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self.refresh()


def make_table(headers: Sequence[str], stretch_last: bool = False) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(list(headers))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    if stretch_last:
        hh.setStretchLastSection(True)
    t.verticalHeader().setDefaultSectionSize(34)
    return t


def table_item(value: object, align_right: bool = False, bold: bool = False,
               color: str | None = None, data: object = None) -> QTableWidgetItem:
    it = QTableWidgetItem("—" if value is None else str(value))
    if align_right:
        it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if bold:
        f = QFont()
        f.setBold(True)
        it.setFont(f)
    if color:
        it.setForeground(QBrush(QColor(color)))
    if data is not None:
        it.setData(Qt.ItemDataRole.UserRole, data)
    return it


def fill_table(t: QTableWidget, rows: Iterable[Sequence[object]],
               row_tint: Callable[[int, Sequence[object]], str | None] | None = None) -> None:
    rows = list(rows)
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        tint = row_tint(r, row) if row_tint else None
        for c, v in enumerate(row):
            if isinstance(v, QWidget):
                t.setCellWidget(r, c, v)
                continue
            it = v if isinstance(v, QTableWidgetItem) else table_item(v)
            if tint:
                it.setBackground(QBrush(QColor(tint)))
            t.setItem(r, c, it)


def fmt(v: object, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def hbox(*widgets: QWidget, stretch: Sequence[int] | None = None, spacing: int = 12) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for i, wid in enumerate(widgets):
        lay.addWidget(wid, stretch[i] if stretch else 0)
    return w


def experiment_label(rec: dict) -> str:
    tm = rec.get("test_metrics") or {}
    auc = tm.get("roc_auc")
    auc_s = f", test AUC {auc:.3f}" if isinstance(auc, float) else ""
    return (f"{rec.get('created', '')[:16].replace('T', ' ')} · {rec.get('dataset', {}).get('name', rec['config']['dataset_key'])}"
            f" · {rec.get('strategy_label', rec['config']['strategy'])} · {rec.get('status')}{auc_s}")


class ExperimentPicker(QComboBox):
    """Combo box listing stored experiments (newest first)."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(300)
        self.setMaximumWidth(520)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(34)

    def populate(self, records: list[dict], selected: str | None) -> None:
        self.blockSignals(True)
        self.clear()
        for rec in records:
            self.addItem(experiment_label(rec), rec["id"])
        idx = self.findData(selected) if selected else 0
        self.setCurrentIndex(max(idx, 0) if records else -1)
        self.blockSignals(False)

    def current_id(self) -> str | None:
        return self.currentData()


def confirm(parent: QWidget, title: str, text: str) -> bool:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    box.exec()
    return box.standardButton(box.clickedButton()) == QMessageBox.StandardButton.Yes


def show_error(parent: QWidget, title: str, message: str, details: str = "") -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if details:
        box.setDetailedText(details)
    box.exec()
