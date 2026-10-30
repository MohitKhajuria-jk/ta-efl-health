"""Dark navy navigation sidebar."""
from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import icons
from app.ui.widgets.logo import ShieldLogo

NAV_ITEMS = [
    ("dashboard", "Dashboard", "dashboard"),
    ("clients", "Hospital Clients", "clients"),
    ("training", "Federated Training", "training"),
    ("prediction", "Risk Prediction", "prediction"),
    ("explain", "Explainability (SHAP)", "explain"),
    ("results", "Results & Analysis", "results"),
    ("trust", "Trust Analysis", "trust"),
    ("settings", "Settings", "settings"),
    ("reports", "Reports", "reports"),
    ("about", "About", "about"),
]


class Sidebar(QFrame):
    navigate = pyqtSignal(str)
    reference_toggled = pyqtSignal(bool)
    logout_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 12, 10, 12)
        lay.setSpacing(4)

        brand = QHBoxLayout()
        brand.addWidget(ShieldLogo(28))
        title = QLabel("TA-EFL Health")
        title.setObjectName("SidebarBrand")
        brand.addWidget(title)
        brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(10)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key, text, icon_name in NAV_ITEMS:
            b = QPushButton(text)
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.setIcon(icons.icon(icon_name, "#E5ECFF", 32))
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=key: self.navigate.emit(k))
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
        lay.addStretch(1)

        footer = QWidget()
        footer.setObjectName("SidebarFooter")
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(4, 4, 4, 0)
        fl.setSpacing(6)
        self.reference_box = QCheckBox("Show design reference values")
        self.reference_box.setToolTip(
            "Displays the numbers transcribed from the design image, clearly labelled.\n"
            "They are not experimental results.")
        self.reference_box.toggled.connect(self.reference_toggled.emit)
        self.user_label = QLabel("")
        self.user_label.setStyleSheet("color: #93A4CF; font-size: 9pt;")
        logout = QPushButton("Log out")
        logout.setIcon(icons.icon("logout", "#DCE5FF", 32))
        logout.clicked.connect(lambda _c=False: self.logout_requested.emit())
        fl.addWidget(self.reference_box)
        fl.addWidget(self.user_label)
        fl.addWidget(logout)
        lay.addWidget(footer)

    def select(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def set_user(self, name: str) -> None:
        self.user_label.setText(f"Signed in as {name}")

    def set_reference(self, on: bool) -> None:
        self.reference_box.blockSignals(True)
        self.reference_box.setChecked(on)
        self.reference_box.blockSignals(False)
