"""Login / first-run account creation screen."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from app.services.auth import AuthError
from app.services.context import AppContext
from app.ui import icons, theme
from app.ui.widgets.common import Banner, button, label
from app.ui.widgets.logo import ShieldLogo

FEATURES = [("shield", "Privacy Preserving"), ("people", "Trust Aware"),
            ("brain", "Explainable AI"), ("sync", "Collaborative Learning")]


class BrandPanel(QFrame):
    """Navy gradient panel with logo, title, tagline and feature list."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(380)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.addStretch(1)
        logo = ShieldLogo(96)
        lay.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)
        t = QLabel("TA-EFL Health")
        t.setStyleSheet("color: white; font-size: 24pt; font-weight: 700; background: transparent;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Trust-Aware Federated Learning\nfor Personalized Healthcare\nRisk Prediction")
        sub.setStyleSheet("color: #DCE6FF; font-size: 12pt; background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addSpacing(8)
        lay.addWidget(t)
        lay.addWidget(sub)
        lay.addSpacing(26)
        for icon_name, text in FEATURES:
            row = QHBoxLayout()
            ic = QLabel()
            ic.setPixmap(icons.pixmap(icon_name, "#7FA6FF", 22))
            ic.setStyleSheet("background: transparent;")
            tl = QLabel(text)
            tl.setStyleSheet("color: white; font-size: 10.5pt; background: transparent;")
            row.addSpacing(30)
            row.addWidget(ic)
            row.addWidget(tl)
            row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(2)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        g = QLinearGradient(0, 0, float(self.width()), float(self.height()))
        g.setColorAt(0, QColor("#0E2A6B"))
        g.setColorAt(1, QColor(theme.NAVY))
        p.fillRect(self.rect(), QBrush(g))
        # decorative ECG trace across the upper third
        pen = QPen(QBrush(QColor(120, 160, 255, 70)), 2)
        p.setPen(pen)
        w, y = float(self.width()), self.height() * 0.22
        path = QPainterPath()
        path.moveTo(0, y)
        pts = [(0.30, 0), (0.36, -26), (0.41, 30), (0.46, -8), (0.50, 0), (0.70, 0),
               (0.74, -18), (0.78, 14), (0.81, 0), (1.0, 0)]
        for fx, dy in pts:
            path.lineTo(w * fx, y + dy)
        p.drawPath(path)
        p.end()
        super().paintEvent(event)


class LoginPage(QWidget):
    logged_in = pyqtSignal(str)

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(BrandPanel(), 5)

        right = QWidget()
        right.setObjectName("LoginPane")  # styled by selector so children keep their own styles
        right.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(70, 40, 70, 40)
        rl.addStretch(1)
        self.title = label("Welcome Back")
        self.title.setStyleSheet(f"color: {theme.TITLE}; font-size: 22pt; font-weight: 700;")
        self.subtitle = label("Login to access the system")
        self.subtitle.setStyleSheet("color: #374151; font-size: 11pt;")
        rl.addWidget(self.title)
        rl.addWidget(self.subtitle)
        rl.addSpacing(22)

        self.username = QLineEdit()
        self.username.setPlaceholderText("Username")
        self.username.addAction(icons.icon("user", "#64748B", 32), QLineEdit.ActionPosition.LeadingPosition)
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.addAction(icons.icon("lock", "#64748B", 32), QLineEdit.ActionPosition.LeadingPosition)
        eye = QAction(icons.icon("eye", "#64748B", 32), "Show password", self.password)
        eye.setCheckable(True)
        eye.toggled.connect(lambda on: self.password.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        self.password.addAction(eye, QLineEdit.ActionPosition.TrailingPosition)
        self.confirm = QLineEdit()
        self.confirm.setPlaceholderText("Confirm password")
        self.confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm.addAction(icons.icon("lock", "#64748B", 32), QLineEdit.ActionPosition.LeadingPosition)
        for e in (self.username, self.password, self.confirm):
            e.setMinimumHeight(40)
            rl.addWidget(e)
            rl.addSpacing(6)
        self.remember = QCheckBox("Remember Me")
        rl.addWidget(self.remember)
        rl.addSpacing(10)
        self.submit = button("Login", "Primary", on_click=self._submit)
        self.submit.setMinimumHeight(42)
        self.submit.setDefault(True)
        rl.addWidget(self.submit)
        self.error = Banner("error", "")
        rl.addWidget(self.error)
        self.forgot = button("Forgot Password?", "Link", on_click=self._forgot)
        rl.addWidget(self.forgot, 0, Qt.AlignmentFlag.AlignLeft)
        self.note = label("", "Muted", wrap=True)
        rl.addWidget(self.note)
        rl.addStretch(2)
        root.addWidget(right, 4)
        self.password.returnPressed.connect(self._submit)
        self.confirm.returnPressed.connect(self._submit)
        self.username.returnPressed.connect(self.password.setFocus)
        self.prepare()

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        self.first_run = not self.ctx.auth.has_users()
        self.password.clear()
        self.confirm.clear()
        self.error.set("error", "")
        remembered = self.ctx.settings.remembered_username
        self.username.setText(remembered)
        self.remember.setChecked(bool(remembered))
        if self.first_run:
            self.title.setText("Create Local Account")
            self.subtitle.setText("First launch: choose a username and password for this computer")
            self.submit.setText("Create Account")
            self.confirm.show()
            self.forgot.hide()
            self.note.setText("There are no built-in accounts. Passwords are stored only as salted "
                              "PBKDF2 hashes in the application's storage folder.")
        else:
            self.title.setText("Welcome Back")
            self.subtitle.setText("Login to access the system")
            self.submit.setText("Login")
            self.confirm.hide()
            self.forgot.show()
            self.note.setText("")
        (self.password if remembered else self.username).setFocus()

    def _submit(self) -> None:
        user = self.username.text().strip()
        pw = self.password.text()
        auth = self.ctx.auth
        try:
            if self.first_run:
                if pw != self.confirm.text():
                    raise AuthError("Passwords do not match")
                auth.create_user(user, pw)
            elif not auth.verify(user, pw):
                raise AuthError("Invalid username or password")
        except AuthError as exc:
            self.error.set("error", str(exc))
            self.password.clear()
            self.confirm.clear()
            return
        s = self.ctx.settings
        s.remembered_username = user if self.remember.isChecked() else ""
        try:
            self.ctx.settings_manager.save(s)
        except Exception:  # noqa: BLE001 - remembering the name is optional
            pass
        self.ctx.current_user = user
        self.password.clear()
        self.confirm.clear()
        self.error.set("error", "")
        self.logged_in.emit(user)

    def _forgot(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Reset local accounts")
        box.setText("Passwords cannot be recovered because only hashes are stored.")
        box.setInformativeText("Reset all local accounts on this computer? Experiments, models "
                               "and settings are kept. You will create a new account next.")
        box.setStandardButtons(QMessageBox.StandardButton.Reset | QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.standardButton(box.clickedButton()) == QMessageBox.StandardButton.Reset:
            self.ctx.auth.reset()
            self.prepare()
