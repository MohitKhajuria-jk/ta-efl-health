"""Main window: login screen, then sidebar + page stack."""
from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QHBoxLayout, QMainWindow, QStackedWidget, QWidget

from app.services.context import AppContext
from app.ui.pages.about import AboutPage
from app.ui.pages.clients import ClientsPage
from app.ui.pages.dashboard import DashboardPage
from app.ui.pages.explainability import ExplainabilityPage
from app.ui.pages.login import LoginPage
from app.ui.pages.prediction import PredictionPage
from app.ui.pages.reports import ReportsPage
from app.ui.pages.results import ResultsPage
from app.ui.pages.settings import SettingsPage
from app.ui.pages.training import TrainingPage
from app.ui.pages.trust import TrustPage
from app.ui.widgets.common import confirm
from app.ui.widgets.sidebar import Sidebar


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("TA-EFL Health")
        self.resize(1360, 860)
        self.setMinimumSize(QSize(1100, 700))

        self.root = QStackedWidget()
        self.setCentralWidget(self.root)
        self.login = LoginPage(ctx)
        self.login.logged_in.connect(self._on_login)
        self.root.addWidget(self.login)

        shell = QWidget()
        sl = QHBoxLayout(shell)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.navigate.connect(self.navigate)
        self.sidebar.reference_toggled.connect(self.ctx.set_reference_mode)
        self.sidebar.logout_requested.connect(self._logout)
        sl.addWidget(self.sidebar)
        self.pages = QStackedWidget()
        self.pages.setObjectName("Pages")
        sl.addWidget(self.pages, 1)
        self.root.addWidget(shell)

        self.page_map = {
            "dashboard": DashboardPage(ctx),
            "clients": ClientsPage(ctx),
            "training": TrainingPage(ctx),
            "prediction": PredictionPage(ctx),
            "explain": ExplainabilityPage(ctx),
            "results": ResultsPage(ctx),
            "trust": TrustPage(ctx),
            "settings": SettingsPage(ctx),
            "reports": ReportsPage(ctx),
            "about": AboutPage(ctx),
        }
        for page in self.page_map.values():
            if hasattr(page, "set_navigator"):
                page.set_navigator(self.navigate)
            self.pages.addWidget(page)

        for ev in ("experiments_changed", "settings_changed", "reference_mode_changed",
                   "dataset_changed"):
            ctx.on(ev, lambda *_a: self._refresh_current())
        self.root.setCurrentIndex(0)

    # ------------------------------------------------------------------
    def current_key(self) -> str:
        w = self.pages.currentWidget()
        for k, p in self.page_map.items():
            if p is w:
                return k
        return "dashboard"

    def navigate(self, key: str) -> None:
        page = self.page_map.get(key)
        if page is None:
            return
        self.sidebar.select(key)
        if self.pages.currentWidget() is page:
            page.refresh()
        else:
            self.pages.setCurrentWidget(page)  # showEvent triggers refresh()

    def _refresh_current(self) -> None:
        w = self.pages.currentWidget()
        if w is not None and self.root.currentIndex() == 1:
            w.refresh()

    def _on_login(self, user: str) -> None:
        self.sidebar.set_user(user)
        self.sidebar.set_reference(self.ctx.reference_mode)
        self.root.setCurrentIndex(1)
        self.navigate("dashboard")

    def _logout(self) -> None:
        if self.ctx.training_running and not confirm(
                self, "Log out", "Training is still running in the background. Log out anyway?"):
            return
        self.ctx.current_user = None
        self.login.prepare()
        self.root.setCurrentIndex(0)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.ctx.training_running and not confirm(
                self, "Quit", "A training job is running. Stop it and quit?"):
            event.ignore()
            return
        for key in ("training", "results"):
            page = self.page_map[key]
            if hasattr(page, "shutdown"):
                page.shutdown()
        event.accept()
