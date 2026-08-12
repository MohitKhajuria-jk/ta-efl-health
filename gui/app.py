"""TA-EFL Health — PyQt6 desktop application.

Screens: Login, Dashboard, Clients, Federated Training, Trust Management,
Model Evaluation, Explainability (SHAP), Prediction, Reports.
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QListWidget, QStackedWidget,
    QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QProgressBar, QDialog, QFormLayout, QMessageBox, QFrame,
    QFileDialog, QHeaderView, QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.session import Session
from core.datasets import DATASETS, DEFAULT_DATASET, get_spec
from core.dataset import FEATURE_LABELS
from core.fl import HAS_FLWR
from core.model import evaluate
from gui import report_data as rd
from gui import theme
from gui.theme import T


def flabel(session, key: str) -> str:
    """Human-readable label for a feature of the session's dataset."""
    if session is not None and getattr(session, "spec", None) is not None:
        return session.spec.labels.get(key, FEATURE_LABELS.get(key, key))
    return FEATURE_LABELS.get(key, key)

#: Accent is theme-dependent; ACC() is read at paint time, not import time.
def ACC() -> str:
    return T()["accent"]


def DANGER() -> str:
    return T()["danger"]


ACCENT = "#1a56db"   # retained for backwards compatibility
PRESETS = {
    "No attack (all honest)": {},
    "1 label-flipping hospital": {2: "label_flip"},
    "Label flip + noise (2 attackers)": {2: "label_flip", 4: "noise"},
    "3 attackers incl. scaled poisoning": {1: "scale", 2: "label_flip", 4: "noise"},
}


# --------------------------------------------------------------------------
# Shared application state + training worker
# --------------------------------------------------------------------------
class TrainWorker(QThread):
    round_done = pyqtSignal(object)   # RoundRecord
    finished_all = pyqtSignal()

    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self._stop = False

    def run(self):
        for r in range(1, self.session.n_rounds + 1):
            if self._stop:
                break
            rec = self.session.server.run_round(r)
            self.round_done.emit(rec)
        self.finished_all.emit()

    def stop(self):
        self._stop = True


class AppState:
    def __init__(self):
        self.session: Session | None = None
        self.worker: TrainWorker | None = None
        # Last patient entered on the Predictions page, so the Explainability
        # page can explain that same person instead of a test-set patient.
        self.last_input: dict | None = None


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
CANVASES: list = []       # every (canvas, figure) the app creates


def metric_card(title: str) -> tuple[QFrame, QLabel]:
    card = QFrame()
    card.setObjectName("card")
    lay = QVBoxLayout(card)
    t = QLabel(title)
    t.setObjectName("muted")
    v = QLabel("—")
    v.setObjectName("metric")
    lay.addWidget(t)
    lay.addWidget(v)
    return card, v


def make_canvas(w=5, h=3.2) -> tuple[Canvas, Figure]:
    fig = Figure(figsize=(w, h), tight_layout=True)
    fig.patch.set_facecolor(T()["chart_bg"])
    canvas = Canvas(fig)
    CANVASES.append((canvas, fig))       # so a theme switch can restyle them
    return canvas, fig


# --------------------------------------------------------------------------
# Login dialog (cosmetic)
# --------------------------------------------------------------------------
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TA-EFL Health — Login")
        self.setFixedSize(360, 260)
        lay = QVBoxLayout(self)
        title = QLabel("TA-EFL Health")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Trust-Aware Explainable Federated Learning\n"
                     "for Healthcare Risk Prediction")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setObjectName("muted")
        form = QFormLayout()
        self.user = QLineEdit(); self.user.setPlaceholderText("Username")
        self.pw = QLineEdit(); self.pw.setPlaceholderText("Password")
        self.pw.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self.user)
        form.addRow(self.pw)
        btn = QPushButton("Login")
        btn.setObjectName("primary")
        btn.clicked.connect(self._login)
        lay.addWidget(title); lay.addWidget(sub)
        lay.addLayout(form); lay.addWidget(btn)

    def _login(self):
        if self.user.text().strip():
            self.accept()
        else:
            QMessageBox.warning(self, "Login", "Enter a username.")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
class DashboardPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Dashboard Overview"))
        cards = QHBoxLayout()
        self.c_clients = metric_card("Total Clients")
        self.c_rounds = metric_card("Global Rounds")
        self.c_acc = metric_card("Global Accuracy")
        self.c_auc = metric_card("Global AUC")
        for c, _ in (self.c_clients, self.c_rounds, self.c_acc, self.c_auc):
            cards.addWidget(c)
        lay.addLayout(cards)
        plots = QHBoxLayout()
        self.canvas_acc, self.fig_acc = make_canvas()
        self.canvas_trust, self.fig_trust = make_canvas()
        plots.addWidget(self.canvas_acc)
        plots.addWidget(self.canvas_trust)
        lay.addLayout(plots)

    def refresh(self):
        s = self.state.session
        if not s or not s.server.records:
            return
        recs = s.server.records
        m = recs[-1].metrics
        self.c_clients[1].setText(str(len(s.clients)))
        self.c_rounds[1].setText(str(len(recs)))
        self.c_acc[1].setText(f"{m['accuracy']*100:.2f}%")
        self.c_auc[1].setText(f"{m['auc']:.3f}")

        self.fig_acc.clear()
        ax = self.fig_acc.add_subplot(111)
        ax.plot([r.metrics["accuracy"] for r in recs], color=ACC(),
                marker="o", ms=3, label="Accuracy")
        ax.plot([r.metrics["auc"] for r in recs], color=DANGER(),
                marker="o", ms=3, label="AUC")
        ax.set_title("Global Model Performance"); ax.set_xlabel("Round")
        ax.legend(); ax.grid(alpha=0.3)
        theme.style_axes(self.fig_acc)
        self.canvas_acc.draw()

        self.fig_trust.clear()
        ax = self.fig_trust.add_subplot(111)
        if recs[-1].trust is not None:
            names = [f"H{c.cid+1}" for c in s.clients]
            colors = [DANGER() if c.attack != "none" else ACC()
                      for c in s.clients]
            ax.bar(names, recs[-1].trust, color=colors)
            ax.set_ylim(0, 1)
            ax.set_title("Trust Scores of Clients")
        else:
            ax.set_title("Trust Scores (trust strategy only)")
        ax.grid(alpha=0.3, axis="y")
        theme.style_axes(self.fig_trust)
        self.canvas_trust.draw()


class ClientsPage(QWidget):
    COLS = ["ID", "Hospital Name", "Samples", "Trust Score", "Behavior", "Status"]

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Clients (Hospitals)"))
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table)

    def refresh(self):
        s = self.state.session
        if not s:
            return
        trust = (s.server.records[-1].trust
                 if s.server.records and s.server.records[-1].trust is not None
                 else None)
        engine = s.server.trust_engine
        tau = engine.tau if engine is not None else 0.0
        self.table.setRowCount(len(s.clients))
        for i, c in enumerate(s.clients):
            t = trust[i] if trust is not None else None
            status = ("EXCLUDED" if t is not None and t < tau else "Active")
            vals = [f"H{c.cid+1}", c.name, str(c.n_samples),
                    f"{t:.3f}" if t is not None else "—",
                    c.attack if c.attack != "none" else "honest", status]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 5 and status == "EXCLUDED":
                    item.setForeground(Qt.GlobalColor.red)
                self.table.setItem(i, j, item)


class TrainingPage(QWidget):
    training_updated = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Federated Training"))

        form = QGridLayout()
        self.n_clients = QSpinBox(); self.n_clients.setRange(2, 10)
        self.n_clients.setValue(5)
        self.rounds = QSpinBox(); self.rounds.setRange(5, 200)
        self.rounds.setValue(30)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0.05, 10.0)
        self.alpha.setSingleStep(0.05); self.alpha.setValue(0.5)
        self.strategy = QComboBox()
        self.strategy.addItems(["trust", "fedavg", "median"])
        self.preset = QComboBox(); self.preset.addItems(PRESETS.keys())
        self.dataset = QComboBox()
        for key, spec in DATASETS.items():
            self.dataset.addItem(f"{spec.name} — {spec.task}", key)
        self.dataset.setCurrentIndex(
            list(DATASETS).index(DEFAULT_DATASET))
        self.backend = QComboBox()
        self.backend.addItem("Local Flower driver (stepped, live progress)", "local")
        self.backend.addItem(
            "Flower simulation runtime (Ray)" + ("" if HAS_FLWR else " — flwr not installed"),
            "flower")
        if not HAS_FLWR:                       # keep the option visible but inert
            try:
                self.backend.model().item(1).setEnabled(False)
            except Exception:
                pass
        form.addWidget(QLabel("Hospitals:"), 0, 0); form.addWidget(self.n_clients, 0, 1)
        form.addWidget(QLabel("Rounds:"), 0, 2); form.addWidget(self.rounds, 0, 3)
        form.addWidget(QLabel("Non-IID α:"), 1, 0); form.addWidget(self.alpha, 1, 1)
        form.addWidget(QLabel("Aggregation:"), 1, 2); form.addWidget(self.strategy, 1, 3)
        form.addWidget(QLabel("Attack scenario:"), 2, 0)
        form.addWidget(self.preset, 2, 1, 1, 3)
        form.addWidget(QLabel("Dataset:"), 3, 0)
        form.addWidget(self.dataset, 3, 1, 1, 3)
        form.addWidget(QLabel("Execution:"), 4, 0)
        form.addWidget(self.backend, 4, 1, 1, 3)
        lay.addLayout(form)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        btns.addWidget(self.start_btn); btns.addWidget(self.stop_btn)
        btns.addStretch()
        lay.addLayout(btns)

        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        row = QHBoxLayout()
        self.canvas, self.fig = make_canvas()
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumWidth(340)
        row.addWidget(self.canvas); row.addWidget(self.log)
        lay.addLayout(row)

    def start(self):
        attacks = dict(PRESETS[self.preset.currentText()])
        attacks = {k: v for k, v in attacks.items()
                   if k < self.n_clients.value()}
        self.log.clear()
        self.log.append("Building session (loading data, partitioning)...")
        QApplication.processEvents()
        self.state.session = Session(
            n_clients=self.n_clients.value(),
            dirichlet_alpha=self.alpha.value(),
            attacks=attacks,
            strategy=self.strategy.currentText(),
            n_rounds=self.rounds.value(),
            dataset=self.dataset.currentData(),
            backend=self.backend.currentData(),
        )
        s = self.state.session
        self.log.append(s.describe())
        self.log.append(f"Predicting: {s.spec.positive_label}")
        self.log.append(f"Engine: Flower ({s.server.backend} backend)")
        for c in s.clients:
            tag = f"  [{c.attack}]" if c.attack != "none" else ""
            self.log.append(f"H{c.cid+1} {c.name}: {c.n_samples} samples{tag}")
        self.progress.setMaximum(self.rounds.value())
        self.progress.setValue(0)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.state.worker = TrainWorker(s)
        self.state.worker.round_done.connect(self.on_round)
        self.state.worker.finished_all.connect(self.on_done)
        self.state.worker.start()

    def stop(self):
        if self.state.worker:
            self.state.worker.stop()

    def on_round(self, rec):
        self.progress.setValue(rec.rnd)
        m = rec.metrics
        self.log.append(f"Round {rec.rnd}: acc={m['accuracy']:.3f} "
                        f"auc={m['auc']:.3f}")
        recs = self.state.session.server.records
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot([r.metrics["accuracy"] for r in recs], color=ACC(),
                label="Accuracy")
        ax.plot([r.metrics["auc"] for r in recs], color=DANGER(), label="AUC")
        ax.set_xlabel("Round"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Training Progress")
        theme.style_axes(self.fig)
        self.canvas.draw()
        self.training_updated.emit()

    def on_done(self):
        self.log.append("Training finished.")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.training_updated.emit()


class TrustPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Trust Scores & History"))
        row = QHBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Client", "Trust", "Update Sim", "Explanation Sim", "Magnitude"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.canvas, self.fig = make_canvas()
        row.addWidget(self.table); row.addWidget(self.canvas)
        lay.addLayout(row)
        self.formula = QLabel(
            "T_k(t) = β·T_k(t−1) + (1−β)·[α·sim_update + (1−α)·sim_explanation]"
            "    |    weights ∝ n_k · T_k^γ  (dropped if T_k < τ)")
        self.formula.setObjectName("muted")
        lay.addWidget(self.formula)

    def refresh(self):
        s = self.state.session
        if not s or not s.server.records:
            return
        rec = s.server.records[-1]
        if rec.trust is None:
            return
        self.table.setRowCount(len(s.clients))
        for i, c in enumerate(s.clients):
            comp = rec.components[i] if rec.components else {}
            vals = [f"H{c.cid+1} {c.name}", f"{rec.trust[i]:.3f}",
                    f"{comp.get('update_sim', 0):.3f}",
                    f"{comp.get('explanation_sim', 0):.3f}",
                    f"{comp.get('magnitude_factor', 0):.3f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(v))
        hist = np.stack(s.server.trust_engine.history)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for k, c in enumerate(s.clients):
            style = "--" if c.attack != "none" else "-"
            ax.plot(hist[:, k], style, label=f"H{k+1}")
        ax.set_xlabel("Round"); ax.set_ylabel("Trust")
        ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.axhline(s.server.trust_engine.tau, color="red", lw=0.8, ls=":")
        ax.set_title("Trust Score History (dashed = attacker)")
        theme.style_axes(self.fig)
        self.canvas.draw()


class EvaluationPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Global Model Evaluation"))
        cards = QHBoxLayout()
        self.cards = {k: metric_card(k.upper())
                      for k in ("accuracy", "precision", "recall", "f1", "auc")}
        for c, _ in self.cards.values():
            cards.addWidget(c)
        lay.addLayout(cards)
        row = QHBoxLayout()
        self.canvas_cm, self.fig_cm = make_canvas(4, 3.4)
        self.canvas_roc, self.fig_roc = make_canvas(4, 3.4)
        row.addWidget(self.canvas_cm); row.addWidget(self.canvas_roc)
        lay.addLayout(row)

    def refresh(self):
        s = self.state.session
        if not s or not s.server.records:
            return
        m = evaluate(s.server.model, s.X_test, s.y_test)
        for k, (_, lbl) in self.cards.items():
            v = m[k]
            lbl.setText(f"{v*100:.2f}%" if k == "accuracy" else f"{v:.3f}")
        self.fig_cm.clear()
        ax = self.fig_cm.add_subplot(111)
        cm = m["confusion"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="#0f1b33")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        theme.style_axes(self.fig_cm)
        self.canvas_cm.draw()
        self.fig_roc.clear()
        ax = self.fig_roc.add_subplot(111)
        fpr, tpr = m["roc"]
        ax.plot(fpr, tpr, color=ACC(), label=f"AUC = {m['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", lw=0.8, color=T()["chart_grid"])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
        theme.style_axes(self.fig_roc)
        self.canvas_roc.draw()


def format_feature_value(session, key: str, raw_value: float) -> str:
    """Render a feature value in its clinical units / category name."""
    try:
        f = session.spec.feature(key)
    except Exception:
        return f"{raw_value:.2f}"
    if f.choices:
        nearest = min(f.choices, key=lambda c: abs(c - raw_value))
        return str(f.choices[nearest])
    if f.decimals == 0:
        return f"{raw_value:.0f}"
    return f"{raw_value:.{f.decimals}f}"


def strength_phrase(magnitude: float, strongest: float, direction: str) -> str:
    """Plain-English size of one feature's contribution to the prediction."""
    share = magnitude / strongest if strongest > 1e-12 else 0.0
    if share >= 0.66:
        adverb = "Strongly "
    elif share >= 0.33:
        adverb = ""
    else:
        adverb = "Slightly "
    verb = "increased" if direction == "up" else "reduced"
    return f"{adverb}{verb} the predicted risk".replace("  ", " ").capitalize()


class ExplainPage(QWidget):
    """Human-readable explanation on top of the existing SHAP values."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        outer.addWidget(_header("SHAP Explanation — Individual Prediction"))

        # Nothing is explained until the user asks for it.
        self._explained = False
        self._session_seen = None

        top = QHBoxLayout()
        top.addWidget(QLabel("Explain:"))
        self.source = QComboBox()
        self.source.addItem("Patient entered on the Predictions page", "form")
        self.source.addItem("Test patient (index below)", "test")
        self.source.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self.source)
        self.idx_label = QLabel("Test patient index:")
        top.addWidget(self.idx_label)
        self.idx = QSpinBox(); self.idx.setRange(0, 0)
        self.idx.valueChanged.connect(self._on_index_changed)
        btn = QPushButton("Explain Prediction")
        btn.setObjectName("primary")
        btn.clicked.connect(self.explain)
        top.addWidget(self.idx); top.addWidget(btn); top.addStretch()
        outer.addLayout(top)

        # Everything below the controls scrolls.
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 8, 0)

        # Shown until a prediction has actually been requested.
        self.placeholder = QFrame()
        self.placeholder.setObjectName("placeholder")
        ph_lay = QVBoxLayout(self.placeholder)
        ph_lay.setContentsMargins(20, 26, 20, 26)
        self.placeholder_text = QLabel("")
        self.placeholder_text.setWordWrap(True)
        self.placeholder_text.setObjectName("muted")
        ph_lay.addWidget(self.placeholder_text)
        body_lay.addWidget(self.placeholder)

        # The explanation itself, hidden until then.
        self.content = QWidget()
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(0, 0, 0, 0)
        self.content.setVisible(False)
        body_lay.addWidget(self.content)

        # --- 1. Patient information -------------------------------------
        self.patient = QLabel("—")
        self.patient.setObjectName("muted")
        self.patient.setWordWrap(True)
        lay.addWidget(self.patient)

        # --- 2. The prediction itself, before any explanation ------------
        self.result_card = QFrame()
        self.result_card.setObjectName("card")
        res_lay = QVBoxLayout(self.result_card)
        res_lay.setContentsMargins(20, 16, 20, 18)
        res_lay.setSpacing(4)
        self.risk = QLabel("—")
        self.risk.setFont(QFont("Arial", 30, QFont.Weight.Bold))
        self.risk.setStyleSheet("border:none;")
        res_lay.addWidget(self.risk)
        self.prob = QLabel("Train a model, then choose a patient.")
        self.prob.setFont(QFont("Arial", 13))
        self.prob.setWordWrap(True)
        self.prob.setObjectName("soft")
        res_lay.addWidget(self.prob)
        lay.addWidget(self.result_card)

        # --- 3. Explanation heading --------------------------------------
        why_head = QLabel("Why did the model make this prediction?")
        why_head.setObjectName("h2")
        lay.addWidget(why_head)
        note = QLabel("These are the model's own contributions to this "
                      "prediction, not statements of medical cause and effect.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        lay.addWidget(note)

        # --- 4. Human-readable factors -----------------------------------
        cols = QHBoxLayout()
        self.up_box, up_wrap = self._factor_column(
            "Factors increasing predicted risk", DANGER())
        self.down_box, down_wrap = self._factor_column(
            "Factors lowering predicted risk", ACC())
        cols.addWidget(up_wrap); cols.addWidget(down_wrap)
        lay.addLayout(cols)

        # --- 5. In simple terms ------------------------------------------
        simple = QFrame()
        simple.setObjectName("card")
        simple_lay = QVBoxLayout(simple)
        simple_head = QLabel("In simple terms")
        simple_head.setObjectName("muted")
        simple_lay.addWidget(simple_head)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("body")
        simple_lay.addWidget(self.summary)
        lay.addWidget(simple)

        # --- 6/7. Technical SHAP explanation + existing chart -------------
        tech = QLabel("Technical SHAP Explanation")
        tech.setObjectName("h3")
        lay.addWidget(tech)
        self.canvas, self.fig = make_canvas(6, 4)
        self.canvas.setMinimumHeight(360)
        lay.addWidget(self.canvas)

        disclaimer = QLabel("⚠ This is an AI-based risk prediction, not a "
                            "medical diagnosis.")
        disclaimer.setWordWrap(True)
        disclaimer.setObjectName("warn")
        lay.addWidget(disclaimer)
        body_lay.addStretch()

        self.scroll.setWidget(body)
        outer.addWidget(self.scroll)
        self.idx.setVisible(False)          # only shown for test patients
        self.idx_label.setVisible(False)
        self._show_placeholder()

    # ---- layout helpers --------------------------------------------------
    def _factor_column(self, heading: str, color: str):
        wrap = QFrame()
        wrap.setObjectName("panel")
        col = QVBoxLayout(wrap)
        head = QLabel(heading)
        head.setObjectName("dangerHead" if color == DANGER() else "accentHead")
        col.addWidget(head)
        box = QVBoxLayout()
        col.addLayout(box)
        col.addStretch()
        return box, wrap

    def _fill_column(self, box: QVBoxLayout, items: list[tuple[str, str, str]],
                     color: str):
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not items:
            empty = QLabel("None for this patient.")
            empty.setStyleSheet(
                f"color:{T()['disabled']};font-size:12px;border:none;")
            box.addWidget(empty)
            return
        for label, value, phrase in items:
            head = QLabel(f"● {label} — {value}")
            head.setWordWrap(True)
            head.setStyleSheet(f"color:{color};font-size:13px;font-weight:600;"
                               "border:none;")
            sub = QLabel(f"    {phrase}.")
            sub.setWordWrap(True)
            sub.setStyleSheet(f"color:{T()['text_soft']};font-size:12px;"
                              "border:none;padding-bottom:4px;")
            box.addWidget(head); box.addWidget(sub)

    # ---- gating ----------------------------------------------------------
    def _show_placeholder(self):
        s = self.state.session
        if not (s and s.server.records):
            msg = ("No trained model yet. Run <b>Federated Training</b> first, "
                   "then come back and click <b>Explain Prediction</b>.")
        elif self.source.currentData() == "form" and self._form_patient() is None:
            msg = ("No patient entered yet. Go to <b>Predictions</b>, fill in "
                   "the patient and press <b>Predict</b> — then come back here "
                   "to see why the model reached that result. Or switch the "
                   "selector above to a test patient.")
        else:
            msg = ("Choose a patient above and click <b>Explain Prediction</b> "
                   "to see the risk result and why the model reached it.")
        self.placeholder_text.setText(msg)
        self.content.setVisible(False)
        self.placeholder.setVisible(True)

    def _on_index_changed(self):
        """Changing the patient only re-renders once an explanation exists."""
        if self._explained:
            self.refresh()

    def _on_source_changed(self):
        is_test = self.source.currentData() == "test"
        self.idx.setVisible(is_test)
        self.idx_label.setVisible(is_test)
        if self._explained:
            self.refresh()

    def _form_patient(self):
        """The patient last submitted on the Predictions page, if still valid."""
        s = self.state.session
        got = self.state.last_input
        if not s or not got or got.get("dataset") != s.spec.key:
            return None
        return got

    def explain(self):
        """Explicit user request — this is what unlocks the explanation."""
        s = self.state.session
        if not s or not s.server.records:
            self._explained = False
            self._show_placeholder()
            return
        if self.source.currentData() == "form" and self._form_patient() is None:
            self._explained = False
            self._show_placeholder()
            return
        self._explained = True
        self.refresh()

    # ---- refresh ---------------------------------------------------------
    def refresh(self):
        s = self.state.session
        # A new training run invalidates any explanation of the old model.
        if s is not self._session_seen:
            self._session_seen = s
            self._explained = False
        if not self._explained or not s or not s.server.records:
            self._show_placeholder()
            return

        # Which patient are we explaining: the one entered on the Predictions
        # page, or one from the held-out test set?
        use_form = self.source.currentData() == "form"
        entered = self._form_patient() if use_form else None
        if use_form and entered is None:
            self._explained = False
            self._show_placeholder()
            return

        self.placeholder.setVisible(False)
        self.content.setVisible(True)
        self.idx.blockSignals(True)
        self.idx.setMaximum(max(0, len(s.X_test) - 1))
        self.idx.blockSignals(False)
        i = min(self.idx.value(), len(s.X_test) - 1)

        # Existing prediction + SHAP logic, unchanged.
        x = entered["x"] if entered is not None else s.X_test[i]
        p = float(s.server.model.predict_proba(x[None, :])[0])
        high = p >= 0.5
        phi = s.server.model.shap_values(x[None, :], s.bg_mean)[0]

        who = ("the patient you entered on the Predictions page"
               if entered is not None
               else f"test patient #{i} of {len(s.X_test) - 1}")
        self.patient.setText(f"{s.spec.name} — {s.spec.task}   |   {who}")
        colour = DANGER() if high else T()["success"]
        self.risk.setText("HIGH RISK" if high else "LOW RISK")
        self.risk.setStyleSheet(f"color:{colour};border:none;")
        self.prob.setText(f"{p * 100:.0f}% estimated probability of "
                          f"{s.spec.positive_label.lower()}")
        tint = ((T()["risk_high_bg"], T()["risk_high_border"]) if high
                else (T()["risk_low_bg"], T()["risk_low_border"]))
        self.result_card.setStyleSheet(
            "QFrame{background:%s;border:1px solid %s;border-radius:10px;}" % tint)

        # Plain-English factors, driven by the same SHAP values as the chart.
        raw = entered["raw"] if entered is not None else s.destandardize_row(x)
        strongest = float(np.max(np.abs(phi))) if len(phi) else 0.0
        order = np.argsort(-np.abs(phi))
        up, down = [], []
        for j in order:
            if abs(phi[j]) < 1e-9:
                continue
            key = s.feature_names[j]
            entry = (flabel(s, key), format_feature_value(s, key, raw[j]),
                     strength_phrase(abs(phi[j]), strongest,
                                     "up" if phi[j] > 0 else "down"))
            (up if phi[j] > 0 else down).append(entry)
        up, down = up[:5], down[:5]
        self._fill_column(self.up_box, up, DANGER())
        self._fill_column(self.down_box, down, ACC())

        bits = []
        if up:
            bits.append(f"{up[0][0]} ({up[0][1]}) pushed the prediction "
                        "upward")
        if down:
            bits.append(f"{down[0][0]} ({down[0][1]}) pulled it downward")
        joined = ", while ".join(bits) if bits else \
            "no single feature stood out"
        joined = joined[0].upper() + joined[1:]      # keep acronyms intact
        self.summary.setText(
            f"The model estimated a {p * 100:.0f}% probability of "
            f"{s.spec.positive_label.lower()}, which it reports as "
            f"{'high' if high else 'low'} risk. {joined}.")

        # Technical chart (unchanged data, clearer axis label).
        chart_order = np.argsort(np.abs(phi))
        labels = [flabel(s, s.feature_names[j]) for j in chart_order]
        vals = phi[chart_order]
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.barh(labels, vals,
                color=[DANGER() if v > 0 else ACC() for v in vals])
        ax.axvline(0, color=T()["chart_grid"], lw=0.8)
        ax.set_xlabel("Impact on prediction")
        ax.set_title("Feature contributions — entered patient"
                     if entered is not None else
                     f"Feature contributions — test patient #{i}")
        theme.style_axes(self.fig)
        self.canvas.draw()


class PredictPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        lay.addWidget(_header("Make Prediction"))
        row = QHBoxLayout()
        self.inputs: dict[str, QWidget] = {}   # QDoubleSpinBox or QComboBox
        self._form_dataset: str | None = None
        self.form_host = QWidget()
        self.form_host.setMaximumWidth(340)
        self.form = QFormLayout(self.form_host)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.form_host)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMaximumWidth(360)
        self.build_form(get_spec(DEFAULT_DATASET))
        row.addWidget(self.scroll)
        right = QVBoxLayout()
        self.hint = QLabel("Train a model, then enter patient values.")
        self.hint.setObjectName("muted")
        right.addWidget(self.hint)
        btn = QPushButton("Predict")
        btn.setObjectName("primary")
        btn.clicked.connect(self.predict)
        self.result = QLabel("—")
        self.result.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.canvas, self.fig = make_canvas(5, 3.4)
        right.addWidget(btn); right.addWidget(self.result)
        right.addWidget(self.canvas)
        row.addLayout(right)
        lay.addLayout(row)

    def build_form(self, spec):
        """Rebuild the patient input form for the active dataset."""
        while self.form.rowCount():
            self.form.removeRow(0)
        self.inputs = {}
        for f in spec.features:
            if f.choices:
                # Coded categorical (Sex, Exercise Induced Angina, ...):
                # show the human labels, keep the numeric code as item data.
                widget = QComboBox()
                for code in sorted(f.choices):
                    widget.addItem(str(f.choices[code]), int(code))
                pos = widget.findData(int(round(f.default)))
                widget.setCurrentIndex(pos if pos >= 0 else 0)
            else:
                widget = QDoubleSpinBox()
                widget.setDecimals(f.decimals)
                widget.setRange(float(f.lo), float(f.hi))
                widget.setSingleStep(10 ** -f.decimals if f.decimals else 1)
                widget.setValue(float(f.default))
            self.inputs[f.key] = widget
            self.form.addRow(f.label + ":", widget)
        self._form_dataset = spec.key

    def input_value(self, key: str) -> float:
        """Current value of one input, as the numeric code the model expects."""
        widget = self.inputs[key]
        if isinstance(widget, QComboBox):
            return float(widget.currentData())
        return float(widget.value())

    def refresh(self):
        s = self.state.session
        if s is None:
            return
        if self._form_dataset != s.spec.key:
            self.build_form(s.spec)
            self.result.setText("—")
        self.hint.setText(f"{s.spec.name} — predicting: {s.spec.positive_label}")

    def predict(self):
        s = self.state.session
        if not s or not s.server.records:
            QMessageBox.information(self, "Predict", "Train a model first.")
            return
        raw = np.array([self.input_value(f) for f in s.feature_names])
        x = s.standardize_row(raw)
        # Remember it so Explainability can explain this exact patient.
        self.state.last_input = {"raw": raw, "x": x, "dataset": s.spec.key}
        p = float(s.server.model.predict_proba(x[None, :])[0])
        high = p >= 0.5
        self.result.setText(f"{'HIGH RISK' if high else 'LOW RISK'}  "
                            f"(probability {p:.2f})")
        self.result.setStyleSheet(
            f"color:{DANGER() if high else T()['success']};")
        phi = s.server.model.shap_values(x[None, :], s.bg_mean)[0]
        order = np.argsort(np.abs(phi))[-6:]
        labels = [flabel(s, s.feature_names[j]) for j in order]
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.barh(labels, phi[order],
                color=[DANGER() if v > 0 else ACC() for v in phi[order]])
        ax.axvline(0, color=T()["chart_grid"], lw=0.8)
        ax.set_title("Top contributing features (SHAP)")
        theme.style_axes(self.fig)
        self.canvas.draw()


class ReportsPage(QWidget):
    """Tabular summary of everything the project has already computed."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        outer = QVBoxLayout(self)
        outer.addWidget(_header("Reports & Export"))
        btns = QHBoxLayout()
        b1 = QPushButton("Export metrics CSV"); b1.clicked.connect(self.export_csv)
        b2 = QPushButton("Export figures (PNG)"); b2.clicked.connect(self.export_figs)
        btns.addWidget(b1); btns.addWidget(b2); btns.addStretch()
        outer.addLayout(btns)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        outer.addWidget(self.status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 8, 0)

        self.t_perf = self._section(
            lay, "1. Model Performance",
            ["Dataset", "Strategy", "Accuracy", "Precision", "Recall", "F1",
             "ROC-AUC"])
        self.t_attack = self._section(
            lay, "2. Attack Robustness",
            ["Dataset", "Attack", "Trust-Aware AUC", "FedAvg AUC",
             "Median AUC", "Trust vs FedAvg"])
        self.t_trust = self._section(
            lay, "3. Trust Analysis",
            ["Client / Hospital", "Malicious?", "Performance Trust",
             "History Trust", "Final Trust", "Aggregation Weight"])
        self.t_exp = self._section(
            lay, "4. Experiment Results (E1–E5)",
            ["Experiment", "Configuration", "AUC / Margin", "Result"])
        self.t_cross = self._section(
            lay, "5. Cross-Dataset Results",
            ["Dataset", "Trust-Aware", "FedAvg", "Median", "Trust Advantage"])

        chart_head = QLabel("Final metrics — current session")
        chart_head.setObjectName("h3")
        lay.addWidget(chart_head)
        self.canvas, self.fig = make_canvas(6, 3.6)
        self.canvas.setMinimumHeight(300)
        lay.addWidget(self.canvas)
        lay.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ---- table helpers ---------------------------------------------------
    def _section(self, lay: QVBoxLayout, title: str,
                 cols: list[str]) -> QTableWidget:
        head = QLabel(title)
        head.setObjectName("h3")
        lay.addWidget(head)
        table = QTableWidget(0, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setMinimumHeight(120)
        lay.addWidget(table)
        return table

    def _fill(self, table: QTableWidget, rows: list[list],
              highlight: dict[int, set[int]] | None = None,
              empty_msg: str = "No data yet."):
        """Populate a table; ``highlight`` maps row index -> best columns."""
        highlight = highlight or {}
        if not rows:
            table.setRowCount(1)
            table.setSpan(0, 0, 1, table.columnCount())
            item = QTableWidgetItem(empty_msg)
            item.setForeground(Qt.GlobalColor.gray)
            table.setItem(0, 0, item)
            table.resizeRowsToContents()
            return
        table.clearSpans()
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                if c >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if c in highlight.get(r, set()):
                    font = item.font(); font.setBold(True); item.setFont(font)
                    item.setForeground(Qt.GlobalColor.darkGreen)
                table.setItem(r, c, item)
        table.resizeRowsToContents()

    @staticmethod
    def _num(value, digits: int = 3, signed: bool = False) -> str:
        if value is None:
            return "—"
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

    # ---- refresh ---------------------------------------------------------
    def refresh(self):
        rows = rd.load_study()
        self._fill_performance(rows)
        self._fill_attack(rows)
        self._fill_trust()
        self._fill_experiments(rows)
        self._fill_cross(rows)

        s = self.state.session
        bits = []
        bits.append(f"Session: {s.spec.name}, {s.server.strategy_name}, "
                    f"{len(s.server.records)} rounds"
                    if s and s.server.records else
                    "No trained model in this session yet.")
        bits.append(f"Study file: {len(rows)} rows from "
                    f"results/study_results.csv" if rows else
                    "No study results found — run  python run_study.py  to "
                    "populate sections 2, 4 and 5.")
        self.status.setText("   |   ".join(bits))
        self._draw_chart()

    def _fill_performance(self, rows):
        s = self.state.session
        out, hi = [], {}
        m = rd.session_metrics(s)
        if m is not None:
            out.append([
                s.spec.name,
                f"{rd.STRATEGY_LABELS.get(s.server.strategy_name, s.server.strategy_name)}"
                " (current session)",
                self._num(m["accuracy"]), self._num(m["precision"]),
                self._num(m["recall"]), self._num(m["f1"]),
                self._num(m["auc"])])
        # Study means: E5 records AUC per dataset and strategy (2/6 attackers).
        e5 = rd.mean_by(rd.experiment(rows, "E5"), "dataset", "config")
        for ds in rd.datasets_in(rd.experiment(rows, "E5")):
            best = max((e5.get((ds, st)) for st in rd.STRATEGY_ORDER
                        if e5.get((ds, st)) is not None), default=None)
            for st in rd.STRATEGY_ORDER:
                auc = e5.get((ds, st))
                if auc is None:
                    continue
                out.append([rd.dataset_title(ds),
                            f"{rd.STRATEGY_LABELS[st]} (study mean)",
                            "—", "—", "—", "—", self._num(auc)])
                if best is not None and abs(auc - best) < 1e-12:
                    hi[len(out) - 1] = {6}
        self._fill(self.t_perf, out, hi,
                   "Train a model, or run  python run_study.py  for study means.")

    def _fill_attack(self, rows):
        e1 = rd.experiment(rows, "E1")
        means = rd.mean_by(e1, "dataset", "attack_frac", "config")
        out, hi = [], {}
        seen = []
        for r in e1:
            key = (r["dataset"], r["attack_frac"])
            if key not in seen:
                seen.append(key)
        for ds, frac in seen:
            vals = {st: means.get((ds, frac, st)) for st in rd.STRATEGY_ORDER}
            delta = (None if vals["trust"] is None or vals["fedavg"] is None
                     else vals["trust"] - vals["fedavg"])
            label = f"{frac} malicious" + (" (none)" if frac.startswith("0/") else "")
            out.append([rd.dataset_title(ds), label,
                        self._num(vals["trust"]), self._num(vals["fedavg"]),
                        self._num(vals["median"]),
                        self._num(delta, signed=True)])
            present = {st: v for st, v in vals.items() if v is not None}
            if present:
                best = max(present.values())
                cols = {2 + rd.STRATEGY_ORDER.index(st)
                        for st, v in present.items() if abs(v - best) < 1e-12}
                hi[len(out) - 1] = cols
        self._fill(self.t_attack, out, hi,
                   "Run  python run_study.py  to populate attack robustness.")

    def _fill_trust(self):
        s = self.state.session
        trust_rows = rd.session_trust(s)
        tau = rd.trust_threshold(s)
        out, hi = [], {}
        for r in trust_rows:
            excluded = (tau is not None and r["final"] is not None
                        and r["final"] < tau)
            name = f"H{r['cid'] + 1} {r['name']}"
            malicious = (f"Yes ({r['attack']})" if r["malicious"] else "No")
            weight = ("—" if r["weight"] is None
                      else f"{r['weight'] * 100:.1f}%"
                      + (" (excluded)" if excluded else ""))
            out.append([name, malicious, self._num(r["performance"]),
                        self._num(r["history"]), self._num(r["final"]),
                        weight])
            if not r["malicious"] and r["final"] is not None:
                honest = [x["final"] for x in trust_rows
                          if not x["malicious"] and x["final"] is not None]
                if honest and abs(r["final"] - max(honest)) < 1e-12:
                    hi[len(out) - 1] = {4}
        msg = ("Train a model on the Federated Training page."
               if not trust_rows else
               "Trust scores are only produced by the Trust-Aware strategy.")
        self._fill(self.t_trust, out, hi, msg)

    def _fill_experiments(self, rows):
        out = []
        tau = rd.trust_threshold(self.state.session) or rd.default_tau()

        # E1 — strategy comparison per attack fraction.
        e1 = rd.mean_by(rd.experiment(rows, "E1"), "attack_frac", "config")
        fracs = []
        for r in rd.experiment(rows, "E1"):
            if r["attack_frac"] not in fracs:
                fracs.append(r["attack_frac"])
        for frac in fracs:
            t, f_, m = (e1.get((frac, st)) for st in rd.STRATEGY_ORDER)
            if t is None:
                continue
            delta = None if f_ is None else t - f_
            verdict = ("—" if delta is None else
                       f"Trust-Aware {'ahead' if delta > 0 else 'behind'} "
                       f"FedAvg by {abs(delta):.3f}")
            out.append(["E1 Robustness", f"{frac} malicious",
                        f"AUC {t:.3f}", verdict])

        # E2 — trust-signal ablation (value = AUC, extra = detection margin).
        e2_rows = rd.experiment(rows, "E2")
        e2_auc = rd.mean_by(e2_rows, "config")
        e2_margin = rd.mean_by(e2_rows, "config", field="extra")
        best_margin = max(e2_margin.values(), default=None)
        for cfg, auc in e2_auc.items():
            margin = e2_margin.get(cfg)
            verdict = ("—" if margin is None else
                       ("Best separation" if best_margin is not None
                        and abs(margin - best_margin) < 1e-12 else
                        "Attackers not separated" if margin <= 0 else
                        "Attackers separated"))
            out.append(["E2 Ablation", cfg[0],
                        f"AUC {auc:.3f} / margin {self._num(margin)}", verdict])

        # E3 — kappa sweep (value = detection margin).
        for cfg, margin in rd.mean_by(rd.experiment(rows, "E3"), "config").items():
            out.append(["E3 Extrapolation κ", cfg[0],
                        f"margin {margin:+.3f}",
                        "Label flipper detected" if margin > 0
                        else "Label flipper missed"])

        # E4 — heterogeneity false positives (value = min honest trust).
        for cfg, min_trust in rd.mean_by(rd.experiment(rows, "E4"), "config").items():
            out.append(["E4 Non-IID", cfg[0], f"min honest trust {min_trust:.3f}",
                        f"No honest client excluded (τ={tau:.2f})"
                        if min_trust >= tau else
                        f"Honest client below τ={tau:.2f}"])

        # E5 — cross-dataset.
        e5 = rd.mean_by(rd.experiment(rows, "E5"), "dataset", "config")
        for ds in rd.datasets_in(rd.experiment(rows, "E5")):
            t, f_ = e5.get((ds, "trust")), e5.get((ds, "fedavg"))
            if t is None:
                continue
            delta = None if f_ is None else t - f_
            out.append(["E5 Cross-dataset", rd.dataset_title(ds),
                        f"AUC {t:.3f}",
                        "—" if delta is None else
                        f"{delta:+.3f} vs FedAvg"])

        self._fill(self.t_exp, out, {},
                   "Run  python run_study.py  to populate E1–E5.")

    def _fill_cross(self, rows):
        e5 = rd.experiment(rows, "E5")
        means = rd.mean_by(e5, "dataset", "config")
        out, hi = [], {}
        for ds in rd.datasets_in(e5):
            vals = {st: means.get((ds, st)) for st in rd.STRATEGY_ORDER}
            delta = (None if vals["trust"] is None or vals["fedavg"] is None
                     else vals["trust"] - vals["fedavg"])
            out.append([rd.dataset_title(ds), self._num(vals["trust"]),
                        self._num(vals["fedavg"]), self._num(vals["median"]),
                        self._num(delta, signed=True)])
            present = {st: v for st, v in vals.items() if v is not None}
            if present:
                best = max(present.values())
                hi[len(out) - 1] = {1 + rd.STRATEGY_ORDER.index(st)
                                    for st, v in present.items()
                                    if abs(v - best) < 1e-12}
        self._fill(self.t_cross, out, hi,
                   "Run  python run_study.py  to populate cross-dataset results.")

    def _draw_chart(self):
        s = self.state.session
        m = rd.session_metrics(s)
        if m is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        keys = ["accuracy", "precision", "recall", "f1", "auc"]
        ax.bar([k.upper() for k in keys], [m[k] for k in keys], color=ACC())
        ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"Final metrics — {s.spec.name}, strategy: "
                     f"{s.server.strategy_name}")
        theme.style_axes(self.fig)
        self.canvas.draw()

    def export_csv(self):
        s = self.state.session
        if not s or not s.server.records:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "metrics.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["round", "accuracy", "auc", "f1"]
            header += [f"trust_H{c.cid+1}" for c in s.clients]
            w.writerow(header)
            for r in s.server.records:
                row = [r.rnd, f"{r.metrics['accuracy']:.4f}",
                       f"{r.metrics['auc']:.4f}", f"{r.metrics['f1']:.4f}"]
                row += ([f"{t:.4f}" for t in r.trust]
                        if r.trust is not None else [""] * len(s.clients))
                w.writerow(row)
        QMessageBox.information(self, "Export", f"Saved {path}")

    def export_figs(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder")
        if not folder:
            return
        self.fig.savefig(os.path.join(folder, "final_metrics.png"), dpi=150)
        QMessageBox.information(self, "Export", f"Figures saved to {folder}")


def _header(text: str) -> QLabel:
    h = QLabel(text)
    h.setObjectName("h1")
    return h


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    # Predictions comes before Explainability: the user sees the risk result
    # first, and only then the explanation of it.
    PAGES = ["Dashboard", "Clients (Hospitals)", "Federated Training",
             "Trust Management", "Model Evaluation", "Predictions",
             "Explainability (SHAP)", "Reports"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TA-EFL Health — Trust-Aware Explainable "
                            "Federated Learning")
        self.resize(1180, 720)
        self.state = AppState()

        root = QWidget(); self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        # Left column: navigation plus the theme switch underneath it.
        side = QWidget()
        side.setMaximumWidth(210)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(0)
        self.nav = QListWidget()
        self.nav.addItems(self.PAGES)
        self.nav.setObjectName("nav")
        side_lay.addWidget(self.nav)
        self.theme_btn = QPushButton()
        self.theme_btn.setToolTip("Switch between light and dark appearance")
        self.theme_btn.clicked.connect(self._toggle_theme)
        side_lay.addWidget(self.theme_btn)
        self.stack = QStackedWidget()

        self.p_dash = DashboardPage(self.state)
        self.p_clients = ClientsPage(self.state)
        self.p_train = TrainingPage(self.state)
        self.p_trust = TrustPage(self.state)
        self.p_eval = EvaluationPage(self.state)
        self.p_explain = ExplainPage(self.state)
        self.p_predict = PredictPage(self.state)
        self.p_reports = ReportsPage(self.state)
        for p in (self.p_dash, self.p_clients, self.p_train, self.p_trust,
                  self.p_eval, self.p_predict, self.p_explain, self.p_reports):
            self.stack.addWidget(p)   # must stay in sync with PAGES

        self.nav.currentRowChanged.connect(self._switch)
        self.p_train.training_updated.connect(self._refresh_all)
        lay.addWidget(side); lay.addWidget(self.stack)
        self._sync_theme_button()
        self.nav.setCurrentRow(2)   # start on Training page

    def _switch(self, row: int):
        self.stack.setCurrentIndex(row)
        self._refresh_all()

    def _refresh_all(self):
        for p in (self.p_dash, self.p_clients, self.p_trust, self.p_eval,
                  self.p_explain, self.p_predict, self.p_reports):
            p.refresh()

    # ---- theme -----------------------------------------------------------
    def _sync_theme_button(self):
        dark = theme.name() == "dark"
        self.theme_btn.setText("☀  Light mode" if dark else "🌙  Dark mode")

    def _toggle_theme(self):
        apply_theme(QApplication.instance(), theme.toggle())
        self._sync_theme_button()
        for canvas, fig in CANVASES:        # re-theme charts already drawn
            theme.restyle(fig)
            canvas.draw_idle()
        self._refresh_all()


def apply_theme(app: QApplication, name: str) -> None:
    """Activate a theme and restyle every widget currently on screen."""
    theme.set_theme(name)
    app.setStyleSheet(theme.stylesheet())
    QSettings("TA-EFL", "Health").setValue("theme", name)


def main():
    app = QApplication(sys.argv)
    saved = QSettings("TA-EFL", "Health").value("theme", "light")
    apply_theme(app, str(saved))
    login = LoginDialog()
    if login.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

