"""About: project description, technology stack, team and scientific limitations."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.services import reference_data as ref
from app.services.context import AppContext
from app.ui import icons, theme
from app.ui.widgets.common import Card, Page, fill_table, hbox, label, make_table
from app.ui.widgets.logo import ShieldLogo

STACK = [("Python", "#3776AB"), ("Flower", "#F2B705"), ("scikit-learn", "#F7931E"),
         ("SHAP", "#E11D48"), ("PyQt6", "#41CD52"), ("Matplotlib", "#11557C")]

OBJECTIVE = (
    "Research objective: study whether weighting hospital updates by a measured trust score "
    "makes federated risk models more robust to faulty or malicious participants than "
    "standard FedAvg and coordinate-wise median aggregation, and explain individual "
    "predictions of the resulting model with SHAP.")

ARCHITECTURE = [
    "Data layer – five dataset schemas with validation, documented missing-value codes and "
    "stratified, seeded train / server-validation / test splits.",
    "Federated layer – an in-process Flower simulation: Flower's Server, ClientManager and "
    "strategies coordinate simulated hospital clients (NumPyClient); messages are encoded with "
    "Flower's protobuf format through Flower's ClientProxy interface.",
    "Preprocessing – fitted only from client training rows via aggregate statistics "
    "(pipeline v2); validation and test rows are only transformed.",
    "Model – L2-regularised logistic regression trained by mini-batch SGD on each client.",
    "Aggregation – Trust-Aware (TA-EFL), FedAvg and coordinate-wise median; malformed updates "
    "are rejected by all three.",
    "Explainability – shap.LinearExplainer in log-odds space with an additivity check.",
    "Desktop app – PyQt6; long jobs run in background threads; results stored as JSON "
    "records and versioned model artifacts.",
]

LIMITATIONS = [
    "Simulation, not deployment: all hospitals run inside one program on one computer. "
    "There is no network deployment and no real hospital is connected.",
    "Privacy: raw rows stay inside each simulated client, but model parameters, example "
    "counts and aggregate statistics are shared. No differential privacy, secure aggregation "
    "or encryption is implemented, so no formal privacy guarantee is claimed.",
    "Robustness: attacks are configured in experiments. Low trust or zero weight is a policy "
    "output, not proof of malicious behaviour, and the scaling attack is not detected.",
    "Clinical status: research prototype, not a medical device. Probabilities are "
    "model-estimated scores, not clinically calibrated risks, and are not a diagnosis.",
    "SHAP values explain how the model uses its inputs in log-odds units; they do not show "
    "causation.",
    "'Show design reference values' displays numbers transcribed from the design image; they "
    "are labelled and never mixed with experiment results.",
]


class AboutPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("About This Project",
                         "Trust-Aware Explainable Federated Learning for Personalized Healthcare "
                         "Risk Prediction")
        self.ctx = ctx
        main = Card()
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(ShieldLogo(64), 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        t = label("TA-EFL Health")
        t.setStyleSheet(f"color: {theme.TITLE}; font-size: 16pt; font-weight: 700;")
        col.addWidget(t)
        col.addWidget(label(
            "This system implements a trust-aware federated learning framework with explainable "
            "AI for healthcare risk prediction using multiple datasets. Built with Python, Flower, "
            "scikit-learn, SHAP and PyQt6.", wrap=True))
        rl.addLayout(col, 1)
        main.add(row)
        main.add(label("Technology Stack", "CardTitle"))
        stack = QWidget()
        sl = QHBoxLayout(stack)
        sl.setContentsMargins(0, 0, 0, 0)
        for name, color in STACK:
            dot = QLabel()
            dot.setPixmap(icons.pixmap("check", color, 16))
            sl.addWidget(dot)
            sl.addWidget(label(name))
            sl.addSpacing(10)
        sl.addStretch(1)
        main.add(stack)
        main.add(label(OBJECTIVE, wrap=True))
        self.versions = label("", "Muted", wrap=True)
        main.add(self.versions)

        team = Card("Project Team")
        self.team = make_table(["Name", "USN"])
        fill_table(self.team, ref.TEAM)
        self.team.setMinimumHeight(170)
        team.add(self.team)
        self.root.addWidget(hbox(main, team, stretch=[3, 2]))

        arch = Card("Architecture")
        for text in ARCHITECTURE:
            arch.add(label("•  " + text, wrap=True))
        self.root.addWidget(arch)
        lim = Card("Scope, privacy and limitations")
        for text in LIMITATIONS:
            lim.add(label("•  " + text, wrap=True))
        self.root.addWidget(lim)
        self.root.addStretch(1)

    def refresh(self) -> None:
        self.reference_banner.hide()
        try:
            import flwr
            import numpy
            import sklearn
            from PyQt6.QtCore import PYQT_VERSION_STR
            parts = [f"Flower {flwr.__version__}", f"scikit-learn {sklearn.__version__}",
                     f"NumPy {numpy.__version__}", f"PyQt {PYQT_VERSION_STR}"]
            try:
                import shap
                parts.append(f"SHAP {shap.__version__}")
            except Exception:  # noqa: BLE001
                parts.append("SHAP not importable")
            self.versions.setText("Installed: " + ", ".join(parts))
        except Exception as exc:  # noqa: BLE001
            self.versions.setText(f"Version information unavailable: {exc}")
