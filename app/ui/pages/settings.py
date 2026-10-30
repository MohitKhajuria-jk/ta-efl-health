"""System settings: dataset, training, trust, paths and logging (validated, persisted)."""
from __future__ import annotations

import copy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.configuration.settings import AppSettings, SettingsError
from app.services.context import AppContext
from app.ui.widgets.common import Card, Page, badge, button, fill_table, hbox, label, make_table
from app.ui.dataset_actions import import_dataset_dialog, open_download_page
from app.ui.labels import TRUST_MODE_LABELS, attack_label
from federated.clients.attacks import ATTACK_DESCRIPTIONS
from ml.datasets.registry import DatasetRegistry


def _spin(lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    return s


def _dspin(lo, hi, dec, step):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    return s


def _form() -> tuple[QWidget, QFormLayout]:
    w = QWidget()
    f = QFormLayout(w)
    f.setHorizontalSpacing(18)
    f.setVerticalSpacing(9)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    return w, f


class SettingsPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("System Settings", "Configure dataset, model and federated learning parameters")
        self.ctx = ctx
        self.tabs = QTabWidget()

        # ---- dataset settings (layout of the design reference) -------------
        tab = QWidget()
        tl = QHBoxLayout(tab)
        left, lf = _form()
        self.dataset = QComboBox()
        for k in DatasetRegistry.keys():
            self.dataset.addItem(DatasetRegistry.get(k).name, k)
        self.data_dir = QLineEdit()
        browse = button("Browse", None, on_click=self._browse_data)
        self.clients = _spin(2, 20)
        self.rounds = _spin(1, 500)
        self.batch = _spin(1, 4096)
        self.lr = _dspin(0.00001, 1.0, 5, 0.005)
        lf.addRow("Default Dataset", self.dataset)
        lf.addRow("Data Directory", hbox(self.data_dir, browse, stretch=[1, 0], spacing=6))
        lf.addRow("Number of Clients", self.clients)
        lf.addRow("Number of Rounds", self.rounds)
        lf.addRow("Batch Size", self.batch)
        lf.addRow("Learning Rate", self.lr)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.trust_enabled = QCheckBox("Enable Trust-Aware Aggregation")
        self.malicious_enabled = QCheckBox("Enable Malicious Client Simulation")
        self.save_logs = QCheckBox("Save Experiment Logs")
        self.gen_charts = QCheckBox("Generate Charts Automatically")
        for cb in (self.trust_enabled, self.malicious_enabled, self.save_logs, self.gen_charts):
            rl.addWidget(cb)
        rl.addStretch(1)
        tl.addWidget(left, 3)
        tl.addSpacing(30)
        tl.addWidget(right, 2)
        self.tabs.addTab(tab, "Dataset Settings")

        # ---- training parameters --------------------------------------------
        tab, f = _form()
        self.epochs = _spin(1, 50)
        self.l2 = _dspin(0.0, 1.0, 5, 0.0001)
        self.class_weight = QComboBox()
        self.class_weight.addItem("None", "none")
        self.class_weight.addItem("Balanced", "balanced")
        self.partition = QComboBox()
        self.partition.addItem("IID (stratified)", "iid")
        self.partition.addItem("Non-IID (Dirichlet label skew)", "dirichlet")
        self.dir_alpha = _dspin(0.05, 100.0, 2, 0.1)
        self.fraction = _dspin(0.1, 1.0, 2, 0.1)
        self.seed = _spin(0, 2_000_000_000)
        self.threshold = _dspin(0.05, 0.95, 2, 0.05)
        self.baseline = QComboBox()
        self.baseline.addItem("FedAvg (Flower)", "fedavg")
        self.baseline.addItem("Coordinate-wise median (Flower FedMedian)", "median")
        for t, w in (("Local epochs per round", self.epochs), ("L2 penalty", self.l2),
                     ("Class weighting", self.class_weight), ("Client partition", self.partition),
                     ("Dirichlet alpha (non-IID)", self.dir_alpha),
                     ("Fraction of clients per round", self.fraction), ("Random seed", self.seed),
                     ("Decision threshold (fixed before training)", self.threshold),
                     ("Strategy when trust-aware is disabled", self.baseline)):
            f.addRow(t, w)
        self.tabs.addTab(tab, "Training Parameters")

        # ---- trust parameters ---------------------------------------------------
        tab, f = _form()
        self.t_alpha = _dspin(0.0, 0.99, 2, 0.05)
        self.t_init = _dspin(0.0, 1.0, 2, 0.05)
        self.t_perf = _dspin(0.0, 1.0, 2, 0.05)
        self.t_thresh = _dspin(0.0, 1.0, 2, 0.05)
        self.mal_clients = QLineEdit()
        self.mal_clients.setPlaceholderText("e.g. 4  or  2, 4")
        self.attack = QComboBox()
        for k in ATTACK_DESCRIPTIONS:
            if k != "none":
                self.attack.addItem(attack_label(k), k)
        self.t_mode = QComboBox()
        for k, lab in TRUST_MODE_LABELS.items():
            self.t_mode.addItem(lab, k)
        self.a_scale = _dspin(0.1, 100.0, 2, 0.5)
        self.a_noise = _dspin(0.01, 100.0, 2, 0.1)
        for t, w in (("Trust alpha (history weight)", self.t_alpha), ("Initial trust T(0)", self.t_init),
                     ("Performance weight λ", self.t_perf), ("Exclusion threshold τ", self.t_thresh),
                     ("Performance signal", self.t_mode),
                     ("Malicious clients (numbers)", self.mal_clients), ("Attack type", self.attack),
                     ("Attack scale s", self.a_scale), ("Attack noise σ", self.a_noise)):
            f.addRow(t, w)
        f.addRow(label("T_k(t) = α·T_k(t-1) + (1-α)·[λ·P_k + (1-λ)·S_k];  "
                       "ω_k = T_k·n_k / Σ T_j·n_j for T_k ≥ τ. Relative P_k compares a client's "
                       "validation AUC with the round's median client (honest-majority "
                       "assumption); absolute P_k uses (AUC − 0.5)/0.5 and can push honest "
                       "clients below τ on harder datasets.", "Muted", wrap=True))
        self.tabs.addTab(tab, "Trust Parameters")

        # ---- paths & logging ------------------------------------------------------
        tab, f = _form()
        self.storage_dir = QLineEdit()
        sbrowse = button("Browse", None, on_click=self._browse_storage)
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        f.addRow("Storage directory (models, experiments, logs)",
                 hbox(self.storage_dir, sbrowse, stretch=[1, 0], spacing=6))
        f.addRow("Log level", self.log_level)
        f.addRow(label("Relative paths are resolved from the application folder. Patient inputs "
                       "entered for prediction are never written to the log.", "Muted", wrap=True))
        self.tabs.addTab(tab, "Paths & Logging")

        card = Card()
        card.add(self.tabs)
        self.save_btn = button("Save Settings", "Primary", on_click=self._save)
        self.save_btn.setMinimumHeight(40)
        self.reset_btn = button("Restore defaults", None, on_click=lambda: self._load(AppSettings()))
        row = QWidget()
        rl2 = QHBoxLayout(row)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.addStretch(1)
        rl2.addWidget(self.reset_btn)
        rl2.addWidget(self.save_btn)
        card.add(row)
        self.root.addWidget(card)

        self.avail_card = Card("Dataset availability")
        self.avail = make_table(["Dataset", "Status", "Expected file", "Action"])
        self.avail.setMinimumHeight(210)
        self.avail_card.add(self.avail)
        self.root.addWidget(self.avail_card)
        self.root.addStretch(1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.reference_banner.hide()
        self._load(self.ctx.settings)
        for w in self.ctx.settings_manager.warnings:
            self.show_message("warning", w)

    def _load(self, s: AppSettings) -> None:
        self.dataset.setCurrentIndex(max(0, self.dataset.findData(s.default_dataset)))
        self.data_dir.setText(s.data_dir)
        self.clients.setValue(s.num_clients)
        self.rounds.setValue(s.num_rounds)
        self.batch.setValue(s.batch_size)
        self.lr.setValue(s.learning_rate)
        self.trust_enabled.setChecked(s.trust_aware_enabled)
        self.malicious_enabled.setChecked(s.malicious_simulation_enabled)
        self.save_logs.setChecked(s.save_experiment_logs)
        self.gen_charts.setChecked(s.generate_charts)
        self.epochs.setValue(s.local_epochs)
        self.l2.setValue(s.l2)
        self.class_weight.setCurrentIndex(max(0, self.class_weight.findData(s.class_weight)))
        self.partition.setCurrentIndex(max(0, self.partition.findData(s.partition)))
        self.dir_alpha.setValue(s.dirichlet_alpha)
        self.fraction.setValue(s.fraction_fit)
        self.seed.setValue(s.seed)
        self.threshold.setValue(s.decision_threshold)
        self.baseline.setCurrentIndex(max(0, self.baseline.findData(s.baseline_strategy)))
        self.t_alpha.setValue(s.trust_alpha)
        self.t_init.setValue(s.trust_initial)
        self.t_perf.setValue(s.trust_performance_weight)
        self.t_thresh.setValue(s.trust_exclusion_threshold)
        self.t_mode.setCurrentIndex(max(0, self.t_mode.findData(s.trust_performance_mode)))
        self.mal_clients.setText(", ".join(str(m) for m in s.malicious_clients))
        self.attack.setCurrentIndex(max(0, self.attack.findData(s.attack_type)))
        self.a_scale.setValue(s.attack_scale)
        self.a_noise.setValue(s.attack_noise_std)
        self.storage_dir.setText(s.storage_dir)
        self.log_level.setCurrentText(s.log_level)
        self._fill_availability(s)

    def _fill_availability(self, s: AppSettings) -> None:
        reg = DatasetRegistry(s.data_path)
        rows = []
        for av in reg.all_availability():
            if av.spec.file_name is None:
                action = label("bundled")
            else:
                key = av.spec.key
                action = hbox(button("Import…", None, None, lambda k=key: self._import(k)),
                              button("Download page", None, None, lambda k=key: open_download_page(k)),
                              spacing=4)
            rows.append([av.spec.name, badge("available" if av.available else "missing",
                                             "ok" if av.available else "warn"),
                         av.spec.file_name or "bundled (scikit-learn)", action])
        fill_table(self.avail, rows)
        self.avail.verticalHeader().setDefaultSectionSize(40)
        self.avail.resizeRowsToContents()

    def _import(self, key: str) -> None:
        if import_dataset_dialog(self, self.ctx, key):
            self._fill_availability(self.ctx.settings)

    def _collect(self) -> AppSettings:
        s = copy.deepcopy(self.ctx.settings)
        s.default_dataset = self.dataset.currentData()
        s.data_dir = self.data_dir.text().strip()
        s.num_clients = self.clients.value()
        s.num_rounds = self.rounds.value()
        s.batch_size = self.batch.value()
        s.learning_rate = self.lr.value()
        s.trust_aware_enabled = self.trust_enabled.isChecked()
        s.malicious_simulation_enabled = self.malicious_enabled.isChecked()
        s.save_experiment_logs = self.save_logs.isChecked()
        s.generate_charts = self.gen_charts.isChecked()
        s.local_epochs = self.epochs.value()
        s.l2 = self.l2.value()
        s.class_weight = self.class_weight.currentData()
        s.partition = self.partition.currentData()
        s.dirichlet_alpha = self.dir_alpha.value()
        s.fraction_fit = self.fraction.value()
        s.seed = self.seed.value()
        s.decision_threshold = self.threshold.value()
        s.baseline_strategy = self.baseline.currentData()
        s.trust_alpha = self.t_alpha.value()
        s.trust_initial = self.t_init.value()
        s.trust_performance_weight = self.t_perf.value()
        s.trust_exclusion_threshold = self.t_thresh.value()
        s.trust_performance_mode = self.t_mode.currentData()
        text = self.mal_clients.text().replace(";", ",")
        try:
            s.malicious_clients = [int(x) for x in text.split(",") if x.strip()]
        except ValueError:
            raise SettingsError("Malicious clients must be a comma-separated list of numbers")
        s.attack_type = self.attack.currentData()
        s.attack_scale = self.a_scale.value()
        s.attack_noise_std = self.a_noise.value()
        s.storage_dir = self.storage_dir.text().strip()
        s.log_level = self.log_level.currentText()
        return s

    def _save(self) -> None:
        try:
            s = self._collect()
            if s.malicious_simulation_enabled and not s.malicious_clients:
                raise SettingsError("Enter at least one malicious client number, or disable the "
                                    "malicious client simulation")
            self.ctx.apply_settings(s)
        except SettingsError as exc:
            self.show_message("error", f"Settings not saved: {exc}")
            return
        self.show_message("success", f"Settings saved to {self.ctx.settings_manager.path}.")
        self._fill_availability(s)

    def _browse_data(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select data directory",
                                             str(self.ctx.settings.data_path))
        if d:
            self.data_dir.setText(d)

    def _browse_storage(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select storage directory",
                                             str(self.ctx.settings.storage_path))
        if d:
            self.storage_dir.setText(d)
