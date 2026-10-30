"""Federated training: configure, start/stop a genuine Flower experiment, watch progress."""
from __future__ import annotations

from datetime import datetime
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services import reference_data as ref
from app.services.context import AppContext
from app.ui.widgets.common import (
    Card,
    Page,
    badge,
    button,
    confirm,
    fill_table,
    fmt,
    hbox,
    label,
    make_table,
    table_item,
)
from app.ui.dataset_actions import import_dataset_dialog, open_download_page
from app.ui.workers import ExperimentWorker, run_in_background
from ml.datasets.loader import dataset_summary
from app.ui.labels import (
    TRUST_MODE_LABELS,
    attack_label,
    attack_summary,
    client_name,
    metric,
    status_label,
)
from experiments.store import is_current
from federated.clients.attacks import ATTACK_DESCRIPTIONS
from federated.server.runner import ExperimentConfig, ExperimentError, run_experiment
from federated.strategies.strategies import STRATEGY_LABELS
from ml.datasets.registry import DatasetRegistry

ATTACK_CHOICES = ["sign_flip", "scaling", "gaussian_noise", "label_flip", "invalid_update"]


def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _dspin(lo: float, hi: float, val: float, decimals: int, step: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    return s


class TrainingPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Federated Training", "Configure parameters and start federated learning")
        self.ctx = ctx
        self.worker: ExperimentWorker | None = None
        self._total_rounds = 0
        self._latest_auc: float | None = None

        # ------------------------------------------------ configuration card
        self.cfg_card = Card("Training Configuration")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        self.dataset = QComboBox()
        self.dataset.currentIndexChanged.connect(lambda _i: self._describe_dataset())
        self.clients = _spin(2, 20, 5)
        self.clients.valueChanged.connect(self._fill_malicious)
        self.rounds = _spin(1, 500, 30)
        self.lr = _dspin(0.00001, 1.0, 0.01, 5, 0.005)
        self.batch = _spin(1, 4096, 32)
        self.strategy = QComboBox()
        for key, lab in (("trust", STRATEGY_LABELS["trust"]), ("fedavg", STRATEGY_LABELS["fedavg"]),
                         ("median", STRATEGY_LABELS["median"])):
            self.strategy.addItem(lab, key)
        self.malicious = QComboBox()
        self.attack = QComboBox()
        for key in ATTACK_CHOICES:
            self.attack.addItem(attack_label(key), key)
        self.attack.currentIndexChanged.connect(
            lambda _i: self.attack.setToolTip(ATTACK_DESCRIPTIONS[self.attack.currentData()]))
        self.alpha = _dspin(0.0, 0.99, 0.70, 2, 0.05)
        form.addRow("Dataset", self.dataset)
        form.addRow("Number of Clients", self.clients)
        form.addRow("Number of Rounds", self.rounds)
        form.addRow("Learning Rate", self.lr)
        form.addRow("Batch Size", self.batch)
        form.addRow("Aggregation Strategy", self.strategy)
        form.addRow("Malicious Client", self.malicious)
        form.addRow("Attack Type", self.attack)
        form.addRow("Trust Alpha", self.alpha)
        self.cfg_card.body.addLayout(form)

        self.adv_toggle = QCheckBox("Show advanced settings")
        self.cfg_card.add(self.adv_toggle)
        self.adv = QGroupBox()
        af = QFormLayout(self.adv)
        self.epochs = _spin(1, 50, 5)
        self.l2 = _dspin(0.0, 1.0, 0.0001, 5, 0.0001)
        self.class_weight = QComboBox()
        self.class_weight.addItem("None", "none")
        self.class_weight.addItem("Balanced", "balanced")
        self.partition = QComboBox()
        self.partition.addItem("IID (stratified)", "iid")
        self.partition.addItem("Non-IID (Dirichlet label skew)", "dirichlet")
        self.dir_alpha = _dspin(0.05, 100.0, 0.5, 2, 0.1)
        self.fraction = _dspin(0.1, 1.0, 1.0, 2, 0.1)
        self.seed = _spin(0, 2_000_000_000, 42)
        self.t_init = _dspin(0.0, 1.0, 0.5, 2, 0.05)
        self.t_perf = _dspin(0.0, 1.0, 0.6, 2, 0.05)
        self.t_thresh = _dspin(0.0, 1.0, 0.3, 2, 0.05)
        self.t_mode = QComboBox()
        for key, lab in TRUST_MODE_LABELS.items():
            self.t_mode.addItem(lab, key)
        self.a_scale = _dspin(0.1, 100.0, 3.0, 2, 0.5)
        self.a_noise = _dspin(0.01, 100.0, 1.0, 2, 0.1)
        self.threshold = _dspin(0.05, 0.95, 0.5, 2, 0.05)
        for text, w in (("Local epochs per round", self.epochs), ("L2 penalty", self.l2),
                        ("Class weighting", self.class_weight), ("Client partition", self.partition),
                        ("Dirichlet alpha", self.dir_alpha), ("Fraction of clients per round", self.fraction),
                        ("Random seed", self.seed), ("Initial trust T(0)", self.t_init),
                        ("Performance weight λ", self.t_perf), ("Exclusion threshold τ", self.t_thresh),
                        ("Performance signal", self.t_mode),
                        ("Attack scale s", self.a_scale), ("Attack noise σ", self.a_noise),
                        ("Decision threshold", self.threshold)):
            af.addRow(text, w)
        self.adv.setVisible(False)
        self.adv_toggle.toggled.connect(self.adv.setVisible)
        self.cfg_card.add(self.adv)

        self.start_btn = button("Start Training", "Success", "play", self._start)
        self.stop_btn = button("Stop", None, "stop", self._stop)
        self.stop_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(40)
        self.stop_btn.setMinimumHeight(40)
        self.cfg_card.add(hbox(self.start_btn, self.stop_btn, stretch=[3, 1]))
        self.cfg_card.add(label("Training runs a real Flower server with one Flower client per "
                                "simulated hospital. The window stays responsive; Stop takes effect "
                                "between rounds and keeps the model from the last completed round.",
                                "Muted", wrap=True))

        # ------------------------------------------------ progress card
        self.prog_card = Card("Training Progress")
        self.round_label = label("Round 0 / 0")
        self.round_label.setStyleSheet("font-size: 15pt; font-weight: 700; color: #0F172A;")
        self.status_badge = badge("idle", "neutral")
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.round_label)
        tl.addStretch(1)
        tl.addWidget(self.status_badge)
        self.prog_card.add(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.percent = label("0%")
        self.prog_card.add(hbox(self.progress, self.percent, stretch=[1, 0]))
        self.auc_label = label("Global model validation ROC-AUC: —")
        self.prog_card.add(self.auc_label)

        self.log_card = Card("Training Log")
        self.show_flower = QCheckBox("Show Flower server log")
        self.log_card.header.addWidget(self.show_flower)
        self.log = QPlainTextEdit()
        self.log.setObjectName("Log")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        self.log.setMaximumBlockCount(5000)
        self.log_card.add(self.log)

        self.clients_card = Card("Clients – latest round")
        self.client_table = make_table(["Client", "Valid", "Val AUC", "Trust", "Weight", "Status"])
        self.client_table.setMinimumHeight(200)
        self.clients_card.add(self.client_table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        rl.addWidget(self.prog_card)
        rl.addWidget(self.log_card, 1)
        self.ds_card = Card("Selected Dataset")
        self.ds_status = badge("checking…", "neutral")
        self.ds_card.header.addWidget(self.ds_status)
        self.ds_text = label("", wrap=True)
        self.ds_text.setTextFormat(Qt.TextFormat.RichText)
        self.ds_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ds_card.add(self.ds_text)
        self.ds_import_btn = button("Import downloaded file…", "Primary", "folder",
                                    lambda: self._import_dataset())
        self.ds_open_btn = button("Open download page", None, None,
                                  lambda: open_download_page(self.dataset.currentData()))
        self.ds_actions = hbox(self.ds_import_btn, self.ds_open_btn, label(""),
                               stretch=[0, 0, 1])
        self.ds_actions.hide()
        self.ds_card.add(self.ds_actions)
        self.root.addWidget(self.ds_card)
        self.root.addWidget(hbox(self.cfg_card, right, stretch=[4, 5]))
        self.root.addWidget(self.clients_card)

        self.hist_card = Card("Experiment History")
        self.history = make_table(["Created", "Dataset", "Strategy", "Attack", "Rounds", "Status",
                                   "Test AUC"])
        self.history.setMinimumHeight(200)
        self.history.cellDoubleClicked.connect(self._open_history)
        self.hist_card.add(self.history)
        self.root.addWidget(self.hist_card)
        self.root.addStretch(1)
        self._navigate = None
        self._loaded_settings = False

    def set_navigator(self, nav) -> None:
        self._navigate = nav

    # ------------------------------------------------------------ population
    def refresh(self) -> None:
        if not self._loaded_settings:
            self.load_from_settings()
            self._loaded_settings = True
        self._fill_datasets()
        self._fill_history()
        running = self.worker is not None
        if self.ctx.reference_mode and not running:
            self.reference_banner.show()
            t = ref.TRAINING
            self.round_label.setText(f"Round {t['progress_round']} / {t['rounds']}")
            self.progress.setValue(t["progress_percent"])
            self.percent.setText(f"{t['progress_percent']}%")
            self.auc_label.setText(f"Global Model ROC-AUC: {t['global_auc']:.3f} (reference)")
            self.log.setPlainText("\n".join(t["log"]) + "\n\n[reference log text from the design image]")
            self.status_badge.setText("reference")
        else:
            self.reference_banner.hide()
            if not running and self.status_badge.text() == "reference":
                self._reset_progress()

    def load_from_settings(self) -> None:
        s = self.ctx.settings
        self.clients.setValue(s.num_clients)
        self.rounds.setValue(s.num_rounds)
        self.lr.setValue(s.learning_rate)
        self.batch.setValue(s.batch_size)
        strat = "trust" if s.trust_aware_enabled else s.baseline_strategy
        self.strategy.setCurrentIndex(max(0, self.strategy.findData(strat)))
        self.alpha.setValue(s.trust_alpha)
        self.epochs.setValue(s.local_epochs)
        self.l2.setValue(s.l2)
        self.class_weight.setCurrentIndex(max(0, self.class_weight.findData(s.class_weight)))
        self.partition.setCurrentIndex(max(0, self.partition.findData(s.partition)))
        self.dir_alpha.setValue(s.dirichlet_alpha)
        self.fraction.setValue(s.fraction_fit)
        self.seed.setValue(s.seed)
        self.t_init.setValue(s.trust_initial)
        self.t_perf.setValue(s.trust_performance_weight)
        self.t_thresh.setValue(s.trust_exclusion_threshold)
        self.t_mode.setCurrentIndex(max(0, self.t_mode.findData(s.trust_performance_mode)))
        self.a_scale.setValue(s.attack_scale)
        self.a_noise.setValue(s.attack_noise_std)
        self.threshold.setValue(s.decision_threshold)
        self.attack.setCurrentIndex(max(0, self.attack.findData(s.attack_type)))
        self._fill_malicious()
        mal = s.malicious_clients[0] if (s.malicious_simulation_enabled and s.malicious_clients) else 0
        self.malicious.setCurrentIndex(max(0, self.malicious.findData(mal)))

    def _fill_datasets(self) -> None:
        current = self.dataset.currentData() or self.ctx.dataset_key
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for av in self.ctx.registry.all_availability():
            self.dataset.addItem(av.spec.name + ("" if av.available else "  (file missing)"), av.spec.key)
            self.dataset.setItemData(self.dataset.count() - 1, av.message, Qt.ItemDataRole.ToolTipRole)
        self.dataset.setCurrentIndex(max(0, self.dataset.findData(current)))
        self.dataset.blockSignals(False)
        self._describe_dataset()

    def _describe_dataset(self) -> None:
        key = self.dataset.currentData()
        if not key:
            return
        self._ds_key = key
        self.ds_status.setText("checking…")
        self.ds_text.setText("Validating the dataset file…")
        run_in_background(dataset_summary, key, self.ctx.settings.data_path,
                          on_done=self._show_dataset, on_error=lambda m, _d: self._show_dataset(
                              {"key": key, "status": "invalid", "error": m, "name": key}))

    def _show_dataset(self, d: dict) -> None:
        if d.get("key") != getattr(self, "_ds_key", None):
            return  # a newer selection is already being described
        status = d["status"]
        kind = "ok" if status == "valid" else ("warn" if status.startswith("valid") else "bad")
        self.ds_status.setText({"missing": "file missing", "invalid": "invalid"}.get(status, status))
        self.ds_status.setStyleSheet(badge("", kind).styleSheet())
        html = (f"<b>{d['name']}</b> – {d.get('description', '')}<br>"
                f"<b>Target:</b> {d.get('target', '—')} (1 = {d.get('positive_label', '?')}, "
                f"0 = {d.get('negative_label', '?')}) · <b>Features:</b> {d.get('n_features', '—')}<br>")
        if status == "missing":
            html += f"<span style='color:#B45309'>{d.get('message', '')}</span>"
        elif status == "invalid":
            html += f"<span style='color:#B91C1C'>Cannot be used: {d.get('error')}</span>"
        else:
            miss = d.get("missing") or {}
            html += (f"<b>Usable records:</b> {d['rows']} · <b>Class distribution:</b> "
                     f"{d['positives']} positive ({100 * d['positive_rate']:.1f}%), "
                     f"{d['negatives']} negative · <b>Duplicates:</b> {d['duplicates']}<br>"
                     f"<b>Missing values (imputed from training rows only):</b> "
                     + (", ".join(f"{k} {v}" for k, v in miss.items()) or "none") + "<br>"
                     f"<b>Preprocessing:</b> mean / most-frequent imputation, standardisation, "
                     f"one-hot encoding – fitted from client training rows during training<br>")
            for w in d.get("warnings", []):
                html += f"<span style='color:#B45309'>⚠ {w}</span><br>"
        if d.get("source"):
            html += f"<span style='color:#6B7280'>Source: {d['source']}. {d.get('licence', '')}</span>"
        self.ds_text.setText(html)
        self.start_btn.setEnabled(self.worker is None and status.startswith("valid"))
        self.ds_actions.setVisible(status in ("missing", "invalid") and d.get("key") != "breast_cancer")

    def _import_dataset(self) -> None:
        key = self.dataset.currentData()
        if key and import_dataset_dialog(self, self.ctx, key):
            self._fill_datasets()

    def _fill_malicious(self, *_args) -> None:
        cur = self.malicious.currentData()
        self.malicious.clear()
        self.malicious.addItem("None", 0)
        for k in range(1, self.clients.value() + 1):
            self.malicious.addItem(f"Client {k}", k)
        idx = self.malicious.findData(cur)
        self.malicious.setCurrentIndex(idx if idx >= 0 else 0)

    def _fill_history(self) -> None:
        rows = []
        for r in self.ctx.store.list()[:20]:
            c = r["config"]
            tm = r.get("test_metrics") or {}
            att = attack_summary(c)
            stale = not is_current(r)
            rows.append([table_item(r.get("created", "")[:16].replace("T", " ")
                                    + ("  (superseded)" if stale else ""), data=r["id"]),
                         r.get("dataset", {}).get("name", c["dataset_key"]),
                         r.get("strategy_label", c["strategy"]), att,
                         f"{r.get('rounds_completed', '—')}/{c['num_rounds']}",
                         badge(status_label(r.get("status")), {"completed": "ok", "stopped": "warn"}.get(
                             r.get("status"), "bad")),
                         "superseded" if stale else metric(tm.get("roc_auc"), missing="—")])
        fill_table(self.history, rows)

    # ------------------------------------------------------------ actions
    def current_config(self) -> ExperimentConfig:
        mal = self.malicious.currentData() or 0
        return ExperimentConfig(
            dataset_key=self.dataset.currentData(), num_clients=self.clients.value(),
            num_rounds=self.rounds.value(), learning_rate=self.lr.value(),
            batch_size=self.batch.value(), local_epochs=self.epochs.value(), l2=self.l2.value(),
            class_weight=self.class_weight.currentData(), strategy=self.strategy.currentData(),
            trust_alpha=self.alpha.value(), trust_initial=self.t_init.value(),
            trust_performance_weight=self.t_perf.value(),
            trust_exclusion_threshold=self.t_thresh.value(),
            trust_performance_mode=self.t_mode.currentData(), partition=self.partition.currentData(),
            dirichlet_alpha=self.dir_alpha.value(), fraction_fit=self.fraction.value(),
            malicious_clients=[mal] if mal else [],
            attack_type=self.attack.currentData() if mal else "none",
            attack_scale=self.a_scale.value(), attack_noise_std=self.a_noise.value(),
            seed=self.seed.value(), decision_threshold=self.threshold.value())

    def _start(self) -> None:
        if self.worker is not None:
            return
        cfg = self.current_config()
        try:
            cfg.validate()
        except ExperimentError as exc:
            self.show_message("error", f"Invalid configuration: {exc}")
            return
        av = self.ctx.registry.availability(cfg.dataset_key)
        if not av.available:
            self.show_message("error", av.message)
            return
        self.clear_message()
        self.ctx.set_dataset(cfg.dataset_key)
        self._reset_progress()
        self._total_rounds = cfg.num_rounds
        self.status_badge.setText("running")
        self._append(f"{datetime.now():%H:%M:%S}  Starting experiment on {av.spec.name} "
                     f"({cfg.num_clients} clients, {cfg.num_rounds} rounds, "
                     f"{STRATEGY_LABELS[cfg.strategy]})")
        if cfg.malicious_clients:
            self._append(f"Robustness experiment: Client {cfg.malicious_clients[0]} configured "
                         f"as attacker ({attack_label(cfg.attack_type)}).")
        settings = self.ctx.settings
        fn = partial(run_experiment, cfg, settings.data_path, settings.storage_path)
        self.worker = ExperimentWorker(fn)
        # Only bound methods of this page are connected, so slots run in the GUI thread.
        self.worker.progress.connect(self._on_event)
        self.worker.succeeded.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(self._on_thread_done)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.ctx.training_running = True
        self.worker.start()

    def _stop(self) -> None:
        if self.worker and confirm(self, "Stop training",
                                   "Stop after the current round? The model from the last "
                                   "completed round will be evaluated and saved."):
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self._append("Stop requested – finishing the current round...")

    def _reset_progress(self) -> None:
        self.round_label.setText("Round 0 / 0")
        self.progress.setValue(0)
        self.percent.setText("0%")
        self.auc_label.setText("Global model validation ROC-AUC: —")
        self.log.clear()
        self.client_table.setRowCount(0)
        self.status_badge.setText("idle")

    def _append(self, line: str) -> None:
        self.log.appendPlainText(line)

    # ------------------------------------------------------------ worker slots
    def _on_event(self, e: dict) -> None:
        t = e.get("type")
        if t == "flower_log":
            if self.show_flower.isChecked():
                self._append(f"  flwr | {e['message']}")
            return
        if t == "status" or t == "log":
            self._append(e.get("message", ""))
        elif t == "round_start":
            self._append(f"[Round {e['round']}] Global model sent to {len(e['clients'])} clients.")
        elif t == "round_aggregated":
            r = e["round"]
            n_ok = sum(1 for c in e["clients"] if c["valid"])
            self._append(f"[Round {r}] {n_ok} client updates received.")
            self._append(f"[Round {r}] Trust scores updated"
                         + ("." if e["trust_used_for_weighting"] else " (monitored only)."))
            excl = [client_name(c["cid"]) for c in e["clients"] if c["excluded"]]
            self._append(f"[Round {r}] Aggregating client updates"
                         + (f" – zero weight (rejected or below trust threshold): "
                            f"{', '.join(excl)}" if excl else "") + "...")
            for f in e.get("failures", []):
                self._append(f"[Round {r}] {client_name(f['cid'])} failed: {f['error']}")
            pct = int(round(100 * r / max(1, self._total_rounds)))
            self.round_label.setText(f"Round {r} / {self._total_rounds}")
            self.progress.setValue(pct)
            self.percent.setText(f"{pct}%")
            self._fill_clients(e)
        elif t == "evaluate":
            if e["round"] == 0:
                return
            self._latest_auc = e.get("auc")
            self.auc_label.setText(f"Global model validation ROC-AUC: {fmt(e.get('auc'))}  "
                                   f"(accuracy {e['accuracy']:.3f}, round {e['round']})")
            self._append(f"[Round {e['round']}] Global model AUC = {fmt(e.get('auc'))} (validation)")

    def _fill_clients(self, e: dict) -> None:
        rows = []
        for c in e["clients"]:
            status = ("rejected" if not c["valid"] else "zero weight" if c["excluded"] else "included")
            rows.append([client_name(c["cid"]),
                         "yes" if c["valid"] else "no", fmt(c["val_auc"]), fmt(c["trust"]),
                         "median" if c["weight"] is None else fmt(c["weight"]),
                         badge(status, {"included": "ok", "zero weight": "warn", "rejected": "bad"}[status])])
        fill_table(self.client_table, rows)

    def _on_finished(self, record: dict) -> None:
        tm = record.get("test_metrics", {})
        self.status_badge.setText(record.get("status", "done"))
        self.show_message("success", f"Experiment {record['status']}: {record['rounds_completed']} "
                          f"rounds. Held-out test ROC-AUC {fmt(tm.get('roc_auc'))}, accuracy "
                          f"{fmt(tm.get('accuracy'))}. Model saved – use Risk Prediction or "
                          f"Results & Analysis.")
        self.ctx.select_experiment(record["id"])
        self.ctx.notify_experiments_changed()

    def _on_failed(self, message: str, details: str) -> None:
        self.status_badge.setText("failed")
        self._append(f"ERROR: {message}")
        self.show_message("error", f"Training failed: {message}")
        self.ctx.notify_experiments_changed()

    def _on_thread_done(self) -> None:
        if self.worker is not None:
            self.worker.wait(5000)  # make sure the QThread has fully exited before release
        self.worker = None
        self.ctx.training_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._fill_history()

    def _open_history(self, row: int, _col: int) -> None:
        it = self.history.item(row, 0)
        if it is not None:
            self.ctx.select_experiment(it.data(Qt.ItemDataRole.UserRole))
            if self._navigate:
                self._navigate("results")

    def shutdown(self) -> None:
        """Called when the window closes: stop and wait for the worker thread."""
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(30000)
