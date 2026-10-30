"""Results & experimental analysis: test metrics, ROC, sensitivity sweeps, comparisons."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QProgressBar, QTabWidget, QVBoxLayout, QWidget

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import (
    Card,
    ExperimentPicker,
    Page,
    button,
    fill_table,
    fmt,
    hbox,
    label,
    make_table,
    table_item,
)
from app.ui.labels import STRATEGY_SHORT, attack_summary, metric, status_label
from app.ui.workers import ExperimentWorker
from experiments.sweeps import SWEEPS, list_sweeps, run_sweep
from ml.datasets.registry import DatasetRegistry

METRICS = [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
           ("Recall (sensitivity)", "recall"), ("F1-Score", "f1"), ("Brier score", "brier"),
           ("Expected calibration error", "ece"), ("Log loss", "log_loss")]


def _tab() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    return w, lay


class ResultsPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Experimental Results", "Model performance and analysis")
        self.ctx = ctx
        self.sweep_worker: ExperimentWorker | None = None
        self.picker = ExperimentPicker()
        self.picker.currentIndexChanged.connect(self._picked)
        self.header.add_right(self.picker)

        self.tabs = QTabWidget()
        # ---- model performance
        w, lay = _tab()
        self.roc_card = Card("ROC Curve (Test Set)")
        self.roc = ChartCanvas(5, 3.4, 300)
        self.roc_card.add(self.roc)
        self.metrics_card = Card("Model Metrics")
        self.metrics = make_table(["Metric", "Test (held-out)", "Validation (server)",
                                   "Centralised reference (test)"])
        self.metrics.setMinimumHeight(300)
        self.metrics_card.add(self.metrics)
        self.split_note = label("", "Muted", wrap=True)
        self.metrics_card.add(self.split_note)
        lay.addWidget(hbox(self.roc_card, self.metrics_card, stretch=[3, 2]))
        self.cm_card = Card("Confusion Matrix & Calibration (Test Set)")
        self.cm = ChartCanvas(6, 3, 260)
        self.cm_card.add(self.cm)
        self.val_card = Card("Global Model on Validation Split per Round")
        self.val = ChartCanvas(5, 3, 250)
        self.val_card.add(self.val)
        lay.addWidget(hbox(self.cm_card, self.val_card, stretch=[2, 3]))
        self.tabs.addTab(w, "Model Performance")
        # ---- trust analysis
        w, lay = _tab()
        self.trust_card = Card("Client Trust Over Rounds")
        self.trust = ChartCanvas(6, 3.2, 290)
        self.trust_card.add(self.trust)
        self.detect = label("", wrap=True)
        self.trust_card.add(self.detect)
        lay.addWidget(self.trust_card)
        self.tabs.addTab(w, "Trust Analysis")
        # ---- sensitivity analysis
        w, lay = _tab()
        self.sweep_card = Card("Sensitivity Analysis (E6)")
        self.sweep_kind = QComboBox()
        for k, desc in SWEEPS.items():
            self.sweep_kind.addItem(desc, k)
        self.sweep_btn = button("Run sweep", "Primary", "play", self._run_sweep)
        self.sweep_stop = button("Stop", None, "stop", self._stop_sweep)
        self.sweep_stop.setEnabled(False)
        self.sweep_card.add(hbox(self.sweep_kind, self.sweep_btn, self.sweep_stop, stretch=[1, 0, 0]))
        self.sweep_progress = QProgressBar()
        self.sweep_status = label("Each sweep runs several complete Flower experiments with the "
                                  "settings from the Settings page, varying one factor. Results "
                                  "are test-set metrics of each run.", "Muted", wrap=True)
        self.sweep_card.add(self.sweep_progress)
        self.sweep_card.add(self.sweep_status)
        self.sweep_table = make_table(["Run", "Strategy", "Attackers", "Test ROC-AUC", "Accuracy",
                                       "F1", "Status"])
        self.sweep_table.setMinimumHeight(220)
        self.sweep_card.add(self.sweep_table)
        self.sweep_chart = ChartCanvas(6, 3, 260)
        self.sweep_card.add(self.sweep_chart)
        self.past_sweeps = QComboBox()
        self.past_sweeps.currentIndexChanged.connect(self._show_past_sweep)
        self.sweep_card.add(hbox(label("Saved sweeps"), self.past_sweeps, stretch=[0, 1]))
        lay.addWidget(self.sweep_card)
        self.tabs.addTab(w, "Sensitivity Analysis (E6)")
        # ---- comparison
        w, lay = _tab()
        self.cmp_card = Card("Aggregation Strategy Comparison (same dataset)")
        self.cmp_table = make_table(["Created", "Strategy", "Attack", "Rounds", "Test ROC-AUC",
                                     "Accuracy", "Precision", "Recall", "F1"])
        self.cmp_table.setMinimumHeight(240)
        self.cmp_card.add(self.cmp_table)
        self.cmp_chart = ChartCanvas(6, 3, 270)
        self.cmp_card.add(self.cmp_chart)
        self.cmp_note = label("Only experiments on the same dataset with identical split settings "
                              "(seed, fractions) are directly comparable.", "Muted", wrap=True)
        self.cmp_card.add(self.cmp_note)
        lay.addWidget(self.cmp_card)
        self.tabs.addTab(w, "Comparison")
        self.root.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self._fill_past_sweeps()
        if self.ctx.reference_mode:
            self._render_reference()
            return
        self.reference_banner.hide()
        recs = [r for r in self.ctx.store.list_current() if r.get("status") in ("completed", "stopped")]
        self.picker.populate(recs, self.ctx.selected_experiment_id)
        self._show(self.picker.current_id())

    def _picked(self, _i: int) -> None:
        exp_id = self.picker.current_id()
        if exp_id:
            self.ctx.select_experiment(exp_id)
        self._show(exp_id)

    def _show(self, exp_id: str | None) -> None:
        rec = None
        if exp_id:
            try:
                rec = self.ctx.store.load(exp_id)
            except FileNotFoundError:
                rec = None
        if rec is None:
            self.show_message("info", "No completed experiment yet. Results are computed only "
                              "from experiments that have actually been run.")
            for c in (self.roc, self.cm, self.val, self.trust, self.cmp_chart):
                charts.empty(c.reset()[0], "No experiment yet")
                c.finish()
            self.metrics.setRowCount(0)
            self.cmp_table.setRowCount(0)
            self.detect.setText("")
            self.split_note.setText("")
            return
        self.clear_message()
        tm = rec.get("test_metrics") or {}
        base = (rec.get("centralized_baseline") or {}).get("test_metrics") or {}
        charts.plot_roc(self.roc.reset()[0], tm.get("roc_curve"), tm.get("roc_auc"),
                        "Federated model", extra=[("Centralised reference", base.get("roc_curve"),
                                                   base.get("roc_auc"))] if base else [])
        self.roc.finish()
        vm = rec.get("final_validation_metrics") or {}
        fill_table(self.metrics, [[name, table_item(metric(tm.get(k)), bold=True), metric(vm.get(k)),
                                   metric(base.get(k)) if base else "Not run"]
                                  for name, k in METRICS])
        self.split_note.setText(
            f"Test split: n = {tm.get('n')}, {tm.get('positives')} positive (prevalence "
            f"{100 * tm.get('prevalence', 0):.1f}%), decision threshold {tm.get('threshold')} fixed "
            f"before training. Validation split (n = {vm.get('n')}) was used for trust and "
            f"per-round monitoring only; the test split was used for nothing but this final "
            f"evaluation. Probabilities are model-estimated scores: check the ECE and the "
            f"reliability plot before reading them as risks. N/A = not defined for this split.")
        spec = DatasetRegistry.get(rec["config"]["dataset_key"])
        a_cm, a_cal = self.cm.reset(2)
        charts.plot_confusion(a_cm, tm.get("confusion_matrix"),
                              (spec.negative_label, spec.positive_label))
        charts.plot_calibration(a_cal, tm.get("calibration_bins"), tm.get("ece"))
        self.cm.finish()
        charts.plot_validation_auc(self.val.reset()[0], rec)
        self.val.finish()
        charts.plot_lines(self.trust.reset()[0], rec, "trust", rec["config"]["trust_exclusion_threshold"])
        self.trust.finish()
        det = rec.get("detection", {})
        tp = sum(r["true_positives"] for r in det.get("per_round", []))
        fp = sum(r["false_positives"] for r in det.get("per_round", []))
        self.detect.setText(
            f"Configured attackers: {', '.join(det.get('configured_attackers', [])) or 'none'}. "
            f"Zero-weight decisions over all rounds: {tp} on configured attackers, {fp} on honest "
            f"clients. Trust used for weighting: "
            f"{'yes' if det.get('trust_used_for_weighting') else 'no (monitored only)'}. "
            f"{det.get('note', '')}")
        self._fill_comparison(rec)

    def _fill_comparison(self, rec: dict) -> None:
        key = rec["config"]["dataset_key"]
        recs = [r for r in self.ctx.store.list_current(key) if r.get("status") in ("completed", "stopped")]
        rows = []
        for r in recs[:30]:
            c = r["config"]
            tm = r.get("test_metrics") or {}
            att = attack_summary(c)
            rows.append([r.get("created", "")[:16].replace("T", " "), r.get("strategy_label"), att,
                         r.get("rounds_completed"), fmt(tm.get("roc_auc")), fmt(tm.get("accuracy")),
                         fmt(tm.get("precision")), fmt(tm.get("recall")), fmt(tm.get("f1"))])
        fill_table(self.cmp_table, rows,
                   lambda i, _row: "#EFF6FF" if recs[i]["id"] == rec["id"] else None)
        latest: dict[str, dict] = {}
        for r in recs:  # newest first: keep the latest run per (strategy, attack) setting
            c = r["config"]
            k = f"{r.get('strategy_label')}\n{attack_summary(c)}"
            latest.setdefault(k, r)
        labels = list(latest)[:8]
        charts.plot_bar_metrics(self.cmp_chart.reset()[0], labels,
                                [(latest[k].get("test_metrics") or {}).get("roc_auc") for k in labels],
                                "Test ROC-AUC – latest run per strategy/attack setting")
        self.cmp_chart.finish()

    # ------------------------------------------------------------------ sweeps
    def _run_sweep(self) -> None:
        if self.sweep_worker is not None or self.ctx.training_running:
            self.show_message("warning", "Another training job is running.")
            return
        s = self.ctx.settings
        base = s.experiment_config(dataset_key=self.ctx.dataset_key)
        av = self.ctx.registry.availability(base.dataset_key)
        if not av.available:
            self.show_message("error", av.message)
            return
        kind = self.sweep_kind.currentData()

        def job(on_event, stop_event):
            return run_sweep(kind, base, s.data_path, s.storage_path,
                             on_progress=lambda i, n, name: on_event({"i": i, "n": n, "name": name}),
                             stop_event=stop_event)

        self.sweep_worker = ExperimentWorker(job)
        self.sweep_worker.progress.connect(self._sweep_progress)
        self.sweep_worker.succeeded.connect(self._sweep_done)
        self.sweep_worker.failed.connect(self._sweep_failed)
        self.sweep_worker.done.connect(self._sweep_thread_done)
        self.sweep_btn.setEnabled(False)
        self.sweep_stop.setEnabled(True)
        self.ctx.training_running = True
        self.sweep_status.setText(f"Running sweep on {av.spec.name}…")
        self.sweep_worker.start()

    def _stop_sweep(self) -> None:
        if self.sweep_worker:
            self.sweep_worker.stop()
            self.sweep_status.setText("Stopping after the current round…")

    def _sweep_progress(self, e: dict) -> None:
        if "i" in e:
            self.sweep_progress.setValue(int(100 * (e["i"] - 1) / max(1, e["n"])))
            self.sweep_status.setText(f"Run {e['i']} of {e['n']}: {e['name']}")

    def _sweep_done(self, rows: list) -> None:
        self.sweep_progress.setValue(100)
        self.sweep_status.setText(f"Sweep finished: {len(rows)} experiments run and saved.")
        self._show_sweep_rows(rows)
        self.ctx.notify_experiments_changed()
        self._fill_past_sweeps()

    def _sweep_failed(self, msg: str, _details: str) -> None:
        self.sweep_status.setText(f"Sweep failed: {msg}")
        self.ctx.notify_experiments_changed()

    def _sweep_thread_done(self) -> None:
        if self.sweep_worker is not None:
            self.sweep_worker.wait(5000)
        self.sweep_worker = None
        self.ctx.training_running = False
        self.sweep_btn.setEnabled(True)
        self.sweep_stop.setEnabled(False)

    def _show_sweep_rows(self, rows: list) -> None:
        fill_table(self.sweep_table, [[r["run"], STRATEGY_SHORT.get(r["strategy"], r["strategy"]),
                                       r["attackers"], metric(r["roc_auc"]), metric(r["accuracy"]),
                                       metric(r["f1"]), status_label(r["status"])] for r in rows])
        charts.plot_bar_metrics(self.sweep_chart.reset()[0], [r["run"] for r in rows],
                                [r["roc_auc"] for r in rows], "Test ROC-AUC per run")
        self.sweep_chart.finish()

    def _fill_past_sweeps(self) -> None:
        self._sweeps = list_sweeps(self.ctx.settings.storage_path)
        self.past_sweeps.blockSignals(True)
        self.past_sweeps.clear()
        for i, s in enumerate(self._sweeps):
            self.past_sweeps.addItem(f"{s['created'].replace('T', ' ')} – {s['description']}", i)
        self.past_sweeps.setCurrentIndex(-1)
        self.past_sweeps.blockSignals(False)

    def _show_past_sweep(self, _i: int) -> None:
        idx = self.past_sweeps.currentData()
        if idx is not None and 0 <= idx < len(self._sweeps):
            self._show_sweep_rows(self._sweeps[idx]["rows"])

    def shutdown(self) -> None:
        if self.sweep_worker is not None:
            self.sweep_worker.stop()
            self.sweep_worker.wait(30000)

    # ------------------------------------------------------------------ reference
    def _render_reference(self) -> None:
        self.reference_banner.show()
        self.clear_message()
        fill_table(self.metrics, [[n, table_item(f"{v:.3f}", bold=True), "—", "—"]
                                  for n, v in ref.RESULTS_METRICS])
        self.split_note.setText("Reference values from the design image.")
        charts.empty(self.roc.reset()[0], "Reference AUC = 0.805 (curve points are not readable "
                     "from the design image)")
        self.roc.finish()
        for c in (self.cm, self.val, self.trust, self.cmp_chart):
            charts.empty(c.reset()[0], "Not shown in the design image")
            c.finish()
        self.detect.setText("")
