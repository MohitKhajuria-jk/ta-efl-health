"""Generate and export experiment reports (PDF / HTML / JSON)."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QFrame, QVBoxLayout

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.services.reports import ReportOptions, generate_report
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import Card, ExperimentPicker, Page, button, fmt, hbox, label
from app.ui.workers import run_in_background
from ml.explainability.shap_service import explanation_to_dict, global_importance

FORMATS = [("PDF Report", "pdf"), ("HTML Report", "html"), ("JSON Data", "json")]


def _report_job(record, path, fmt_key, opts, explanation, artifact):
    importance = None
    if opts.shap_explanation and artifact is not None:
        importance = global_importance(artifact).features
    return generate_report(record, path, fmt_key, opts, explanation, importance)


class ReportsPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Generate Report", "Export experiment results and analysis")
        self.ctx = ctx
        self._last_path: Path | None = None

        self.opt_card = Card("Report Options")
        self.c_details = QCheckBox("Experiment Details")
        self.c_metrics = QCheckBox("Model Performance Metrics")
        self.c_trust = QCheckBox("Trust Analysis")
        self.c_shap = QCheckBox("SHAP Explanation")
        self.c_charts = QCheckBox("Charts and Plots")
        self.c_dataset = QCheckBox("Dataset Information")
        for cb in (self.c_details, self.c_metrics, self.c_trust, self.c_shap, self.c_charts, self.c_dataset):
            cb.setChecked(True)
            self.opt_card.add(cb)
        self.out_card = Card("Output Format")
        self.fmt = QComboBox()
        for lab, key in FORMATS:
            self.fmt.addItem(lab, key)
        self.out_card.add(self.fmt)
        self.out_card.add(label("Experiment", "Muted"))
        self.picker = ExperimentPicker()
        self.picker.setMinimumWidth(260)
        self.picker.currentIndexChanged.connect(lambda _i: self._preview())
        self.out_card.add(self.picker)
        self.gen_btn = button("Generate Report", "Success", "reports", self._generate)
        self.gen_btn.setMinimumHeight(42)
        self.out_card.add(self.gen_btn)
        self.open_btn = button("Open containing folder", None, "folder", self._open_folder)
        self.open_btn.setEnabled(False)
        self.out_card.add(self.open_btn)
        self.out_card.body.addStretch(1)

        self.prev_card = Card("Preview")
        paper = QFrame()
        paper.setObjectName("ReportPaper")
        pl = QVBoxLayout(paper)
        self.p_title = label("TA-EFL Health\nExperimental Report")
        self.p_title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #0B1D4A; border: none;")
        self.p_body = label("", wrap=True)
        self.p_body.setStyleSheet("border: none; font-size: 9pt;")
        self.p_chart = ChartCanvas(3.6, 2.0, 170)
        pl.addWidget(self.p_title)
        pl.addWidget(self.p_body)
        pl.addWidget(self.p_chart)
        self.prev_card.add(paper)
        self.shap_note = label("", "Muted", wrap=True)
        self.prev_card.add(self.shap_note)
        self.root.addWidget(hbox(self.opt_card, self.out_card, self.prev_card, stretch=[2, 2, 3]))
        self.root.addStretch(1)

    def refresh(self) -> None:
        recs = [r for r in self.ctx.store.list_current() if r.get("status") in ("completed", "stopped")]
        self.picker.populate(recs, self.ctx.selected_experiment_id)
        self.gen_btn.setEnabled(bool(recs))
        if not recs:
            self.show_message("info", "No completed experiment to report yet.")
        else:
            self.clear_message()
        self._preview()

    def _preview(self) -> None:
        if self.ctx.reference_mode:
            self.reference_banner.show()
            p = ref.REPORT_PREVIEW
            self.p_body.setText(f"Dataset: {p['dataset']}\nAggregation: {p['aggregation']}\n"
                                f"Total Rounds: {p['total_rounds']}\nTest ROC-AUC: {p['test_roc_auc']:.3f}\n"
                                f"Generated On: {p['generated_on']}")
            charts.empty(self.p_chart.reset()[0], "Reference preview")
            self.p_chart.finish()
            return
        self.reference_banner.hide()
        exp_id = self.picker.current_id()
        if not exp_id:
            self.p_body.setText("No experiment selected.")
            charts.empty(self.p_chart.reset()[0], "")
            self.p_chart.finish()
            return
        rec = self.ctx.store.load(exp_id)
        tm = rec.get("test_metrics") or {}
        self.p_body.setText(
            f"Dataset: {rec.get('dataset', {}).get('name')}\nAggregation: {rec.get('strategy_label')}\n"
            f"Total Rounds: {rec.get('rounds_completed')}\nTest ROC-AUC: {fmt(tm.get('roc_auc'))} "
            f"(test split, n = {tm.get('n')})\nGenerated On: {datetime.now():%Y-%m-%d}")
        keys = [("AUC", "roc_auc"), ("Acc", "accuracy"), ("Prec", "precision"), ("Rec", "recall"),
                ("F1", "f1")]
        charts.plot_bar_metrics(self.p_chart.reset()[0], [k for k, _ in keys],
                                [tm.get(v) for _, v in keys])
        self.p_chart.finish()
        exp = self.ctx.last_explanation
        pred = self.ctx.last_prediction
        if exp is not None and pred is not None and pred.experiment_id == exp_id:
            self.shap_note.setText("The report will include the local SHAP explanation of your "
                                   "latest prediction and global importance for this model.")
        else:
            self.shap_note.setText("The report will include global SHAP importance for this model. "
                                   "Explain a prediction made with this model to also include a "
                                   "local explanation.")

    def _generate(self) -> None:
        exp_id = self.picker.current_id()
        if not exp_id:
            return
        rec = self.ctx.store.load(exp_id)
        fmt_key = self.fmt.currentData()
        default = self.ctx.settings.storage_path / "reports" / f"report-{exp_id}.{fmt_key}"
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save report", str(default),
                                              {"pdf": "PDF (*.pdf)", "html": "HTML (*.html)",
                                               "json": "JSON (*.json)"}[fmt_key])
        if not path:
            return
        opts = ReportOptions(self.c_details.isChecked(), self.c_metrics.isChecked(),
                             self.c_trust.isChecked(), self.c_shap.isChecked(),
                             self.c_charts.isChecked(), self.c_dataset.isChecked())
        explanation = None
        pred, exp = self.ctx.last_prediction, self.ctx.last_explanation
        if exp is not None and pred is not None and pred.experiment_id == exp_id:
            explanation = explanation_to_dict(exp)
            explanation["experiment_id"] = exp_id
        artifact = None
        if opts.shap_explanation and rec.get("artifact_dir"):
            try:
                artifact = self.ctx.artifact_for(rec["config"]["dataset_key"], exp_id)
            except Exception as exc:  # noqa: BLE001
                self.show_message("warning", f"SHAP section limited: {exc}")
        self.gen_btn.setEnabled(False)
        self.gen_btn.setText("Generating…")
        run_in_background(_report_job, rec, Path(path), fmt_key, opts, explanation, artifact,
                          on_done=self._done, on_error=self._failed)

    def _done(self, path: Path) -> None:
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate Report")
        self._last_path = Path(path)
        self.open_btn.setEnabled(True)
        self.show_message("success", f"Report saved: {path}")

    def _failed(self, msg: str, _details: str) -> None:
        self.gen_btn.setEnabled(True)
        self.gen_btn.setText("Generate Report")
        self.show_message("error", f"Report generation failed: {msg}")

    def _open_folder(self) -> None:
        if self._last_path is None:
            return
        folder = self._last_path.parent
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(self._last_path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
