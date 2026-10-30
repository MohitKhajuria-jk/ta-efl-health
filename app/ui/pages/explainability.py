"""SHAP explanation of the latest prediction, plus global feature importance."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.ui import icons, theme
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import Card, Page, button, fill_table, fmt, hbox, label, make_table
from app.ui.workers import run_in_background
from ml.explainability.shap_service import (
    LocalExplanation,
    explain_prediction,
    feature_statements,
    global_importance,
)


class StatementList(QWidget):
    """Arrow-prefixed explanation sentences (red ↑ increases, blue ↓ decreases)."""

    def __init__(self) -> None:
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(8)

    def set(self, items: list[tuple[int, str]]) -> None:
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for direction, text in items:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            ic = QLabel()
            if direction > 0:
                ic.setPixmap(icons.pixmap("up", theme.RED, 16))
            elif direction < 0:
                ic.setPixmap(icons.pixmap("down", "#2563EB", 16))
            else:
                ic.setPixmap(icons.pixmap("about", theme.MUTED, 16))
            t = label(text, wrap=True)
            rl.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
            rl.addWidget(t, 1)
            self.lay.addWidget(row)


class TipBox(QFrame):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.setObjectName("TipBox")
        lay = QHBoxLayout(self)
        ic = QLabel()
        ic.setPixmap(icons.pixmap("bulb", "#F59E0B", 26))
        ic.setStyleSheet("border: none; background: transparent;")
        self.text = label(text, wrap=True)
        self.text.setStyleSheet("border: none; background: transparent; color: #1E3A8A;")
        lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.text, 1)


class ExplainabilityPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("SHAP Explanation for Prediction",
                         "Understand feature contributions using SHAP")
        self.ctx = ctx
        self._navigate = None
        self._selected: str | None = None
        self._importance_for: str | None = None

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # ---- tab 1: feature contribution -----------------------------------
        t1 = QWidget()
        l1 = QHBoxLayout(t1)
        l1.setContentsMargins(0, 0, 0, 0)
        l1.setSpacing(12)
        self.chart_card = Card("Feature Contribution (SHAP Values)")
        self.explained = QComboBox()
        self.explained.currentIndexChanged.connect(lambda _i: self._recompute_for_class())
        self.chart_card.header.addWidget(label("Explain class", "Muted"))
        self.chart_card.header.addWidget(self.explained)
        self.chart = ChartCanvas(5.0, 3.6, 320)
        self.chart.on_pick(self._select_feature)
        self.chart_card.add(self.chart, 1)
        self.feature_pick = QComboBox()
        self.feature_pick.currentIndexChanged.connect(
            lambda _i: self._select_feature(self.feature_pick.currentData()))
        self.feature_detail = label("Click a bar or choose a feature to see its value and "
                                    "contribution.", "Muted", wrap=True)
        self.chart_card.add(hbox(label("Feature"), self.feature_pick, stretch=[0, 1]))
        self.chart_card.add(self.feature_detail)
        self.text_card = Card("Human-Readable Explanation")
        self.statements = StatementList()
        self.text_card.add(self.statements)
        self.summary = label("", wrap=True)
        self.text_card.add(self.summary)
        self.tip = TipBox("These features are the most influential for this prediction. SHAP "
                          "explains model behaviour, not medical causation.")
        self.text_card.body.addStretch(1)
        self.text_card.add(self.tip)
        l1.addWidget(self.chart_card, 11)
        l1.addWidget(self.text_card, 9)
        self.tabs.addTab(t1, "Feature Contribution")
        # ---- tab 2: patient summary ----------------------------------------
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        l2.setContentsMargins(0, 0, 0, 0)
        self.patient_card = Card("Prediction Summary")
        self.patient_info = label("", wrap=True)
        self.patient_info.setTextFormat(Qt.TextFormat.RichText)
        self.patient_card.add(self.patient_info)
        self.patient_table = make_table(["Feature", "Patient value", "SHAP (log-odds)", "Direction"])
        self.patient_table.setMinimumHeight(320)
        self.patient_card.add(self.patient_table)
        l2.addWidget(self.patient_card)
        self.tabs.addTab(t2, "Patient Summary")
        # ---- tab 3: global importance --------------------------------------
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        l3.setContentsMargins(0, 0, 0, 0)
        self.global_card = Card("Global Feature Importance")
        self.global_btn = button("Compute for current model", "Primary", on_click=self._compute_global)
        self.global_card.header.addWidget(self.global_btn)
        self.global_chart = ChartCanvas(6, 3.8, 340)
        self.global_card.add(self.global_chart)
        self.global_note = label("Global importance summarises the model across the server-"
                                 "validation patients (mean |SHAP|). It is different from the "
                                 "explanation of one patient's prediction.", "Muted", wrap=True)
        self.global_card.add(self.global_note)
        l3.addWidget(self.global_card)
        self.tabs.addTab(t3, "Global Importance")
        self.root.addWidget(self.tabs, 1)

        self.empty_btn = button("Go to Risk Prediction", "Primary", "prediction",
                                lambda: self._navigate and self._navigate("prediction"))
        self.root.addWidget(self.empty_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.root.addStretch(1)

    def set_navigator(self, nav) -> None:
        self._navigate = nav

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        exp = self.ctx.last_explanation
        pred = self.ctx.last_prediction
        if exp is None and self.ctx.reference_mode:
            self._render_reference()
            return
        self.reference_banner.hide()
        if pred is None:
            self.show_message("info", "No prediction to explain yet. Make a prediction on the "
                              "Risk Prediction page, then press 'View Explanation'.")
            self.empty_btn.show()
            self._clear()
            return
        self.empty_btn.hide()
        err = getattr(self.ctx, "last_explanation_error", None)
        if exp is None:
            self._clear()
            self._fill_summary(None)
            if err:
                self.show_message("error", f"SHAP explanation could not be computed: {err}. The "
                                  "prediction itself is unchanged; no explanation values are shown.")
            else:
                self.show_message("info", "Press 'View Explanation' on the prediction page to "
                                  "compute the SHAP explanation for the current prediction.")
            return
        self.clear_message()
        self._fill_classes(pred)
        self._render(exp)

    def _clear(self) -> None:
        charts.empty(self.chart.reset()[0], "No explanation")
        self.chart.finish()
        self.statements.set([])
        self.summary.setText("")
        self.feature_pick.clear()
        self.patient_table.setRowCount(0)
        self.patient_info.setText("")

    def _fill_classes(self, pred) -> None:
        self.explained.blockSignals(True)
        self.explained.clear()
        for cls in (1, 0):
            self.explained.addItem(pred.class_labels[cls], cls)
        cur = self.ctx.last_explanation.explained_class if self.ctx.last_explanation else pred.predicted_class
        self.explained.setCurrentIndex(self.explained.findData(cur))
        self.explained.blockSignals(False)

    def _render(self, exp: LocalExplanation) -> None:
        items = [(c.label, c.shap_value) for c in exp.contributions]
        charts.plot_contributions(self.chart.reset()[0], items, self._selected,
                                  xlabel=f"SHAP value (log-odds toward '{exp.explained_label}')")
        self.chart.finish()
        self.statements.set(feature_statements(exp, k=5))
        self.summary.setText(exp.summary + "\n\n" + " ".join(exp.statements[-1:]))
        self.feature_pick.blockSignals(True)
        self.feature_pick.clear()
        for c in exp.contributions:
            self.feature_pick.addItem(c.label, c.label)
        self.feature_pick.setCurrentIndex(-1)
        self.feature_pick.blockSignals(False)
        self._fill_summary(exp)

    def _fill_summary(self, exp: LocalExplanation | None) -> None:
        pred = self.ctx.last_prediction
        if pred is None:
            return
        txt = (f"<b>Predicted class:</b> {pred.predicted_label} &nbsp; "
               f"<b>Model probability of '{pred.class_labels[1]}':</b> {pred.probability_positive:.3f} "
               f"&nbsp; <b>Threshold:</b> {pred.threshold:.2f}<br>"
               f"<b>Model:</b> {pred.model_description} (experiment {pred.experiment_id})")
        if exp is not None:
            txt += (f"<br><b>Baseline log-odds:</b> {exp.base_value:+.3f} (probability "
                    f"{exp.base_probability:.3f}) &nbsp; <b>This patient:</b> {exp.output_value:+.3f} "
                    f"(probability {exp.output_probability:.3f}) – explained class "
                    f"'{exp.explained_label}', background n = {exp.n_background}")
        self.patient_info.setText(txt)
        rows = []
        if exp is not None:
            for c in exp.contributions:
                rows.append([c.label, c.value_display, f"{c.shap_value:+.4f}",
                             "↑ toward" if c.shap_value > 0 else ("↓ away" if c.shap_value < 0 else "–")])
        else:
            spec = self.ctx.registry.get(pred.dataset_key)
            for f in spec.features:
                rows.append([f.label, f.display_value(pred.inputs.get(f.name)), "—", "—"])
        fill_table(self.patient_table, rows)

    def _select_feature(self, name: str | None) -> None:
        exp = self.ctx.last_explanation
        if not name or exp is None:
            return
        self._selected = name
        for c in exp.contributions:
            if c.label == name:
                direction = "toward" if c.shap_value > 0 else "away from"
                self.feature_detail.setText(
                    f"{c.label} = {c.value_display}: SHAP {c.shap_value:+.3f} log-odds {direction} "
                    f"'{exp.explained_label}', {c.reference_display}.")
                break
        idx = self.feature_pick.findData(name)
        if idx != self.feature_pick.currentIndex():
            self.feature_pick.blockSignals(True)
            self.feature_pick.setCurrentIndex(idx)
            self.feature_pick.blockSignals(False)
        charts.plot_contributions(self.chart.reset()[0], [(c.label, c.shap_value) for c in exp.contributions],
                                  name, xlabel=f"SHAP value (log-odds toward '{exp.explained_label}')")
        self.chart.finish()

    def _recompute_for_class(self) -> None:
        pred = self.ctx.last_prediction
        cls = self.explained.currentData()
        if pred is None or cls is None:
            return
        try:
            artifact = self.ctx.artifact_for(pred.dataset_key, pred.experiment_id)
        except Exception as exc:  # noqa: BLE001
            self.show_message("error", str(exc))
            return

        def done(exp: LocalExplanation) -> None:
            self.ctx.last_explanation = exp
            self._render(exp)

        run_in_background(explain_prediction, artifact, pred, cls, on_done=done,
                          on_error=lambda m, _d: self.show_message("error", f"SHAP failed: {m}"))

    def _compute_global(self) -> None:
        pred = self.ctx.last_prediction
        try:
            artifact = (self.ctx.artifact_for(pred.dataset_key, pred.experiment_id) if pred
                        else self.ctx.artifact_for(self.ctx.dataset_key, None))
        except Exception as exc:  # noqa: BLE001
            charts.empty(self.global_chart.reset()[0], str(exc))
            self.global_chart.finish()
            return
        self.global_btn.setEnabled(False)

        def done(g) -> None:
            self.global_btn.setEnabled(True)
            charts.plot_contributions(self.global_chart.reset()[0], g.features,
                                      xlabel="mean |SHAP| (log-odds)", max_items=15)
            self.global_chart.ax.set_title(f"{artifact.spec.name} – {g.split} split (n = {g.n}), "
                                           f"experiment {artifact.experiment_id}", fontsize=8, loc="left")
            self.global_chart.finish()

        def fail(m: str, _d: str) -> None:
            self.global_btn.setEnabled(True)
            charts.empty(self.global_chart.reset()[0], f"SHAP failed: {m}")
            self.global_chart.finish()

        run_in_background(global_importance, artifact, on_done=done, on_error=fail)

    def _render_reference(self) -> None:
        self.reference_banner.show()
        self.clear_message()
        self.empty_btn.hide()
        charts.plot_contributions(self.chart.reset()[0], ref.SHAP_VALUES, xlabel="SHAP Value")
        self.chart.finish()
        self.statements.set([(1 if "increased" in s or "contributed" in s else -1, s)
                             for s in ref.SHAP_STATEMENTS])
        self.summary.setText("Reference text from the design image – not computed by this software.")
        self.patient_table.setRowCount(0)
        self.patient_info.setText("")
