"""Healthcare risk prediction with the selected dataset's trained federated model."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from app.services import reference_data as ref
from app.services.context import AppContext
from app.ui import icons, theme
from app.ui.widgets.common import Card, Page, button, experiment_label, hbox, label
from app.ui.workers import run_in_background
from ml.datasets.registry import DatasetRegistry
from ml.datasets.schema import DatasetSpec, FeatureSpec
from ml.explainability.shap_service import explain_prediction
from ml.models.artifact import ArtifactError
from ml.prediction.service import InputValidationError, PredictionResult, predict, validate_inputs

TITLES = {"heart": ("Heart Disease Risk Prediction", "heart disease"),
          "diabetes": ("Diabetes Risk Prediction", "diabetes"),
          "breast_cancer": ("Breast Cancer Malignancy Prediction", "malignancy"),
          "framingham": ("10-Year CHD Risk Prediction", "coronary heart disease"),
          "stroke": ("Stroke Risk Prediction", "stroke")}


class ResultBox(QFrame):
    """Tinted result panel: icon, headline, probability."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ResultBox")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        self.icon = QLabel()
        col = QVBoxLayout()
        col.setSpacing(2)
        self.headline = QLabel("No prediction yet")
        self.prob = QLabel("")
        self.sub = QLabel("")
        self.sub.setWordWrap(True)
        col.addWidget(self.headline)
        col.addWidget(self.prob)
        col.addWidget(self.sub)
        lay.addWidget(self.icon)
        lay.addSpacing(10)
        lay.addLayout(col, 1)
        self.set_state("empty")

    def set_state(self, kind: str, headline: str = "No prediction yet", prob: str = "",
                  sub: str = "") -> None:
        bg, fg = {"high": ("#FDECEC", "#DC2626"), "low": ("#ECFDF3", "#15803D"),
                  "empty": ("#F1F5F9", "#64748B"), "reference": ("#FDECEC", "#DC2626")}[kind]
        self.setProperty("tone", kind)
        self.style().unpolish(self)
        self.style().polish(self)
        self.icon.setPixmap(icons.pixmap("heart", fg, 52))
        self.headline.setText(headline)
        self.headline.setStyleSheet(f"color: {fg}; font-size: 20pt; font-weight: 700; background: transparent;")
        self.prob.setText(prob)
        self.prob.setStyleSheet("color: #334155; font-size: 11pt; background: transparent;")
        self.sub.setText(sub)
        self.sub.setStyleSheet("color: #475569; font-size: 9pt; background: transparent;")


class PredictionPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Heart Disease Risk Prediction",
                         "Enter patient details to predict the risk of heart disease")
        self.ctx = ctx
        self._navigate = None
        self._fields: dict[str, QWidget] = {}
        self._spec: DatasetSpec | None = None
        self._busy = False

        self.dataset = QComboBox()
        self.dataset.currentIndexChanged.connect(self._dataset_changed)
        self.model = QComboBox()
        self.model.setMinimumWidth(320)
        self.model.currentIndexChanged.connect(lambda _i: self._model_changed())
        self.header.add_right(label("Dataset"))
        self.header.add_right(self.dataset)
        self.header.add_right(label("Model"))
        self.header.add_right(self.model)

        self.form_card = Card("Patient Details")
        self.form_host = QWidget()
        self.grid = QGridLayout(self.form_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(8)
        self.form_card.add(self.form_host)
        self.form_errors = label("", wrap=True)
        self.form_errors.setStyleSheet(f"color: {theme.RED};")
        self.form_card.add(self.form_errors)
        self.predict_btn = button("Predict Risk", "Primary", on_click=self._predict)
        self.predict_btn.setMinimumHeight(40)
        self.clear_btn = button("Clear", None, on_click=self._clear)
        self.clear_btn.setMinimumHeight(40)
        self.form_card.add(hbox(self.predict_btn, self.clear_btn, stretch=[3, 1]))

        self.result_card = Card("Prediction Result")
        self.result = ResultBox()
        self.result_card.add(self.result)
        self.interp_title = label("Risk Interpretation", "CardTitle")
        self.result_card.add(self.interp_title)
        self.interp = label("", wrap=True)
        self.interp.setTextFormat(Qt.TextFormat.RichText)
        self.result_card.add(self.interp)
        self.model_info = label("", "Muted", wrap=True)
        self.result_card.add(self.model_info)
        self.explain_btn = button("View Explanation  →", "Primary", on_click=self._explain)
        self.explain_btn.setMinimumHeight(40)
        self.explain_btn.setEnabled(False)
        self.result_card.add(self.explain_btn)
        self.no_model_btn = button("Go to Federated Training", None, "training",
                                   lambda: self._navigate and self._navigate("training"))
        self.no_model_btn.hide()
        self.result_card.add(self.no_model_btn)
        self.result_card.body.addStretch(1)
        self.root.addWidget(hbox(self.form_card, self.result_card, stretch=[1, 1]))
        self.root.addWidget(label(
            "Research tool only – not a medical diagnostic system. The output is a model "
            "classification with the model's estimated probability, not a clinically calibrated "
            "risk.", "Muted", wrap=True))
        self.root.addStretch(1)

    def set_navigator(self, nav) -> None:
        self._navigate = nav

    # ------------------------------------------------------------ population
    def refresh(self) -> None:
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for spec in DatasetRegistry.keys():
            self.dataset.addItem(DatasetRegistry.get(spec).name, spec)
        self.dataset.setCurrentIndex(max(0, self.dataset.findData(self.ctx.dataset_key)))
        self.dataset.blockSignals(False)
        self._build_form(DatasetRegistry.get(self.ctx.dataset_key))
        self._fill_models()
        if self.ctx.reference_mode and self.ctx.dataset_key == "heart":
            self.reference_banner.show()
            self._apply_values(ref.PREDICTION_INPUTS)
            r = ref.PREDICTION_RESULT
            self.result.set_state("reference", r["headline"], f"Probability: {r['probability']:.2f}",
                                  "Reference value from the design image – press Predict Risk "
                                  "to run the trained model.")
            self.interp.setText("Reference mode: no model was run. Press <b>Predict Risk</b> to "
                                "obtain a genuine prediction for these inputs (complete the "
                                "remaining required fields first).")
            self.explain_btn.setEnabled(False)
        else:
            self.reference_banner.setVisible(False)
            self._show_prediction(self.ctx.last_prediction)

    def _dataset_changed(self, _i: int) -> None:
        key = self.dataset.currentData()
        if key and key != self.ctx.dataset_key:
            self.ctx.set_dataset(key)
            self.refresh()

    def _fill_models(self) -> None:
        recs = [r for r in self.ctx.store.list_current(self.ctx.dataset_key)
                if r.get("artifact_dir") and r.get("status") in ("completed", "stopped")]
        self.model.blockSignals(True)
        self.model.clear()
        for r in recs:
            self.model.addItem(experiment_label(r), r["id"])
        sel = self.ctx.selected_experiment_id
        idx = self.model.findData(sel) if sel else -1
        self.model.setCurrentIndex(idx if idx >= 0 else (0 if recs else -1))
        self.model.blockSignals(False)
        spec = DatasetRegistry.get(self.ctx.dataset_key)
        av = self.ctx.registry.availability(spec.key)
        if not recs:
            self.show_message("warning", f"No trained model for '{spec.name}'. "
                              + ("Run federated training on this dataset first." if av.available
                                 else av.message))
            self.predict_btn.setEnabled(False)
            self.no_model_btn.show()
        else:
            self.clear_message()
            self.predict_btn.setEnabled(True)
            self.no_model_btn.hide()

    def _model_changed(self) -> None:
        self.ctx.last_prediction = None
        self.ctx.last_explanation = None
        self._show_prediction(None)

    def _build_form(self, spec: DatasetSpec) -> None:
        if self._spec is not None and self._spec.key == spec.key:
            return
        self._spec = spec
        title, noun = TITLES.get(spec.key, (f"{spec.name} Prediction", spec.positive_label.lower()))
        self.header.title.setText(title)
        self.header.subtitle.setText(f"Enter patient details to predict the risk of {noun}")
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._fields = {}
        cols = 2 if len(spec.features) <= 12 else 3
        for i, f in enumerate(spec.features):
            r, c = divmod(i, cols)
            lab = QLabel(f.label + (f" ({f.unit})" if f.unit else ""))
            lab.setToolTip(f.help or f.name)
            w = self._widget_for(f)
            self.grid.addWidget(lab, r * 2, c)
            self.grid.addWidget(w, r * 2 + 1, c)
            self._fields[f.name] = w

    @staticmethod
    def _widget_for(f: FeatureSpec) -> QWidget:
        if f.kind in ("categorical", "binary") and (f.categories or f.category_labels):
            cb = QComboBox()
            cb.setPlaceholderText("Select…")
            if f.kind == "categorical":
                for code, lab in zip(f.categories, f.category_labels or f.categories):
                    cb.addItem(lab, code)
            else:
                for i, lab in enumerate(f.category_labels):
                    cb.addItem(lab, i)
            cb.setCurrentIndex(-1)
            return cb
        e = QLineEdit()
        lo = "" if f.min_value is None else f"{f.min_value:g}"
        hi = "" if f.max_value is None else f"{f.max_value:g}"
        e.setPlaceholderText(f"{lo} – {hi}" if hi else (f"≥ {lo}" if lo else ""))
        return e

    # ------------------------------------------------------------ values
    def _values(self) -> dict:
        out = {}
        for name, w in self._fields.items():
            if isinstance(w, QComboBox):
                out[name] = w.currentData() if w.currentIndex() >= 0 else None
            else:
                out[name] = w.text().strip()
        return out

    def _apply_values(self, values: dict) -> None:
        for name, v in values.items():
            w = self._fields.get(name)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(w.findData(v))
            elif isinstance(w, QLineEdit):
                w.setText(f"{v:g}" if isinstance(v, float) else str(v))

    def _mark(self, errors: dict[str, str]) -> None:
        for name, w in self._fields.items():
            w.setProperty("invalid", name in errors)
            w.style().unpolish(w)
            w.style().polish(w)
        self.form_errors.setText("\n".join(errors.values()))

    def _clear(self) -> None:
        for w in self._fields.values():
            if isinstance(w, QComboBox):
                w.setCurrentIndex(-1)
            else:
                w.clear()
        self._mark({})
        self.ctx.last_prediction = None
        self.ctx.last_explanation = None
        self._show_prediction(None)

    # ------------------------------------------------------------ actions
    def _predict(self) -> None:
        if self._busy or self._spec is None:
            return
        values = self._values()
        try:
            validate_inputs(self._spec, values)
        except InputValidationError as exc:
            self._mark(exc.errors)
            return
        self._mark({})
        try:
            artifact = self.ctx.artifact_for(self._spec.key, self.model.currentData())
        except (ArtifactError, FileNotFoundError) as exc:
            self.show_message("error", str(exc))
            return
        self._busy = True
        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("Predicting…")
        run_in_background(predict, artifact, values, on_done=self._predicted, on_error=self._failed)

    def _predicted(self, result: PredictionResult) -> None:
        self._busy = False
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict Risk")
        self.ctx.last_prediction = result
        self.ctx.last_explanation = None
        self.reference_banner.hide()
        self._show_prediction(result)
        self.ctx.emit("prediction_changed")

    def _failed(self, message: str, _details: str) -> None:
        self._busy = False
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Predict Risk")
        self.show_message("error", f"Prediction failed: {message}")

    def _show_prediction(self, r: PredictionResult | None) -> None:
        if r is None or (self._spec and r.dataset_key != self._spec.key):
            self.result.set_state("empty")
            self.interp.setText("Enter all patient details and press <b>Predict Risk</b>.")
            self.model_info.setText("")
            self.explain_btn.setEnabled(False)
            return
        pos = r.class_labels[1]
        self.result.set_state("high" if r.predicted_class == 1 else "low", r.headline,
                              f"Probability: {r.probability_positive:.2f}",
                              f"Model probability of '{pos}'")
        self.interp.setText(
            f"The model classifies this patient as <b>{r.predicted_label}</b>: its estimated "
            f"probability of '{pos}' is {r.probability_positive:.2f}, compared with the decision "
            f"threshold {r.threshold:.2f} fixed before training. The factors that drove this "
            f"output are shown in the explanation page.<br><i>{r.notes[0]}</i>")
        self.model_info.setText(f"Dataset: {r.dataset_name} · Model: {r.model_description} · "
                                f"Experiment {r.experiment_id}")
        self.explain_btn.setEnabled(True)

    def _explain(self) -> None:
        r = self.ctx.last_prediction
        if r is None:
            return
        try:
            artifact = self.ctx.artifact_for(r.dataset_key, r.experiment_id)
        except (ArtifactError, FileNotFoundError) as exc:
            self.show_message("error", str(exc))
            return
        self.explain_btn.setEnabled(False)
        self.explain_btn.setText("Computing SHAP…")

        def done(exp) -> None:
            self.explain_btn.setEnabled(True)
            self.explain_btn.setText("View Explanation  →")
            self.ctx.last_explanation = exp
            self.ctx.last_explanation_error = None
            if self._navigate:
                self._navigate("explain")

        def fail(msg: str, _details: str) -> None:
            self.explain_btn.setEnabled(True)
            self.explain_btn.setText("View Explanation  →")
            self.ctx.last_explanation = None
            self.ctx.last_explanation_error = msg
            if self._navigate:
                self._navigate("explain")

        run_in_background(explain_prediction, artifact, r, on_done=done, on_error=fail)
