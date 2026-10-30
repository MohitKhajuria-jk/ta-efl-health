"""Optional personalisation endpoints (read-only research results + calibrated prediction).

Registered on the existing authenticated router by ``api.main``; nothing here trains a
model or modifies historical experiments. Personalisation runs are produced offline by
``scripts/run_personalization.py`` and read from ``storage/personalization``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from api.research import contained
from api.schemas import StrictModel
from api.security import authorized
from ml.explainability.shap_service import ExplanationError, explain_prediction, explanation_to_dict
from ml.models.artifact import ARTIFACT_FILE, ArtifactError
from ml.personalization.storage import PersonalizationStorageError, PersonalizationStore, file_sha256
from ml.prediction.service import InputValidationError, predict

REPORT = re.compile(r"^report-[A-Za-z0-9_-]{1,100}$")


class PersonalizedPredictionRequest(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")
    personalization_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_-]+$")
    hospital_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    values: dict[str, str | float | int | None]
    explain: bool = True


def run_summary(rec: dict[str, Any]) -> dict[str, Any]:
    agg = rec.get("aggregate", {})
    matched = agg.get("matched") or {}
    pick = lambda m: None if not m else {k: m.get(k) for k in ("roc_auc", "f1", "precision", "recall",
                                                                "brier", "log_loss", "ece", "n")}
    return {
        "id": rec["id"], "created": rec.get("created"), "status": rec.get("status"),
        "method": rec.get("method"), "label": rec.get("config", {}).get("label", ""),
        "dataset": rec.get("dataset", {}), "global_experiment": rec.get("global_experiment", {}),
        "data_protocol": rec.get("data_protocol", {}),
        "calibration": rec.get("config", {}).get("calibration", {}),
        "n_hospitals": agg.get("n_hospitals"), "n_personalized": agg.get("n_personalized"),
        "matched_global": pick(matched.get("global")), "matched_personalized": pick(matched.get("personalized")),
        "hospitals": [{"hospital_id": h["hospital_id"], "status": h["status"], "reason": h["reason"],
                       "n_calibration": h["n_calibration"], "n_test": h["n_test"],
                       "prevalence": h["prevalence"]} for h in rec.get("hospitals", [])],
    }


def run_detail(rec: dict[str, Any]) -> dict[str, Any]:
    out = run_summary(rec)
    out["formula"] = rec.get("formula")
    out["description"] = rec.get("description")
    out["integrity"] = rec.get("integrity", {})
    out["hospitals"] = [{k: v for k, v in h.items() if k not in ("predictions", "rows")}
                        for h in rec.get("hospitals", [])]
    out["aggregate"] = {k: v for k, v in rec.get("aggregate", {}).items()}
    return out


def register_personalization_routes(router: APIRouter, research, sessions) -> None:
    store = PersonalizationStore(research.storage)

    def load(run_id: str) -> dict[str, Any]:
        try:
            return store.load_record(run_id)
        except PersonalizationStorageError:
            raise HTTPException(404, "Personalisation run not found") from None

    @router.get("/personalization/experiments")
    def personalization_runs(global_experiment_id: str | None = None, dataset: str | None = None) -> list[dict]:
        return [run_summary(r) for r in store.list_records()
                if (not global_experiment_id or r["global_experiment"]["id"] == global_experiment_id)
                and (not dataset or r["dataset"]["key"] == dataset)]

    @router.get("/personalization/experiments/{run_id}")
    def personalization_run(run_id: str) -> dict:
        return run_detail(load(run_id))

    @router.get("/personalization/reports")
    def personalization_reports() -> list[dict]:
        folder = store.root / "reports"
        if not folder.is_dir():
            return []
        out = []
        for p in sorted(folder.glob("report-*.json"), reverse=True):
            if not REPORT.fullmatch(p.stem):
                continue
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
            except (OSError, ValueError, KeyError):
                continue
            out.append({"name": p.stem, "meta": meta})
        return out

    @router.get("/personalization/reports/{name}")
    def personalization_report(name: str) -> dict:
        if not REPORT.fullmatch(name):
            raise HTTPException(400, "Invalid report identifier")
        path = contained(store.root / "reports", store.root / "reports" / f"{name}.json")
        if not path.is_file():
            raise HTTPException(404, "Report not found")
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: data[k] for k in ("meta", "coverage", "groups", "g3_vs_g4_per_hospital_delta", "per_hospital")}

    @router.post("/personalization/predictions")
    def personalized_prediction(body: PersonalizedPredictionRequest, request: Request) -> dict:
        sessions.check_rate(("prediction", authorized(request)["username"]), 30)
        rec = load(body.personalization_id)
        if rec["global_experiment"]["id"] != body.experiment_id:
            raise HTTPException(409, "This personalisation run belongs to a different global model")
        with research.prediction_lock:
            try:
                artifact = research.artifact(body.experiment_id)
                art_dir = contained(research.storage / "models",
                                    research.storage / rec["global_experiment"]["artifact_dir"])
                calibrator, prov = store.load_calibrator(
                    body.personalization_id, body.hospital_id, expected_global_id=body.experiment_id,
                    expected_artifact_sha256=file_sha256(art_dir / ARTIFACT_FILE))
                result = predict(artifact, body.values)
            except InputValidationError as exc:
                raise HTTPException(422, exc.errors) from None
            except ArtifactError:
                raise HTTPException(409, "The saved model is unavailable or incompatible.") from None
            except PersonalizationStorageError as exc:
                raise HTTPException(409, f"Personalisation unavailable for this hospital: {exc}") from None
            z = float(result.decision_value)
            p_pers = float(calibrator.predict_proba([z])[0])
            explanation, error = None, None
            if body.explain:
                try:
                    explanation = explanation_to_dict(explain_prediction(artifact, result, explained_class=1))
                except ExplanationError:
                    error = "SHAP explanation is unavailable for this prediction. The prediction remains valid."
            cls = int(p_pers >= result.threshold)
            hosp = next(h for h in rec["hospitals"] if h["hospital_id"] == body.hospital_id)
            return {
                "experiment_id": result.experiment_id, "personalization_id": rec["id"],
                "hospital_id": body.hospital_id, "dataset_key": result.dataset_key,
                "positive_label": artifact.spec.positive_label, "threshold": result.threshold,
                "global": {"log_odds": z, "probability_positive": result.probability_positive,
                           "predicted_class": result.predicted_class, "predicted_label": result.predicted_label},
                "personalized": {"log_odds": float(calibrator.transform_logits([z])[0]),
                                 "probability_positive": p_pers, "predicted_class": cls,
                                 "predicted_label": artifact.class_labels[cls]},
                "calibration": {"a": calibrator.a, "b": calibrator.b,
                                "log_odds_adjustment": float(calibrator.adjustment([z])[0]),
                                "probability_change": p_pers - result.probability_positive,
                                "n_calibration": calibrator.n_samples, "regularization": calibrator.regularization,
                                "method": calibrator.method,
                                "hospital_local_test": {"global": {k: hosp["global"][k] for k in ("brier", "ece", "roc_auc", "f1")},
                                                        "personalized": {k: hosp["personalized"][k] for k in ("brier", "ece", "roc_auc", "f1")}}},
                "explanation": explanation, "explanation_error": error,
                "explanation_scope": ("SHAP values explain the global model's log-odds only. The hospital "
                                      "calibration is a separate monotone probability adjustment applied "
                                      "afterwards; it is not a feature attribution."),
                "notes": result.notes + ["Calibrated on this simulated hospital's local calibration rows; not "
                                         "a clinically validated risk."],
            }
