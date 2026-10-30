"""FastAPI entry point. Run one worker; this application is a local research workstation."""
from __future__ import annotations
import json
import logging
import os
import platform
import secrets
import time
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.middleware.trustedhost import TrustedHostMiddleware
from filelock import FileLock
from app.configuration.settings import SettingsManager, PROJECT_ROOT
from app.services.auth import AuthError, LocalAuth
from app.services.reports import ReportOptions, generate_report
from experiments.store import atomic_write_text, is_current
from ml.models.artifact import ArtifactError
from ml.prediction.service import InputValidationError, predict
from ml.explainability.shap_service import ExplanationError, explain_prediction, explanation_to_dict
from federated.clients.attacks import ATTACK_DESCRIPTIONS
from federated.strategies.strategies import STRATEGY_LABELS
from api.jobs import JobManager
from api.research import Research, safe_id, protocol
from api.personalization import register_personalization_routes
from api.schemas import (Credentials, SessionInfo, TrainingRequest, PredictionRequest,
                         PredictionResponse, JobView, ExperimentPage, ComparisonRequest, SettingsUpdate)
from api.security import COOKIE, Sessions, authorized
from api.logging import configure_web_logging

log = logging.getLogger("ta_efl.web")


def create_app(settings_path: Path | None = None):
    manager = SettingsManager(settings_path or Path(os.environ.get("TA_EFL_SETTINGS", PROJECT_ROOT / "config/settings.json")))
    settings = manager.load()
    research = Research(settings)
    sessions = Sessions()
    auth = LocalAuth(settings.storage_path)
    auth_lock = threading.Lock()
    web_defaults = settings.storage_path / "web" / "defaults.json"

    @asynccontextmanager
    async def lifespan(app):
        configure_web_logging()
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        process_lock = FileLock(settings.storage_path / ".web-api.lock", timeout=0)
        process_lock.acquire()
        app.state.jobs = JobManager(research.data, research.storage)
        try:
            yield
        finally:
            app.state.jobs.close()
            process_lock.release()

    app = FastAPI(title="TA-EFL Health Research API", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sessions = sessions
    app.state.research = research
    hosts = os.environ.get("TA_EFL_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def boundaries(request: Request, call_next):
        started = time.monotonic()
        request_id = secrets.token_hex(8)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and (urlsplit(origin).netloc != request.headers.get("host") or urlsplit(origin).scheme not in {"http", "https"}):
                return JSONResponse({"detail":"Cross-origin writes are not allowed"}, status_code=403)
            length = request.headers.get("content-length", "0")
            if not length.isdigit() or int(length) > 65536:
                return JSONResponse({"detail":"Request body is too large"}, status_code=413)
            size, chunks = 0, []
            async for chunk in request.stream():
                size += len(chunk)
                if size > 65536:
                    return JSONResponse({"detail":"Request body is too large"}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/api"):
            log.info("api_request", extra={"request_id":request_id, "route":request.url.path,
                     "status":response.status_code, "duration_ms":round((time.monotonic()-started)*1000,1)})
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo passwords or patient values from Pydantic's input/context fields.
        return JSONResponse({"detail":[{"loc":e["loc"], "msg":e["msg"], "type":e["type"]} for e in exc.errors()]}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        log.error("Unhandled API error on %s (%s)", request.url.path, type(exc).__name__)
        return JSONResponse({"detail":"The operation could not be completed. Check local server diagnostics."}, status_code=500)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status":"ok", "engine":"TA-EFL Python", "mode":"in-process simulation"}

    @app.get("/api/auth/session", response_model=SessionInfo)
    def session(request: Request):
        value = sessions.get(request)
        return dict(authenticated=bool(value), registration_open=not auth.has_users(),
                    username=value["username"] if value else None, csrf=value["csrf"] if value else None)

    def login_response(request, response, username):
        with sessions.lock:
            sessions.sessions.pop(request.cookies.get(COOKIE, ""), None)
        token, value = sessions.create(username)
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=8*3600,
                            secure=os.environ.get("TA_EFL_SECURE_COOKIES") == "1", path="/")
        return dict(authenticated=True, username=username, csrf=value["csrf"], registration_open=False)

    @app.post("/api/auth/login", response_model=SessionInfo)
    def login(body: Credentials, request: Request, response: Response):
        sessions.check_rate(("login", request.client.host if request.client else "local"))
        try:
            valid = auth.verify(body.username, body.password)
        except AuthError:
            raise HTTPException(503, "Account store unavailable; contact the workstation owner") from None
        if not valid:
            raise HTTPException(401, "Invalid username or password")
        return login_response(request, response, body.username)

    @app.post("/api/auth/register", response_model=SessionInfo)
    def register(body: Credentials, request: Request, response: Response):
        sessions.check_rate(("register", request.client.host if request.client else "local"), 5)
        with auth_lock:
            if auth.has_users():
                authorized(request)
            try:
                auth.create_user(body.username, body.password)
            except AuthError as exc:
                raise HTTPException(409, str(exc)) from None
        return login_response(request, response, body.username)

    protected = APIRouter(prefix="/api", dependencies=[Depends(authorized)])

    @protected.post("/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        with sessions.lock:
            sessions.sessions.pop(request.cookies.get(COOKIE, ""), None)
        response.delete_cookie(COOKIE, path="/")
        return {"ok":True}

    @protected.get("/schema")
    def schema() -> dict:
        return app.openapi()

    @protected.get("/datasets")
    def datasets() -> list[dict]:
        return research.datasets()

    @protected.get("/dashboard")
    def dashboard(request: Request) -> dict:
        records, data = research.records(), research.datasets()
        current = [r for r in records if r["current"] and r["status"] in {"completed", "stopped"}]
        jobs = request.app.state.jobs.list()
        return dict(total_experiments=len(records), total_models=sum(r["has_model"] for r in records),
                    validated_datasets=sum(d["status"].startswith("valid") for d in data),
                    latest=current[0] if current else None, recent=records[:6], datasets=data,
                    active_job=next((j for j in jobs if j["status"] == "running"), None))

    @protected.get("/experiments", response_model=ExperimentPage)
    def experiments(dataset: str | None = None, strategy: str | None = None,
                    attack: str | None = None, status: str | None = None, q: str = Query("", max_length=120),
                    sort: Literal["newest", "oldest", "auc"] = "newest",
                    page: int = Query(1, ge=1), page_size: int = Query(15, ge=1, le=100)):
        records = [r for r in research.records() if
                   (not dataset or r["config"].get("dataset_key") == dataset) and
                   (not strategy or r["config"].get("strategy") == strategy) and
                   (not attack or r["config"].get("attack_type") == attack) and
                   (not status or r["status"] == status) and
                   (not q or q.lower() in (r["id"] + r["config"].get("label", "")).lower())]
        if sort == "oldest": records.reverse()
        if sort == "auc": records.sort(key=lambda r:r["test_metrics"].get("roc_auc") or -1, reverse=True)
        start = (page - 1) * page_size
        return dict(items=records[start:start+page_size], total=len(records), page=page, page_size=page_size)

    @protected.post("/experiments/compare")
    def compare(body: ComparisonRequest) -> dict:
        if len(set(body.ids)) != len(body.ids):
            raise HTTPException(422, "Choose distinct experiments")
        records = [research.record(x) for x in body.ids]
        baseline = protocol(records[0])
        if any(protocol(r) != baseline for r in records[1:]):
            raise HTTPException(409, "Comparison requires the same dataset version, seed, test split, pipeline and threshold. Adjust the selected runs.")
        return {"items":[summary_for_compare(r) for r in records], "protocol":baseline,
                "note":"Saved held-out test results on the same split. These runs were not independently rerun."}

    @protected.get("/experiments/{exp_id}")
    def experiment(exp_id: str) -> dict:
        rec = research.record(exp_id)
        rec["origin"] = rec.get("origin", "historical")
        rec["current"] = is_current(rec)
        rec.pop("traceback", None)
        return rec

    @protected.get("/models")
    def models(dataset: str | None = None) -> list[dict]:
        return [r for r in research.records() if r["current"] and r["has_model"] and
                r["status"] in {"completed", "stopped"} and (not dataset or r["config"]["dataset_key"] == dataset)]

    @protected.post("/predictions", response_model=PredictionResponse)
    def prediction(body: PredictionRequest, request: Request):
        sessions.check_rate(("prediction", authorized(request)["username"]), 30)
        with research.prediction_lock:
            try:
                artifact = research.artifact(body.experiment_id)
                result = predict(artifact, body.values)
            except InputValidationError as exc:
                raise HTTPException(422, exc.errors) from None
            except ArtifactError:
                raise HTTPException(409, "The saved model is unavailable or incompatible. Retrain this experiment.") from None
            explanation, error = None, None
            if body.explain:
                try:
                    value = explain_prediction(artifact, result, explained_class=1)
                    explanation = explanation_to_dict(value)
                except ExplanationError:
                    error = "SHAP explanation is unavailable for this prediction. The prediction remains valid."
            return dict(experiment_id=result.experiment_id, dataset_key=result.dataset_key,
                        probability_positive=result.probability_positive, threshold=result.threshold,
                        predicted_class=result.predicted_class, predicted_label=result.predicted_label,
                        positive_label=artifact.spec.positive_label, notes=result.notes,
                        explanation=explanation, explanation_error=error)

    @protected.get("/federation", response_model=list[JobView])
    def jobs(request: Request):
        return request.app.state.jobs.list()

    @protected.post("/federation", response_model=JobView, status_code=202)
    def start(body: TrainingRequest, request: Request):
        sessions.check_rate(("training", authorized(request)["username"]), 10)
        data = next(d for d in research.datasets() if d["key"] == body.dataset_key)
        if not data["status"].startswith("valid"):
            raise HTTPException(409, "Dataset must be available and valid before training")
        try:
            return request.app.state.jobs.start(body.model_dump())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @protected.get("/federation/{job_id}", response_model=JobView)
    def job(job_id: str, request: Request):
        try:
            return request.app.state.jobs.get(safe_id(job_id))
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @protected.post("/federation/{job_id}/cancel", response_model=JobView)
    def cancel(job_id: str, request: Request):
        try:
            return request.app.state.jobs.cancel(safe_id(job_id))
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @protected.get("/reports/{exp_id}")
    def report(exp_id: str, request: Request, format: Literal["json", "html", "pdf"] = "pdf"):
        sessions.check_rate(("report", authorized(request)["username"]), 10)
        record = research.record(exp_id)
        if not is_current(record) or record.get("status") not in {"completed", "stopped"}:
            raise HTTPException(409, "Reports require a current completed or stopped experiment")
        with research.report_lock:
            directory = research.storage / "web" / "exports"
            directory.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(suffix=f".{format}", dir=directory)
            os.close(fd)
            path = Path(name)
            try:
                generate_report(record, path, format, ReportOptions())
            except Exception:
                path.unlink(missing_ok=True)
                raise
        return FileResponse(path, filename=f"{exp_id}.{format}",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    @protected.get("/settings")
    def get_settings() -> dict:
        default = {k:v for k,v in asdict(settings.experiment_config()).items() if k in TrainingRequest.model_fields}
        if web_defaults.is_file():
            try: default = TrainingRequest.model_validate_json(web_defaults.read_text()).model_dump()
            except ValueError: pass
        return {"defaults":default, "strategies":STRATEGY_LABELS, "attacks":ATTACK_DESCRIPTIONS,
                "python":platform.python_version(), "pipeline_version":2,
                "data_location":"Configured local dataset directory", "storage_location":"Configured local research storage",
                "execution":"Single process · one concurrent simulation", "sessions":"8 hours · restart signs out all users",
                "privacy":"No differential privacy, secure aggregation or encryption between simulated clients."}

    @protected.put("/settings")
    def update_settings(body: SettingsUpdate) -> dict:
        atomic_write_text(web_defaults, body.defaults.model_dump_json(indent=2))
        return get_settings()

    register_personalization_routes(protected, research, sessions)
    app.include_router(protected)
    dist = PROJECT_ROOT / "web" / "dist"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        candidate = (dist / path).resolve()
        if not candidate.is_relative_to(dist.resolve()):
            raise HTTPException(404)
        if candidate.is_file():
            return FileResponse(candidate)
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        raise HTTPException(404, "Build the frontend or start the Vite development server")
    return app


def summary_for_compare(record):
    from api.research import summary
    return summary(record)


app = create_app()
