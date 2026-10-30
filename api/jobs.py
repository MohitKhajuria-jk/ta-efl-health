"""One real research simulation at a time; durable status and cooperative cancellation."""
from __future__ import annotations
import copy
import json
import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from experiments.store import atomic_write_text, _json_default
from federated.server.runner import ExperimentConfig, ExperimentCancelled, run_experiment

log = logging.getLogger("ta_efl.web")
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, data_dir: Path, storage_dir: Path):
        self.data_dir, self.storage_dir = data_dir, storage_dir
        self.directory = storage_dir / "web" / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="federation")
        self.stop = threading.Event()
        self.jobs = {}
        self.active = None
        self.closed = False
        for path in self.directory.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] not in TERMINAL:
                    job.update(status="interrupted", stage="interrupted", updated=now(),
                               error="The API stopped before completion. Start a new run; this job was not resumed.")
                    atomic_write_text(path, json.dumps(job))
                self.jobs[job["id"]] = job
            except (ValueError, KeyError, OSError):
                log.warning("Unreadable web job record skipped: %s", path.name)

    def _persist(self, job):
        atomic_write_text(self.directory / f"{job['id']}.json", json.dumps(job, default=_json_default))

    def list(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)[:50]
            # Poll summaries only; the selected job endpoint returns its round/event history.
            return copy.deepcopy([{**{k:v for k,v in job.items() if k not in {"rounds", "events", "clients"}},
                                   "rounds":[], "events":[], "clients":{}} for job in jobs])

    def get(self, job_id):
        with self.lock:
            return copy.deepcopy(self.jobs[job_id])

    def start(self, config):
        with self.lock:
            if self.closed or self.active is not None:
                raise RuntimeError("A simulation is already running. Wait for it to finish or cancel it.")
            job_id = secrets.token_hex(12)
            job = dict(id=job_id, status="running", stage="initializing", created=now(), updated=now(),
                       config=config, round=0, completed_rounds=0, experiment_id=None,
                       cancel_requested=False, error=None, clients={}, rounds=[], events=[])
            self.jobs[job_id] = job
            self.active = job_id
            self.stop = threading.Event()
            self._persist(job)
            self.pool.submit(self._run, job_id, self.stop)
            return copy.deepcopy(job)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] in TERMINAL:
                return copy.deepcopy(job)
            job.update(cancel_requested=True, updated=now())
            self.stop.set()
            self._persist(job)
            return copy.deepcopy(job)

    def _run(self, job_id, stop):
        def emit(event):
            with self.lock:
                job = self.jobs[job_id]
                event = json.loads(json.dumps(event, default=_json_default))
                event["received_at"] = now()
                event["sequence"] = job.get("sequence", 0) + 1
                job["sequence"] = event["sequence"]
                job["events"] = (job["events"] + [event])[-250:]
                kind = event["type"]
                if kind == "experiment_created":
                    job["engine_id"] = event["record_id"]
                if kind == "round_start":
                    job.update(stage="local_training", round=event["round"], clients={})
                elif kind == "aggregating":
                    job["stage"] = "aggregating"
                elif kind == "evaluating":
                    job["stage"] = "evaluating"
                elif kind == "round_aggregated":
                    job["rounds"].append(event)
                    job["completed_rounds"] = event["round"]
                elif kind.startswith("client_") and "cid" in event:
                    job["clients"][event["cid"]] = event
                job["updated"] = now()
                if kind in {"round_start", "round_aggregated", "evaluating", "status"}:
                    self._persist(job)
        try:
            record = run_experiment(ExperimentConfig(**self.jobs[job_id]["config"]),
                                    self.data_dir, self.storage_dir, on_event=emit, stop_event=stop)
            # New web runs carry provenance; historical records are never rewritten.
            record["origin"] = "web"
            record["web_job_id"] = job_id
            from experiments.store import ExperimentStore
            ExperimentStore(self.storage_dir).save(record)
            with self.lock:
                status = "cancelled" if record["status"] == "stopped" else "completed"
                self.jobs[job_id].update(status=status, stage=status, experiment_id=record["id"])
        except Exception as exc:
            with self.lock:
                status = "cancelled" if isinstance(exc, ExperimentCancelled) else "failed"
                self.jobs[job_id].update(status=status, stage=status,
                    error=None if status == "cancelled" else "Simulation failed. Review the local server diagnostics.")
                engine_id = self.jobs[job_id].get("engine_id")
                if engine_id:
                    from experiments.store import ExperimentStore
                    store = ExperimentStore(self.storage_dir)
                    try:
                        failed_record = store.load(engine_id)
                    except FileNotFoundError:
                        pass
                    else:
                        failed_record.update(origin="web", web_job_id=job_id)
                        store.save(failed_record)
                        self.jobs[job_id]["experiment_id"] = engine_id
            if status == "failed":
                log.exception("Simulation %s ended without a complete result", job_id)
        finally:
            with self.lock:
                self.jobs[job_id]["updated"] = now()
                self._persist(self.jobs[job_id])
                self.active = None

    def close(self):
        with self.lock:
            self.closed = True
            self.stop.set()
        self.pool.shutdown(wait=True)
