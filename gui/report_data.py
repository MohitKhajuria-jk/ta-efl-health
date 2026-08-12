"""Read-only access to results the project has already produced.

The Reports page presents numbers; it does not compute them. Everything here
either parses ``results/study_results.csv`` (written by ``run_study.py``) or
reads the live :class:`core.session.Session`. No experiment logic is
duplicated — if a value is not in the results file or the session, it is
simply reported as unavailable.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
STUDY_CSV = os.path.join(RESULTS_DIR, "study_results.csv")

STRATEGY_LABELS = {"trust": "Trust-Aware", "fedavg": "FedAvg",
                   "median": "Median"}
STRATEGY_ORDER = ["trust", "fedavg", "median"]


def _f(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_study(path: str | None = None) -> list[dict]:
    """Parse results/study_results.csv into rows; [] if it does not exist."""
    path = path or STUDY_CSV
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "experiment": (r.get("experiment") or "").strip(),
                    "config": (r.get("config") or "").strip(),
                    "attack_frac": (r.get("attack_frac") or "").strip(),
                    "seed": r.get("seed"),
                    "value": _f(r.get("value")),
                    "dataset": (r.get("dataset") or "").strip(),
                    "extra": _f(r.get("extra")),
                })
    except Exception as exc:                                # pragma: no cover
        print(f"[reports] could not read {path}: {exc}")
        return []
    return [r for r in rows if r["value"] is not None]


def mean_by(rows: list[dict], *keys: str, field: str = "value"):
    """Average ``field`` over seeds, grouped by the given row keys."""
    buckets = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is not None:
            buckets[tuple(r.get(k, "") for k in keys)].append(v)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def experiment(rows: list[dict], name: str) -> list[dict]:
    return [r for r in rows if r["experiment"].upper() == name.upper()]


def datasets_in(rows: list[dict]) -> list[str]:
    seen = []
    for r in rows:
        if r["dataset"] and r["dataset"] not in seen:
            seen.append(r["dataset"])
    return seen


def dataset_title(key: str) -> str:
    """Registry name for a dataset key, falling back to the key itself."""
    try:
        from core.datasets import get_spec
        return get_spec(key).name
    except Exception:
        return key


# ---------------------------------------------------------------------------
# Live session views
# ---------------------------------------------------------------------------
def session_metrics(session) -> dict | None:
    """Final-round metrics of the currently trained model."""
    if not session or not session.server.records:
        return None
    return session.server.records[-1].metrics


def session_trust(session) -> list[dict]:
    """Per-client trust breakdown from the last completed round.

    Reuses the values the trust engine already stored: its component scores
    for the round, the EMA history, the final trust vector and the
    aggregation weights actually used.
    """
    if not session or not session.server.records:
        return []
    rec = session.server.records[-1]
    engine = session.server.trust_engine
    rows = []
    for c in session.clients:
        k = c.cid
        comp = (rec.components[k] if rec.components
                and k < len(rec.components) else {})
        perf = None
        if engine is not None and comp:
            # The engine's own per-round score, before the EMA is applied.
            perf = (engine.alpha * comp.get("update_sim", 0.0)
                    + (1 - engine.alpha) * comp.get("explanation_sim", 0.0))
        history = None
        if engine is not None and len(engine.history) >= 2:
            history = float(engine.history[-2][k])
        rows.append({
            "cid": k,
            "name": c.name,
            "attack": c.attack,
            "malicious": c.attack != "none",
            "performance": perf,
            "history": history,
            "final": (float(rec.trust[k]) if rec.trust is not None else None),
            "weight": (float(rec.weights[k]) if rec.weights is not None
                       else None),
            "update_sim": comp.get("update_sim"),
            "explanation_sim": comp.get("explanation_sim"),
        })
    return rows


def trust_threshold(session) -> float | None:
    engine = getattr(session.server, "trust_engine", None) if session else None
    return getattr(engine, "tau", None)


def default_tau() -> float:
    """Exclusion threshold as defined by the trust engine's own default."""
    try:
        from core.trust import TrustEngine
        return float(TrustEngine(1).tau)
    except Exception:
        return 0.25
