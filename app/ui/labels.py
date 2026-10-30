"""Human-readable names for identifiers stored in experiment records."""
from __future__ import annotations

from typing import Any

ATTACK_LABELS = {
    "none": "No attack",
    "sign_flip": "Sign flipping",
    "scaling": "Update scaling",
    "gaussian_noise": "Gaussian noise",
    "label_flip": "Label flipping",
    "invalid_update": "Malformed update (NaN)",
}
STRATEGY_SHORT = {"trust": "Trust-Aware (TA-EFL)", "fedavg": "FedAvg", "median": "Coordinate-wise Median"}
PARTITION_LABELS = {"iid": "IID (stratified)", "dirichlet": "Non-IID (Dirichlet)"}
TRUST_MODE_LABELS = {"relative": "Relative to round median AUC",
                     "absolute": "Absolute AUC (original policy)"}
STATUS_LABELS = {"completed": "Completed", "stopped": "Stopped early", "failed": "Failed",
                 "running": "Running"}


def client_name(cid: str) -> str:
    head, _, num = cid.rpartition("_")
    return f"Client {num}" if head == "client" and num.isdigit() else cid


def attack_label(key: str | None) -> str:
    return ATTACK_LABELS.get(key or "none", (key or "").replace("_", " ").capitalize())


def attack_summary(cfg: dict[str, Any]) -> str:
    mal = cfg.get("malicious_clients") or []
    if not mal:
        return "No attack"
    who = ", ".join(f"Client {m}" for m in mal)
    return f"{attack_label(cfg.get('attack_type'))} – {who}"


def status_label(status: str | None) -> str:
    return STATUS_LABELS.get(status or "", (status or "unknown").capitalize())


def metric(value: Any, digits: int = 3, missing: str = "N/A") -> str:
    """Format a metric; None means 'not applicable' for this split (never shown as 0)."""
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
