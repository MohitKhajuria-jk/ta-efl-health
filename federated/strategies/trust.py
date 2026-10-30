"""Client trust scoring for TA-EFL aggregation.

Trust policy (all signals are measured by the server from what clients send;
self-reported client metrics are never used):

1. Update validity      - an update with the wrong number/shape of arrays or any
                          non-finite value is rejected (weight 0, instant score 0).
2. Performance signal   - the server evaluates each client's *submitted* model on
                          the server-held validation split (never the test split).
                          "relative" mode (default):
                              P_k = clip((AUC_k - 0.5) / (AUC_med - 0.5), 0, 1)
                          where AUC_med is the median AUC of the round's valid updates
                          (assumes an honest majority, like the median baseline). A client
                          as good as the typical client scores 1 on any dataset; chance-level
                          ranking scores 0. Falls back to absolute mode if AUC_med <= 0.5.
                          "absolute" mode: P_k = clip((AUC_k - 0.5) / 0.5, 0, 1). It was
                          the original policy; on harder datasets (e.g. Framingham, AUC ~0.74)
                          it pushed honest clients towards the exclusion threshold.
3. Similarity signal    - with update Delta_k = w_k - w_global and the
                          coordinate-wise median update Delta_med of the round:
                              S_k = max(0, cos(Delta_k, Delta_med))
                          (S_k = 0 if either vector has zero norm).
4. Instant score        - r_k = lambda * P_k + (1 - lambda) * S_k
                          (lambda = performance_weight; lambda = 1 disables S).
5. Historical trust     - exponential moving average across rounds:
                              T_k(t) = alpha * T_k(t-1) + (1 - alpha) * r_k(t),
                          T_k(0) = initial_trust. Clients that do not report in a
                          round keep T_k unchanged and get weight 0 that round.
6. Aggregation weight   - clients with T_k < tau (exclusion threshold) or an invalid
                          update are excluded; for the rest
                              omega_k = T_k * n_k / sum_j (T_j * n_j)
                          If the denominator is 0, no update is applied and the
                          previous global model is kept.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class TrustConfig:
    alpha: float = 0.7
    initial_trust: float = 0.5
    performance_weight: float = 0.6
    exclusion_threshold: float = 0.3
    performance_mode: str = "relative"  # "relative" | "absolute"

    def __post_init__(self) -> None:
        if self.performance_mode not in ("relative", "absolute"):
            raise ValueError("performance_mode must be 'relative' or 'absolute'")
        for name in ("alpha", "initial_trust", "performance_weight", "exclusion_threshold"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0) or not np.isfinite(v):
                raise ValueError(f"Trust parameter {name} must be in [0, 1], got {v}")
        if self.alpha == 1.0:
            raise ValueError("alpha = 1 would freeze trust at its initial value")

    def to_dict(self) -> dict:
        return asdict(self)

    def formula_text(self) -> str:
        lam = self.performance_weight
        perf = ("P_k = clip((AUC_val_k - 0.5)/(median_j AUC_val_j - 0.5), 0, 1)"
                if self.performance_mode == "relative" else "P_k = clip((AUC_val_k - 0.5)/0.5, 0, 1)")
        return (
            f"{perf};  S_k = max(0, cos(Δ_k, median Δ));\n"
            f"r_k = {lam:.2f}·P_k + {1 - lam:.2f}·S_k;  "
            f"T_k(t) = {self.alpha:.2f}·T_k(t-1) + {1 - self.alpha:.2f}·r_k(t), "
            f"T_k(0) = {self.initial_trust:.2f};\n"
            f"ω_k = T_k·n_k / Σ_j T_j·n_j over clients with T_k ≥ {self.exclusion_threshold:.2f} "
            f"and a valid update"
        )


def flatten(params: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])


def validate_update(params: list[np.ndarray], expected_shapes: list[tuple[int, ...]]) -> str | None:
    """Return None if valid, otherwise a human-readable rejection reason."""
    if len(params) != len(expected_shapes):
        return f"expected {len(expected_shapes)} arrays, got {len(params)}"
    for i, (p, shape) in enumerate(zip(params, expected_shapes)):
        if tuple(np.shape(p)) != tuple(shape):
            return f"array {i} has shape {np.shape(p)}, expected {shape}"
        if not np.all(np.isfinite(p)):
            return f"array {i} contains NaN/Inf"
    return None


def performance_signal(auc: float | None, reference_auc: float | None = None) -> float:
    """Absolute signal when ``reference_auc`` is None or <= 0.5, else relative to it."""
    if auc is None or not np.isfinite(auc):
        return 0.0
    denom = 0.5
    if reference_auc is not None and np.isfinite(reference_auc) and reference_auc > 0.5 + 1e-9:
        denom = reference_auc - 0.5
    return float(np.clip((auc - 0.5) / denom, 0.0, 1.0))


def reference_auc(aucs: list[float | None], mode: str) -> float | None:
    vals = [a for a in aucs if a is not None and np.isfinite(a)]
    if mode != "relative" or not vals:
        return None
    return float(np.median(vals))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_signals(deltas: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float]]:
    """Return (clipped similarity S_k, raw cosine) relative to the median update."""
    if not deltas:
        return {}, {}
    median = np.median(np.stack(list(deltas.values())), axis=0)
    raw = {cid: cosine(d, median) for cid, d in deltas.items()}
    return {cid: max(0.0, c) for cid, c in raw.items()}, raw


def trust_weights(trust: dict[str, float], num_examples: dict[str, int],
                  eligible: set[str], threshold: float) -> dict[str, float] | None:
    """omega_k = T_k n_k / sum_j T_j n_j over eligible clients with T_k >= threshold.

    Returns a weight for every client in ``trust`` (0 when excluded), or ``None``
    when no client can receive positive weight.
    """
    raw: dict[str, float] = {}
    for cid, t in trust.items():
        n = num_examples.get(cid, 0)
        if cid in eligible and t >= threshold and n > 0 and t > 0:
            raw[cid] = float(t) * float(n)
        else:
            raw[cid] = 0.0
    total = sum(raw.values())
    if total <= 0 or not np.isfinite(total):
        return None
    weights = {cid: v / total for cid, v in raw.items()}
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    return weights


@dataclass
class TrustManager:
    config: TrustConfig
    client_ids: list[str]
    trust: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for cid in self.client_ids:
            self.trust.setdefault(cid, self.config.initial_trust)

    def instant_score(self, perf: float, sim: float) -> float:
        lam = self.config.performance_weight
        return float(lam * perf + (1.0 - lam) * sim)

    def update(self, cid: str, instant: float) -> float:
        prev = self.trust.get(cid, self.config.initial_trust)
        a = self.config.alpha
        self.trust[cid] = float(a * prev + (1.0 - a) * instant)
        return self.trust[cid]
