"""Trust engine for TA-EFL.

Trust for client k at round t combines:
  1. Update similarity  — cosine similarity between the client's parameter
     update and the coordinate-wise MEDIAN update of the cohort (robust to
     a minority of attackers).
  2. Explanation consistency — similarity between the client's SHAP
     signature (mean |SHAP| per feature on a shared reference set) and the
     cohort's median signature. A poisoned model often attributes risk to
     the wrong features even when its loss looks plausible.
  3. History — exponential moving average for smooth, stable trust.

T_k(t) = beta * T_k(t-1) + (1-beta) * [alpha * s_upd + (1-alpha) * s_exp]

Aggregation weights: p_k proportional to n_k * T_k^gamma, with hard drop
below threshold tau.
"""
from __future__ import annotations

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


class TrustEngine:
    def __init__(
        self, n_clients: int, alpha: float = 0.0, beta: float = 0.6,
        gamma: float = 2.0, tau: float = 0.25,
    ):
        # alpha = 0.0 is the final configuration selected by the E6 sweep
        # (explanation-consistency only). Previous default was 0.5.
        self.alpha = alpha      # weight of update-similarity vs explanation
        self.beta = beta        # EMA memory
        self.gamma = gamma      # aggregation sharpness
        self.tau = tau          # hard-drop threshold
        self.trust = np.ones(n_clients) * 0.8
        self.history: list[np.ndarray] = [self.trust.copy()]
        self.last_components: list[dict] = []

    def score_round(
        self, global_theta: np.ndarray, thetas: list[np.ndarray],
        signatures: list[np.ndarray],
    ) -> np.ndarray:
        updates = [t - global_theta for t in thetas]
        med_update = np.median(np.stack(updates), axis=0)
        med_sig = np.median(np.stack(signatures), axis=0)

        comps = []
        raw = np.zeros(len(thetas))
        for k, (u, s) in enumerate(zip(updates, signatures)):
            s_upd = max(0.0, _cos(u, med_update))          # in [0,1]
            s_exp = max(0.0, _cos(s, med_sig))             # in [0,1]
            # Penalize abnormal update magnitude (catches scaling attacks
            # that keep direction but blow up the norm).
            norms = np.array([np.linalg.norm(x) for x in updates])
            med_norm = np.median(norms) + 1e-12
            mag = min(1.0, med_norm / (np.linalg.norm(u) + 1e-12))
            s_upd = s_upd * mag
            raw[k] = self.alpha * s_upd + (1 - self.alpha) * s_exp
            comps.append({"update_sim": s_upd, "explanation_sim": s_exp,
                          "magnitude_factor": mag})

        self.trust = self.beta * self.trust + (1 - self.beta) * raw
        self.history.append(self.trust.copy())
        self.last_components = comps
        return self.trust

    def aggregation_weights(self, n_samples: list[int]) -> np.ndarray:
        t = self.trust.copy()
        t[t < self.tau] = 0.0                  # hard drop
        w = np.array(n_samples, dtype=float) * np.power(t, self.gamma)
        if w.sum() < 1e-12:                    # degenerate: fall back to uniform
            w = np.ones_like(w)
        return w / w.sum()
