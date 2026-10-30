"""Matplotlib chart builders shared by the GUI and exported reports.

Every function draws ONLY data passed in from an experiment record, a prediction
explanation, or the reference-value module (which callers must label).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes

BLUE = "#1D4ED8"
CLIENT_COLORS = ["#2563EB", "#16A34A", "#F59E0B", "#EF4444", "#EC4899", "#0EA5E9",
                 "#84CC16", "#F97316", "#14B8A6", "#6366F1"]
SERVER_COLOR = "#A855F7"
POS_COLOR = "#EF4444"
NEG_COLOR = "#3B82F6"
GRID = "#E5E7EB"
TEXT = "#1F2937"
MUTED = "#6B7280"


def client_color(i: int) -> str:
    return CLIENT_COLORS[i % len(CLIENT_COLORS)]


def style_axes(ax: Axes, title: str | None = None) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9CA3AF")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT, loc="left", fontweight="bold")


def empty(ax: Axes, message: str) -> None:
    ax.clear()
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color=MUTED, wrap=True,
            transform=ax.transAxes)


# ------------------------------------------------------------------ ROC
def plot_roc(ax: Axes, roc: dict[str, list[float]] | None, auc: float | None,
             label: str = "ROC Curve", extra: Sequence[tuple[str, dict, float | None]] = ()) -> None:
    ax.clear()
    if not roc:
        empty(ax, "No ROC curve: run an experiment first")
        return
    style_axes(ax)
    ax.plot(roc["fpr"], roc["tpr"], color=BLUE, lw=2,
            label=f"{label} (AUC = {auc:.3f})" if auc is not None else label)
    for i, (lab, r, a) in enumerate(extra):
        if r:
            ax.plot(r["fpr"], r["tpr"], lw=1.4, ls="--", color=client_color(i + 1),
                    label=f"{lab} (AUC = {a:.3f})" if a is not None else lab)
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=8, color=TEXT)
    ax.set_ylabel("True Positive Rate", fontsize=8, color=TEXT)
    ax.legend(loc="lower right", fontsize=7, frameon=True)


# ------------------------------------------------------------------ data distribution pie
def plot_distribution_pie(ax: Axes, client_sizes: list[tuple[str, int]], server_size: int,
                          server_label: str = "Server/Test") -> None:
    ax.clear()
    if not client_sizes:
        empty(ax, "No client partitions yet")
        return
    labels = [n for n, _ in client_sizes] + [server_label]
    sizes = [s for _, s in client_sizes] + [server_size]
    colors = [client_color(i) for i in range(len(client_sizes))] + [SERVER_COLOR]
    total = sum(sizes)
    ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
           autopct=lambda p: f"{p:.1f}%", pctdistance=0.72,
           textprops={"color": "white", "fontsize": 7, "fontweight": "bold"},
           wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax.set_aspect("equal")
    handles = [ax.patches[i] for i in range(len(labels))]
    ax.legend(handles, [f"{l}: {s} ({100 * s / total:.1f}%)" for l, s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)


# ------------------------------------------------------------------ trust & weights
def trust_series(record: dict[str, Any]) -> tuple[list[int], dict[str, list[float]]]:
    rounds: list[int] = []
    series: dict[str, list[float]] = {c["cid"]: [] for c in record.get("clients", [])}
    for rnd in record.get("rounds", []):
        rounds.append(rnd["round"])
        got = {c["cid"]: c["trust"] for c in rnd["clients"]}
        got.update(rnd.get("non_participating_trust", {}))
        for cid in series:
            series[cid].append(got.get(cid, np.nan))
    return rounds, series


def weight_series(record: dict[str, Any]) -> tuple[list[int], dict[str, list[float]]]:
    rounds: list[int] = []
    series: dict[str, list[float]] = {c["cid"]: [] for c in record.get("clients", [])}
    for rnd in record.get("rounds", []):
        rounds.append(rnd["round"])
        got = {c["cid"]: c["weight"] for c in rnd["clients"]}
        for cid in series:
            w = got.get(cid, 0.0)
            series[cid].append(np.nan if w is None else w)
    return rounds, series


def _client_name(record: dict[str, Any], cid: str) -> str:
    for c in record.get("clients", []):
        if c["cid"] == cid:
            return c["name"] + (" (attacker)" if c.get("configured_attacker") else "")
    return cid


def plot_lines(ax: Axes, record: dict[str, Any], which: str, threshold: float | None = None) -> None:
    ax.clear()
    rounds, series = trust_series(record) if which == "trust" else weight_series(record)
    if not rounds:
        empty(ax, "No rounds recorded yet")
        return
    if which == "weight" and all(np.all(np.isnan(v)) for v in series.values()):
        empty(ax, "Coordinate-wise median has no per-client aggregation weights")
        return
    style_axes(ax)
    for i, (cid, vals) in enumerate(series.items()):
        ax.plot(rounds, vals, marker="o", ms=2.5, lw=1.6, color=client_color(i),
                label=_client_name(record, cid))
    if threshold is not None and which == "trust":
        ax.axhline(threshold, color=MUTED, ls="--", lw=1, label=f"Exclusion threshold {threshold:.2f}")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Trust Score" if which == "trust" else "Aggregation weight", fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="best", frameon=True)


def plot_validation_auc(ax: Axes, record: dict[str, Any]) -> None:
    ax.clear()
    hist = [h for h in record.get("validation_history", []) if h.get("auc") is not None]
    if not hist:
        empty(ax, "No validation history")
        return
    style_axes(ax)
    ax.plot([h["round"] for h in hist], [h["auc"] for h in hist], color=BLUE, marker="o", ms=2.5,
            label="Validation ROC-AUC")
    ax.plot([h["round"] for h in hist], [h["accuracy"] for h in hist], color="#16A34A",
            marker="o", ms=2.5, label="Validation accuracy")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower right")


def plot_client_performance(ax: Axes, record: dict[str, Any]) -> None:
    ax.clear()
    rounds = record.get("rounds", [])
    if not rounds:
        empty(ax, "No rounds recorded yet")
        return
    style_axes(ax)
    cids = [c["cid"] for c in record.get("clients", [])]
    for i, cid in enumerate(cids):
        xs, ys = [], []
        for rnd in rounds:
            for c in rnd["clients"]:
                if c["cid"] == cid and c["val_auc"] is not None:
                    xs.append(rnd["round"])
                    ys.append(c["val_auc"])
        ax.plot(xs, ys, lw=1.5, marker="o", ms=2.5, color=client_color(i),
                label=_client_name(record, cid))
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Validation AUC of submitted model", fontsize=8)
    ax.legend(fontsize=7)


def plot_confusion(ax: Axes, cm: list[list[int]] | None, labels: tuple[str, str]) -> None:
    ax.clear()
    if not cm:
        empty(ax, "No confusion matrix")
        return
    m = np.asarray(cm)
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > m.max() / 2 else TEXT, fontweight="bold")
    ax.set_xticks([0, 1], [f"Pred: {labels[0]}", f"Pred: {labels[1]}"], fontsize=7)
    ax.set_yticks([0, 1], [f"True: {labels[0]}", f"True: {labels[1]}"], fontsize=7)
    ax.set_title("Confusion matrix (test)", fontsize=9, loc="left")


# ------------------------------------------------------------------ SHAP
def plot_contributions(ax: Axes, items: list[tuple[str, float]], highlight: str | None = None,
                       xlabel: str = "SHAP value (log-odds)", max_items: int = 12) -> list:
    """Horizontal bars, largest |value| on top. Returns the bar artists (for picking)."""
    ax.clear()
    if not items:
        empty(ax, "No explanation yet")
        return []
    style_axes(ax)
    items = sorted(items, key=lambda t: abs(t[1]), reverse=True)[:max_items][::-1]
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, height=0.62, picker=True)
    for b, n in zip(bars, names):
        b.set_gid(n)
        if highlight and n == highlight:
            b.set_edgecolor("#111827")
            b.set_linewidth(1.6)
    ax.set_yticks(range(len(vals)), names, fontsize=8)
    span = max(abs(v) for v in vals) or 1.0
    for i, v in enumerate(vals):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=TEXT)
    ax.axvline(0, color="#6B7280", lw=1)
    ax.set_xlim(-1.3 * span, 1.3 * span)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    return list(bars)


def plot_bar_metrics(ax: Axes, labels: list[str], values: list[float | None], title: str = "") -> None:
    ax.clear()
    if not labels:
        empty(ax, "No data")
        return
    style_axes(ax, title)
    vals = [np.nan if v is None else v for v in values]
    ax.bar(range(len(vals)), vals, color=[client_color(i) for i in range(len(vals))])
    ax.set_xticks(range(len(vals)), labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)


def plot_calibration(ax: Axes, bins: list[dict] | None, ece: float | None,
                     title: str = "Reliability (test set)") -> None:
    """Observed positive rate vs mean predicted probability per bin (real data only)."""
    ax.clear()
    if not bins:
        empty(ax, "No calibration data")
        return
    style_axes(ax, title)
    xs = [b["mean_predicted"] for b in bins]
    ys = [b["observed_rate"] for b in bins]
    sizes = [max(12, 4 * b["n"]) for b in bins]
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1, ls=":", label="_perfect")
    ax.plot(xs, ys, color=BLUE, lw=1.5, label="Model")
    ax.scatter(xs, ys, s=sizes, color=BLUE, alpha=0.6, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean predicted probability (bin)", fontsize=8)
    ax.set_ylabel("Observed positive rate", fontsize=8)
    if ece is not None:
        ax.text(0.03, 0.93, f"ECE = {ece:.3f}", transform=ax.transAxes, fontsize=8, color=TEXT)
