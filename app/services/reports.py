"""Export experiment reports (PDF / HTML / JSON) built only from stored results."""
from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure  # standalone figures: no pyplot, thread-safe per figure

from app.services import charts
from experiments.store import _json_default


class ReportError(Exception):
    pass


@dataclass
class ReportOptions:
    experiment_details: bool = True
    performance_metrics: bool = True
    trust_analysis: bool = True
    shap_explanation: bool = True
    charts: bool = True
    dataset_info: bool = True


METRIC_ROWS = [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
               ("Recall", "recall"), ("F1-Score", "f1"), ("Brier score", "brier"),
               ("Expected calibration error", "ece")]

LIMITATIONS = [
    "In-process Flower simulation: five simulated hospitals in one program; not a deployed "
    "multi-site network. No differential privacy, secure aggregation or encryption.",
    "Model probabilities are model-estimated scores, not clinically calibrated risks; see the "
    "calibration metrics. Research prototype, not a medical device or diagnosis.",
    "Configured attackers are simulation settings. Low trust or zero weight in a simulation is "
    "not proof of malicious behaviour or real-world attack detection.",
    "SHAP values describe model behaviour in log-odds units; they do not establish causation.",
]


def _name(cid: str) -> str:
    head, _, num = cid.rpartition("_")
    return f"Client {num}" if head == "client" and num.isdigit() else cid


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _sections(record: dict[str, Any], opts: ReportOptions,
              explanation: dict[str, Any] | None, importance: list | None) -> list[tuple[str, list[str]]]:
    if record.get("status") not in ("completed", "stopped"):
        raise ReportError("Only completed or stopped experiments with results can be reported")
    from experiments.store import STALE_NOTE, is_current
    if not is_current(record):
        raise ReportError(f"Experiment {record.get('id')} is superseded. {STALE_NOTE}")
    cfg = record["config"]
    tm = record.get("test_metrics", {})
    out: list[tuple[str, list[str]]] = []
    out.append(("Identification", [
        f"Experiment ID: {record['id']}",
        f"Status: {record['status']} ({record.get('rounds_completed', 0)} of {cfg['num_rounds']} rounds)",
        f"Created: {record.get('created')}",
        f"Evaluation split: held-out test set (n = {tm.get('n')}, positives = {tm.get('positives')})",
        f"Report generated: {datetime.now():%Y-%m-%d %H:%M}",
    ]))
    if opts.experiment_details:
        out.append(("Experiment configuration", [
            f"Aggregation: {record.get('strategy_label')}",
            f"Clients: {cfg['num_clients']} simulated hospitals ({cfg['partition']} partition), "
            f"rounds: {cfg['num_rounds']}",
            f"Local training: lr={cfg['learning_rate']}, batch={cfg['batch_size']}, "
            f"epochs={cfg['local_epochs']}, L2={cfg['l2']}, class weight={cfg['class_weight']}",
            f"Decision threshold: {cfg['decision_threshold']} (fixed before training)",
            f"Model: {record['framework']['model']}",
            f"Framework: Flower {record['framework']['flower']} - {record['framework']['transport']}",
            f"Configured attackers: {', '.join(_name(c) for c in record.get('detection', {}).get('configured_attackers', [])) or 'none'}"
            + (f" - {record.get('attack_description')}" if cfg.get('malicious_clients') else ""),
            f"Trust policy: alpha {cfg['trust_alpha']}, initial {cfg['trust_initial']}, "
            f"performance weight {cfg['trust_performance_weight']}, exclusion threshold "
            f"{cfg['trust_exclusion_threshold']}, performance signal "
            f"{cfg.get('trust_performance_mode', 'absolute')}",
            f"Seed: {cfg['seed']}",
        ]))
        pp = record.get("preprocessing", {})
        sc = record.get("split_config", {})
        out.append(("Reproducibility", [
            f"Pipeline version: {record.get('pipeline_version')}; preprocessing fitted on "
            f"{pp.get('fitted_on')} ({pp.get('n_fit_rows')} rows; validation/test rows not used)",
            f"Numeric: {pp.get('numeric')}; categorical: {pp.get('categorical')}",
            f"Split: stratified, seed {sc.get('seed')}, test fraction {sc.get('test_fraction')}, "
            f"validation fraction {sc.get('validation_fraction')}",
            f"Dataset file SHA-256: {record.get('dataset', {}).get('file_sha256') or 'bundled'}",
            f"Rerun: python run_experiment.py --dataset {cfg['dataset_key']} --rounds "
            f"{cfg['num_rounds']} --clients {cfg['num_clients']} --strategy {cfg['strategy']} "
            f"--seed {cfg['seed']}",
        ]))
    if opts.dataset_info:
        d = record.get("dataset", {})
        sp = record.get("splits", {})
        lines = [f"Dataset: {d.get('name')} - {d.get('rows')} records, {d.get('positives')} positive",
                 f"Source: {d.get('source')}",
                 f"Splits: train {sp.get('train')} (partitioned across clients), "
                 f"server validation {sp.get('validation')}, test {sp.get('test')}"]
        for c in record.get("clients", []):
            lines.append(f"  {c['name']}: {c['num_records']} records "
                         f"({c['positive']} positive){' - configured attacker' if c['configured_attacker'] else ''}")
        lines += [f"Note: {w}" for w in d.get("warnings", [])]
        out.append(("Dataset", lines))
    if opts.performance_metrics:
        lines = [f"{name}: {_fmt(tm.get(key))}" for name, key in METRIC_ROWS]
        cm = tm.get("confusion_matrix")
        if cm:
            lines.append(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")
        base = record.get("centralized_baseline")
        if base:
            bt = base["test_metrics"]
            lines.append("Centralised reference (pooled data, not federated): "
                         + ", ".join(f"{n} {_fmt(bt.get(k))}" for n, k in METRIC_ROWS))
        lines.insert(0, f"Held-out test split: n = {tm.get('n')}, prevalence "
                        f"{100 * tm.get('prevalence', 0):.1f}%")
        vm = record.get("final_validation_metrics") or {}
        if vm:
            lines.append(f"Validation split (server-held, used for trust/monitoring; NOT test): "
                         f"n = {vm.get('n')}, ROC-AUC {_fmt(vm.get('roc_auc'))}, accuracy "
                         f"{_fmt(vm.get('accuracy'))}")
        out.append(("Model performance (held-out test set)", lines))
    if opts.trust_analysis:
        lines = [record.get("trust_formula", "").replace("\n", " "),
                 "Trust used for weighting: "
                 + ("yes" if record["config"]["strategy"] == "trust" else "no (monitored only)")]
        last = record["rounds"][-1]["clients"] if record.get("rounds") else []
        weights = {c["cid"]: c["weight"] for c in last}
        for cid, t in record.get("final_trust", {}).items():
            w = weights.get(cid)
            lines.append(f"  {_name(cid)}: final trust {t:.3f}, last-round weight "
                         + ("n/a (median)" if w is None and cid in weights else _fmt(w)))
        det = record.get("detection", {})
        lines.append("Configured attackers: "
                     + (", ".join(_name(c) for c in det.get("configured_attackers", [])) or "none"))
        lines.append("Zero weight in the last round: "
                     + (", ".join(_name(c) for c in det.get("flagged_last_round", [])) or "none"))
        lines.append(det.get("note", ""))
        out.append(("Trust analysis", lines))
    if opts.shap_explanation:
        lines = []
        if explanation:
            lines.append("Local explanation of one prediction:")
            lines.append(explanation["summary"])
            lines += explanation["statements"]
            lines += [f"  {c['label']} = {c['value']}: {c['shap']:+.3f}"
                      for c in explanation["contributions"][:10]]
            lines += explanation["caveats"]
        if importance:
            lines.append("Global importance (mean |SHAP|, log-odds, server-validation split):")
            lines += [f"  {n}: {v:.3f}" for n, v in importance[:10]]
        if not lines:
            lines = ["No SHAP explanation was generated for this experiment's model."]
        out.append(("SHAP explainability", lines))
    out.append(("Limitations", LIMITATIONS))
    return out


def _figures(record: dict[str, Any], explanation: dict[str, Any] | None,
             importance: list | None) -> list[Figure]:
    figs = []
    tm = record.get("test_metrics", {})
    f = Figure(figsize=(8.5, 3.6), dpi=110)
    a1, a2 = f.subplots(1, 2)
    charts.plot_roc(a1, tm.get("roc_curve"), tm.get("roc_auc"), "Federated model (test)")
    from ml.datasets.registry import DatasetRegistry
    spec = DatasetRegistry.get(record["config"]["dataset_key"])
    labels = (spec.negative_label, spec.positive_label)
    charts.plot_confusion(a2, tm.get("confusion_matrix"), labels)
    f.tight_layout()
    figs.append(f)
    f = Figure(figsize=(8.5, 3.6), dpi=110)
    a1, a2 = f.subplots(1, 2)
    charts.plot_calibration(a1, tm.get("calibration_bins"), tm.get("ece"))
    charts.plot_lines(a2, record, "weight")
    a2.set_title("Aggregation weights per round", fontsize=9, loc="left")
    f.tight_layout()
    figs.append(f)
    f = Figure(figsize=(8.5, 3.6), dpi=110)
    a1, a2 = f.subplots(1, 2)
    charts.plot_lines(a1, record, "trust", record["config"]["trust_exclusion_threshold"])
    a1.set_title("Client trust over rounds", fontsize=9, loc="left")
    charts.plot_validation_auc(a2, record)
    a2.set_title("Global model on validation split", fontsize=9, loc="left")
    f.tight_layout()
    figs.append(f)
    if explanation or importance:
        f = Figure(figsize=(8.5, 3.8), dpi=110)
        axes = f.subplots(1, 2) if (explanation and importance) else [f.subplots(1, 1)]
        i = 0
        if explanation:
            charts.plot_contributions(axes[i], [(c["label"], c["shap"]) for c in explanation["contributions"]])
            axes[i].set_title("Local SHAP contributions", fontsize=9, loc="left")
            i += 1
        if importance:
            charts.plot_contributions(axes[i], importance, xlabel="mean |SHAP| (log-odds)")
            axes[i].set_title("Global importance", fontsize=9, loc="left")
        f.tight_layout()
        figs.append(f)
    return figs


def _png(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def generate_report(record: dict[str, Any], path: Path, fmt: str, opts: ReportOptions,
                    explanation: dict[str, Any] | None = None,
                    importance: list[tuple[str, float]] | None = None) -> Path:
    fmt = fmt.lower()
    if fmt not in ("pdf", "html", "json"):
        raise ReportError(f"Unsupported report format '{fmt}'")
    if explanation and explanation.get("experiment_id") not in (None, record["id"]):
        raise ReportError("The SHAP explanation belongs to a different experiment")
    sections = _sections(record, opts, explanation, importance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    title = "TA-EFL Health Experimental Report"
    if fmt == "json":
        payload = {"title": title, "experiment_id": record["id"],
                   "evaluation_split": "test", "sections": {t: l for t, l in sections},
                   "test_metrics": record.get("test_metrics"),
                   "config": record.get("config"),
                   "explanation": explanation if opts.shap_explanation else None,
                   "global_importance": importance if opts.shap_explanation else None}
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path
    figs = _figures(record, explanation if opts.shap_explanation else None,
                    importance if opts.shap_explanation else None) if opts.charts else []
    if fmt == "html":
        parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>",
                 "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:32px auto;"
                 "color:#1f2937}h1{color:#0b1f4b}h2{color:#1d4ed8;border-bottom:1px solid #e5e7eb;"
                 "padding-bottom:4px}li{margin:3px 0}img{max-width:100%}</style></head><body>",
                 f"<h1>{title}</h1>"]
        for t, lines in sections:
            parts.append(f"<h2>{html.escape(t)}</h2><ul>")
            parts += [f"<li>{html.escape(l)}</li>" for l in lines if l]
            parts.append("</ul>")
        for fig in figs:
            parts.append(f"<img alt='chart' src='data:image/png;base64,{_png(fig)}'>")
        parts.append("<p><em>Research software output. Not a medical device; predictions and "
                     "explanations must not be used for diagnosis.</em></p></body></html>")
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
    with PdfPages(path) as pdf:
        lines_out: list[tuple[str, str]] = [("h1", title)]
        for t, lines in sections:
            lines_out.append(("h2", t))
            for l in lines:
                for chunk in _wrap(l, 105):
                    lines_out.append(("p", chunk))
        lines_out.append(("p", ""))
        lines_out.append(("p", "Research software output. Not a medical device."))
        per_page = 52
        for start in range(0, len(lines_out), per_page):
            fig = Figure(figsize=(8.27, 11.69), dpi=100)
            y = 0.95
            for kind, text in lines_out[start:start + per_page]:
                size, weight, color = {"h1": (15, "bold", "#0B1F4B"), "h2": (11, "bold", "#1D4ED8"),
                                       "p": (8, "normal", "#1F2937")}[kind]
                fig.text(0.07, y, text, fontsize=size, fontweight=weight, color=color, va="top",
                         family="DejaVu Sans")
                y -= 0.024 if kind == "h1" else 0.017
            pdf.savefig(fig)
        for fig in figs:
            pdf.savefig(fig)
    return path


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]
