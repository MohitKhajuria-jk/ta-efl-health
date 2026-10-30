"""Personalisation experiment: G1 FedAvg, G2 Median, G3 Trust-aware, G4 = G3 + local calibration.

Uses EXISTING frozen global models (default: the no-attack runs of the benchmark that
docs/RESULTS.md was generated from). Nothing is retrained and historical records/models
are only read. New results go to storage/personalization/.

    python scripts/run_personalization.py                              # primary protocol
    python scripts/run_personalization.py --partition dirichlet --alpha 1.0
    python scripts/run_personalization.py --partition iid
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.configuration.settings import SettingsManager  # noqa: E402
from federated.personalized_runner import PersonalizationConfig, run_personalization  # noqa: E402
from ml.personalization.calibration import CalibrationConfig  # noqa: E402
from ml.personalization.evaluator import METRIC_KEYS, metrics  # noqa: E402
from ml.personalization.storage import PersonalizationStore  # noqa: E402

GROUPS = [("G1", "fedavg", "global", "FedAvg global"), ("G2", "median", "global", "FedMedian global"),
          ("G3", "trust", "global", "Trust-aware global"),
          ("G4", "trust", "personalized", "Trust-aware + hospital calibration")]
SUPPLEMENTARY = [("FedAvg + calibration", "fedavg"), ("FedMedian + calibration", "median")]
REPORT_METRICS = ("roc_auc", "f1", "precision", "recall", "brier", "log_loss", "ece")


def _ms(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def _fmt(m, sd, n=None):
    if m is None:
        return "—"
    return f"{m:.3f} ± {sd:.3f}" + (f" (n={n})" if n is not None else "")


def pooled(records_by_strategy, which, hospitals):
    rec = records_by_strategy
    hs = [h for h in rec["hospitals"] if h["hospital_id"] in hospitals]
    y = np.concatenate([h["predictions"]["y"] for h in hs]).astype(int)
    p = np.concatenate([h["predictions"]["p_personalized" if which == "personalized" else "p_global"] for h in hs])
    return metrics(y, p, rec["global_experiment"]["decision_threshold"], "pooled matched local test")


def build_report(records: list[dict], meta: dict) -> tuple[dict, str]:
    by = {}
    for r in records:
        by.setdefault((r["dataset"]["key"], r["global_experiment"]["seed"]), {})[r["global_experiment"]["strategy"]] = r
    datasets = sorted({k[0] for k in by})
    rows, per_hospital, coverage = [], [], []
    for ds in datasets:
        seeds = sorted(s for d, s in by if d == ds)
        per_group = {g[0]: {m: [] for m in REPORT_METRICS} for g in GROUPS}
        supp = {name: {m: [] for m in REPORT_METRICS} for name, _ in SUPPLEMENTARY}
        seeds_used, n_pers, n_total = [], 0, 0
        for s in seeds:
            runs = by[(ds, s)]
            if len({r["integrity"]["partition_sha256"] for r in runs.values()}) != 1:
                raise RuntimeError(f"{ds} seed {s}: hospital partitions differ between strategies")
            trust = runs.get("trust")
            if trust is None:
                continue
            done = [h["hospital_id"] for h in trust["hospitals"] if h["personalized"] is not None]
            n_pers += len(done)
            n_total += len(trust["hospitals"])
            for h in trust["hospitals"]:
                per_hospital.append({
                    "dataset": ds, "seed": s, "hospital": h["hospital_id"], "status": h["status"],
                    "reason": h["reason"], "n_cal": h["n_calibration"], "n_test": h["n_test"],
                    "cal_pos": h["calibration_positives"], "test_pos": h["test_positives"],
                    "prevalence": h["prevalence"],
                    "a": h["calibrator"]["a"] if h["calibrator"] else None,
                    "b": h["calibrator"]["b"] if h["calibrator"] else None,
                    "global": {m: h["global"][m] for m in REPORT_METRICS},
                    "personalized": ({m: h["personalized"][m] for m in REPORT_METRICS}
                                     if h["personalized"] else None)})
            if not done:
                continue
            seeds_used.append(s)
            for gid, strat, which, _ in GROUPS:
                if strat in runs:
                    m = pooled(runs[strat], which, done)
                    for k in REPORT_METRICS:
                        per_group[gid][k].append(m[k])
            for name, strat in SUPPLEMENTARY:
                r = runs.get(strat)
                if r and all(h["personalized"] is not None for h in r["hospitals"] if h["hospital_id"] in done):
                    m = pooled(r, "personalized", done)
                    for k in REPORT_METRICS:
                        supp[name][k].append(m[k])
        coverage.append({"dataset": ds, "seeds": seeds, "seeds_with_personalised_hospital": seeds_used,
                         "hospitals_personalised": n_pers, "hospitals_total": n_total})
        for gid, _, _, label in GROUPS:
            rows.append({"dataset": ds, "group": gid, "label": label,
                         **{k: _ms(per_group[gid][k]) for k in REPORT_METRICS}})
        for name, _ in SUPPLEMENTARY:
            rows.append({"dataset": ds, "group": "S", "label": name,
                         **{k: _ms(supp[name][k]) for k in REPORT_METRICS}})
    # G3 vs G4 per-hospital deltas (all personalised hospital-seed pairs)
    deltas = {}
    for ds in datasets:
        hs = [h for h in per_hospital if h["dataset"] == ds and h["personalized"]]
        deltas[ds] = {k: _ms([None if h["personalized"][k] is None or h["global"][k] is None
                              else h["personalized"][k] - h["global"][k] for h in hs]) for k in REPORT_METRICS}
        deltas[ds]["n_hospital_seed_pairs"] = len(hs)
        deltas[ds]["improved_brier"] = sum(h["personalized"]["brier"] < h["global"]["brier"] for h in hs)
        deltas[ds]["improved_ece"] = sum(h["personalized"]["ece"] < h["global"]["ece"] for h in hs)
        deltas[ds]["improved_log_loss"] = sum(h["personalized"]["log_loss"] < h["global"]["log_loss"] for h in hs)
    payload = {"meta": meta, "coverage": coverage, "groups": rows, "g3_vs_g4_per_hospital_delta": deltas,
               "per_hospital": per_hospital, "run_ids": [r["id"] for r in records]}

    L = [f"# TA-EFL Health personalisation report {meta['created']}", "",
         f"**Protocol.** Frozen global models from `{meta['benchmark']}` (attack: {meta['attack']}, "
         f"seeds {meta['seeds']}). Each model's held-out test split is partitioned into "
         f"{meta['hospitals']} hospitals ({meta['partition']}"
         + (f", Dirichlet α = {meta['alpha']}" if meta['partition'] == 'dirichlet' else "")
         + f"), and each hospital is split {int(meta['calibration_fraction']*100)}/"
         f"{100-int(meta['calibration_fraction']*100)} (stratified) into local calibration and local test rows. "
         f"Calibrator: sigmoid(a·z + b), penalty λ = {meta['calibration']['regularization']} towards a = 1, b = 0; "
         f"minimum {meta['calibration']['min_samples']} calibration rows and {meta['calibration']['min_per_class']} "
         "per class. Threshold fixed at the model's 0.5. No test data is used for calibration, thresholds or tuning.", "",
         "**Matched evaluation.** For each dataset and seed, all groups are scored on the same rows: the local test "
         "rows of hospitals that received a calibrator in G4. Values are mean ± SD over seeds of pooled metrics; "
         "n = number of seeds contributing. Within one hospital ROC-AUC is unchanged by construction (a > 0); "
         "pooled AUC across hospitals can change.", "",
         "## Coverage", "", "| Dataset | Hospitals personalised | Seeds contributing |", "|---|---|---|"]
    for c in coverage:
        L.append(f"| {c['dataset']} | {c['hospitals_personalised']}/{c['hospitals_total']} | "
                 f"{len(c['seeds_with_personalised_hospital'])}/{len(c['seeds'])} |")
    L += ["", "## G1–G4 on matched local test rows", "",
          "| Dataset | Group | ROC-AUC | F1 | Precision | Recall | Brier | Log loss | ECE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['group']}: {r['label']} | " + " | ".join(
            _fmt(r[k][0], r[k][1]) for k in REPORT_METRICS) + f" |")
    L += ["", "S rows are supplementary (calibration applied to the baseline global models).", "",
          "## G4 − G3 per personalised hospital", "",
          "| Dataset | Hospital-seed pairs | ΔROC-AUC | ΔF1 | ΔBrier | ΔLog loss | ΔECE | Brier improved | ECE improved |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, d in deltas.items():
        n = d["n_hospital_seed_pairs"]
        L.append(f"| {ds} | {n} | {_fmt(*d['roc_auc'][:2])} | {_fmt(*d['f1'][:2])} | {_fmt(*d['brier'][:2])} | "
                 f"{_fmt(*d['log_loss'][:2])} | {_fmt(*d['ece'][:2])} | {d['improved_brier']}/{n} | {d['improved_ece']}/{n} |")
    L += ["", "## Every hospital (G3 global → G4 personalised)", "",
          "| Dataset | Seed | Hospital | Status | Cal n (pos) | Test n (pos) | a | b | Brier | ECE | F1 | ROC-AUC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def arrow(h, k):
        g = h["global"][k]
        p = h["personalized"][k] if h["personalized"] else None
        gs = "—" if g is None else f"{g:.3f}"
        return gs if h["personalized"] is None else f"{gs} → {'—' if p is None else f'{p:.3f}'}"
    for h in per_hospital:
        L.append(f"| {h['dataset']} | {h['seed']} | {h['hospital']} | {h['status'].replace('_', ' ')} | "
                 f"{h['n_cal']} ({h['cal_pos']}) | {h['n_test']} ({h['test_pos']}) | "
                 f"{'—' if h['a'] is None else f'{h['a']:.3f}'} | {'—' if h['b'] is None else f'{h['b']:.3f}'} | "
                 f"{arrow(h, 'brier')} | {arrow(h, 'ece')} | {arrow(h, 'f1')} | {arrow(h, 'roc_auc')} |")
    L += ["", "Not personalised = the declared minimum calibration sample requirement was not met (reason stored "
          "in each run record); those hospitals keep the global model and are excluded from matched comparisons.", ""]
    return payload, "\n".join(L) + "\n"


def main(argv=None) -> int:
    s = SettingsManager().load()
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="benchmark-20261014-052600")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--strategies", nargs="*", default=["fedavg", "median", "trust"])
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    p.add_argument("--attack", default="none")
    p.add_argument("--hospitals", type=int, default=3)
    p.add_argument("--partition", choices=["dirichlet", "iid"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=s.dirichlet_alpha)
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    p.add_argument("--regularization", type=float, default=1.0)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--storage", type=Path, default=s.storage_path)
    p.add_argument("--report-from", nargs="*", default=None,
                   help="rebuild a report from saved personalisation run ids instead of running")
    a = p.parse_args(argv)
    store = PersonalizationStore(a.storage)
    cal = CalibrationConfig(min_samples=a.min_samples, min_per_class=a.min_per_class,
                            regularization=a.regularization)
    if a.report_from:
        records = [store.load_record(r) for r in a.report_from]
    else:
        bench = json.loads((a.storage / "benchmarks" / f"{a.benchmark}.json").read_text(encoding="utf-8"))
        runs = [r for r in bench["runs"] if r["attack"] == a.attack and r["strategy"] in a.strategies
                and r["seed"] in a.seeds and (not a.datasets or r["dataset"] in a.datasets)]
        records = []
        for i, r in enumerate(runs, 1):
            cfg = PersonalizationConfig(global_experiment_id=r["experiment_id"], num_hospitals=a.hospitals,
                                        partition=a.partition, dirichlet_alpha=a.alpha,
                                        calibration_fraction=a.calibration_fraction, calibration=cal,
                                        label=f"{a.benchmark}:{a.attack}")
            rec = run_personalization(cfg, s.data_path, a.storage)
            records.append(rec)
            agg = rec["aggregate"]
            print(f"[{i}/{len(runs)}] {rec['id']}  personalised {agg['n_personalized']}/{agg['n_hospitals']}", flush=True)
    meta = {"created": datetime.now().strftime("%Y%m%d-%H%M%S"), "benchmark": a.benchmark, "attack": a.attack,
            "seeds": sorted({r["global_experiment"]["seed"] for r in records}), "hospitals": a.hospitals,
            "partition": records[0]["config"]["partition"], "alpha": records[0]["config"]["dirichlet_alpha"],
            "calibration_fraction": records[0]["config"]["calibration_fraction"],
            "calibration": records[0]["config"]["calibration"]}
    payload, md = build_report(records, meta)
    tag = meta["partition"] + (f"-a{meta['alpha']:g}".replace('.', 'p') if meta["partition"] == "dirichlet" else "")
    j, m = store.save_report(f"report-{meta['created']}-{tag}", payload, md)
    print(f"Report: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
