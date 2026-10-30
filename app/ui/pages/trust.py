"""Client trust analysis: trust histories, signals, aggregation weights, policy."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.ui.labels import TRUST_MODE_LABELS, attack_label
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import (
    Card,
    ExperimentPicker,
    Page,
    fill_table,
    fmt,
    hbox,
    label,
    make_table,
)

ALERT = "#FDECEC"


def _tab() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    return w, lay


class TrustPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Client Trust Analysis", "Trust scores and aggregation weights across rounds")
        self.ctx = ctx
        self.picker = ExperimentPicker()
        self.picker.currentIndexChanged.connect(self._picked)
        self.header.add_right(self.picker)
        self.tabs = QTabWidget()

        w, lay = _tab()
        self.trust_card = Card("Client Trust Score Over Rounds")
        self.trust_chart = ChartCanvas(6, 3.4, 300)
        self.trust_card.add(self.trust_chart)
        self.latest_card = Card("Latest Trust Scores")
        self.latest = make_table(["Client", "Trust Score"])
        self.latest.setMinimumHeight(220)
        self.latest_card.add(self.latest)
        self.latest_note = label("", "Muted", wrap=True)
        self.latest_card.add(self.latest_note)
        lay.addWidget(hbox(self.trust_card, self.latest_card, stretch=[3, 1]))
        self.tabs.addTab(w, "Trust Scores")

        w, lay = _tab()
        self.weight_card = Card("Aggregation Weights per Round")
        self.weight_chart = ChartCanvas(6, 3.2, 290)
        self.weight_card.add(self.weight_chart)
        self.part_table = make_table(["Client", "Rounds participated", "Rounds with zero weight",
                                      "Invalid updates", "Mean weight", "Last weight"])
        self.part_table.setMinimumHeight(200)
        self.weight_card.add(self.part_table)
        lay.addWidget(self.weight_card)
        self.tabs.addTab(w, "Aggregation Weights")

        w, lay = _tab()
        self.perf_card = Card("Client Performance (server-measured)")
        self.perf_chart = ChartCanvas(6, 3.2, 290)
        self.perf_card.add(self.perf_chart)
        self.signals = make_table(["Client", "Val AUC", "Performance P", "Cosine", "Similarity S",
                                   "Instant r", "Trust T", "Self-reported loss"])
        self.signals.setMinimumHeight(200)
        self.perf_card.add(self.signals)
        self.perf_card.add(label("All signals are measured by the server from the submitted "
                                 "parameters on the server-validation split. Client self-reported "
                                 "training loss is shown for reference and is not used for trust.",
                                 "Muted", wrap=True))
        lay.addWidget(self.perf_card)
        self.tabs.addTab(w, "Client Performance")

        w, lay = _tab()
        self.cfg_card = Card("Trust Policy & Configuration")
        self.formula = label("", wrap=True)
        self.formula.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 9.5pt;")
        self.formula.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cfg_card.add(self.formula)
        self.cfg_text = label("", wrap=True)
        self.cfg_card.add(self.cfg_text)
        lay.addWidget(self.cfg_card)
        lay.addStretch(1)
        self.tabs.addTab(w, "Formula & Configuration")
        self.root.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self.ctx.reference_mode:
            self._render_reference()
            return
        self.reference_banner.hide()
        recs = [r for r in self.ctx.store.list_current() if r.get("rounds")]
        self.picker.populate(recs, self.ctx.selected_experiment_id)
        self._show(self.picker.current_id())

    def _picked(self, _i: int) -> None:
        exp_id = self.picker.current_id()
        if exp_id:
            self.ctx.select_experiment(exp_id)
        self._show(exp_id)

    def _show(self, exp_id: str | None) -> None:
        rec = None
        if exp_id:
            try:
                rec = self.ctx.store.load(exp_id)
            except FileNotFoundError:
                rec = None
        if rec is None or not rec.get("rounds"):
            self.show_message("info", "No experiment with recorded rounds yet.")
            for c in (self.trust_chart, self.weight_chart, self.perf_chart):
                charts.empty(c.reset()[0], "No experiment yet")
                c.finish()
            for t in (self.latest, self.part_table, self.signals):
                t.setRowCount(0)
            self.formula.setText("")
            self.cfg_text.setText("")
            return
        self.clear_message()
        cfg = rec["config"]
        attackers = {c["cid"] for c in rec["clients"] if c["configured_attacker"]}
        names = {c["cid"]: c["name"] for c in rec["clients"]}
        thr = cfg["trust_exclusion_threshold"]
        charts.plot_lines(self.trust_chart.reset()[0], rec, "trust", thr)
        self.trust_chart.finish()
        last = rec["rounds"][-1]
        trust_now = {c["cid"]: c["trust"] for c in last["clients"]}
        trust_now.update(last.get("non_participating_trust", {}))
        rows = [[names[cid] + (" (attacker)" if cid in attackers else ""), fmt(trust_now.get(cid), 2)]
                for cid in names]
        fill_table(self.latest, rows, lambda i, _r: ALERT if list(names)[i] in attackers or
                   (trust_now.get(list(names)[i], 1) < thr) else None)
        used = cfg["strategy"] == "trust"
        self.latest_note.setText(f"After round {last['round']}. Trust is "
                                 + ("used for aggregation weights." if used else
                                    "monitored only – this strategy does not use it."))
        charts.plot_lines(self.weight_chart.reset()[0], rec, "weight")
        self.weight_chart.finish()
        prow = []
        for cid in names:
            part = zero = invalid = 0
            ws = []
            lastw = None
            for rnd in rec["rounds"]:
                for c in rnd["clients"]:
                    if c["cid"] != cid:
                        continue
                    part += 1
                    invalid += 0 if c["valid"] else 1
                    if c["weight"] is not None:
                        ws.append(c["weight"])
                        zero += 1 if c["weight"] == 0 else 0
                        lastw = c["weight"]
            prow.append([names[cid], part, zero if ws else "n/a (median)", invalid,
                         fmt(sum(ws) / len(ws)) if ws else "n/a", fmt(lastw) if ws else "n/a"])
        fill_table(self.part_table, prow, lambda i, _r: ALERT if list(names)[i] in attackers else None)
        charts.plot_client_performance(self.perf_chart.reset()[0], rec)
        self.perf_chart.finish()
        fill_table(self.signals, [[names.get(c["cid"], c["cid"]), fmt(c["val_auc"]), fmt(c["performance"]),
                                   fmt(c["cosine"]), fmt(c["similarity"]), fmt(c["instant"]),
                                   fmt(c["trust"]), fmt(c["train_loss_reported"])]
                                  for c in last["clients"]],
                   lambda i, _r: ALERT if last["clients"][i]["cid"] in attackers else None)
        self.formula.setText(rec.get("trust_formula", ""))
        det = rec.get("detection", {})
        self.cfg_text.setText(
            f"Strategy: {rec.get('strategy_label')} · trust alpha {cfg['trust_alpha']}, initial "
            f"{cfg['trust_initial']}, performance weight {cfg['trust_performance_weight']}, "
            f"exclusion threshold {thr}, performance signal "
            f"{TRUST_MODE_LABELS.get(cfg.get('trust_performance_mode', 'absolute'))}.\n"
            f"Configured attackers (simulation setting): {', '.join(names[a] for a in attackers) or 'none'}"
            + (f" – {attack_label(cfg.get('attack_type'))}: {rec.get('attack_description')}"
               if attackers else "") + "\n"
            f"Flagged (zero weight) in the last round: "
            f"{', '.join(names.get(c, c) for c in det.get('flagged_last_round', [])) or 'none'}.\n"
            f"{det.get('note', '')}\n"
            "Only the signals listed in the formula are implemented. Low trust is a policy output, "
            "not proof that a client is malicious.")

    def _render_reference(self) -> None:
        self.reference_banner.show()
        self.clear_message()
        fill_table(self.latest, [[n, f"{v:.2f}"] for n, v in ref.TRUST_LATEST],
                   lambda i, _r: ALERT if ref.TRUST_LATEST[i][1] < 0.5 else None)
        self.latest_note.setText("Reference values from the design image.")
        for c in (self.trust_chart, self.weight_chart, self.perf_chart):
            charts.empty(c.reset()[0], "Per-round values are not readable from the design image")
            c.finish()
        self.part_table.setRowCount(0)
        self.signals.setRowCount(0)
        self.formula.setText("")
        self.cfg_text.setText("")
