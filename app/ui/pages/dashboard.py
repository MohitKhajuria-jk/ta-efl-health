"""Dashboard: overview of the latest experiment and application state."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QHeaderView, QWidget

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.ui.labels import attack_summary, metric, status_label
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import (
    Card,
    MetricCard,
    Page,
    badge,
    fill_table,
    hbox,
    label,
    make_table,
    table_item,
)
from experiments.store import STALE_NOTE, is_current

NOT_RUN = "Not run"


class DashboardPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Dashboard", "Overview of the federated learning system")
        self.ctx = ctx
        self.m_clients = MetricCard("people", "Hospital Clients", "blue")
        self.m_rounds = MetricCard("training", "Training Rounds", "green")
        self.m_records = MetricCard("database", "Total Records", "purple")
        self.m_auc = MetricCard("chart", "Test ROC-AUC", "red")
        grid = QWidget()
        g = QGridLayout(grid)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(12)
        for i, m in enumerate((self.m_clients, self.m_rounds, self.m_records, self.m_auc)):
            g.addWidget(m, 0, i)
        self.root.addWidget(grid)

        self.dist_card = Card("Client Data Distribution")
        self.pie = ChartCanvas(5.2, 3.0, 250)
        self.dist_card.add(self.pie)
        self.roc_card = Card("Model Performance (Test Set)")
        self.roc = ChartCanvas(4.6, 3.0, 250)
        self.roc_card.add(self.roc)
        self.root.addWidget(hbox(self.dist_card, self.roc_card, stretch=[5, 4]))

        self.latest_card = Card("Latest Experiment")
        self.latest_text = label("", wrap=True)
        self.latest_text.setTextFormat(Qt.TextFormat.RichText)
        self.latest_card.add(self.latest_text)
        self.trust_card = Card("Client Trust (latest experiment)")
        self.trust_chart = ChartCanvas(4.0, 2.4, 210)
        self.trust_card.add(self.trust_chart)
        self.root.addWidget(hbox(self.latest_card, self.trust_card, stretch=[1, 1]))

        self.recent_card = Card("Recent Experiments")
        self.recent = make_table(["Created", "Dataset", "Strategy", "Attack", "Rounds", "Status",
                                  "Test ROC-AUC"])
        self.recent.setMinimumHeight(200)
        self.recent.cellDoubleClicked.connect(self._open_experiment)
        self.recent_card.add(self.recent)
        self.recent_card.add(label("Double-click an experiment to open it in Results & Analysis. "
                                   "'Superseded' experiments were produced by the earlier "
                                   "preprocessing pipeline and are not used for any metric.",
                                   "Muted", wrap=True))
        self.side_card = Card("Datasets & Trained Models")
        self.datasets = make_table(["Dataset", "Data file", "Current models"])
        self.datasets.setMinimumHeight(200)
        hh = self.datasets.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.datasets.setWordWrap(True)
        self.side_card.add(self.datasets)
        self.state = label("", "Muted", wrap=True)
        self.side_card.add(self.state)
        self.root.addWidget(hbox(self.recent_card, self.side_card, stretch=[3, 2]))
        self.root.addStretch(1)
        self._navigate = None

    def set_navigator(self, nav) -> None:
        self._navigate = nav

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if self.ctx.reference_mode:
            self._render_reference()
        else:
            self._render_live()
        self._render_tables()

    def _render_reference(self) -> None:
        self.reference_banner.show()
        self.clear_message()
        d = ref.DASHBOARD
        self.m_clients.set(str(d["hospital_clients"]))
        self.m_rounds.set(str(d["training_rounds"]))
        self.m_records.set(str(d["total_records"]))
        self.m_auc.set(f"{d['test_roc_auc']:.3f}")
        charts.plot_distribution_pie(self.pie.reset()[0], d["distribution"], d["server_test_records"])
        self.pie.finish()
        charts.empty(self.roc.reset()[0], f"Reference AUC = {d['test_roc_auc']:.3f}\n(the curve's points "
                     "cannot be read from the design image,\nso no curve is drawn)")
        self.roc.finish()
        charts.empty(self.trust_chart.reset()[0], "Reference mode: per-client trust values are "
                     "shown on the Hospital Clients page")
        self.trust_chart.finish()
        self.latest_text.setText("Reference mode – no experiment data is shown here.")

    def _render_live(self) -> None:
        self.reference_banner.hide()
        rec = self.ctx.store.latest(None, with_model=False)
        if rec is None:
            self.m_clients.set(str(self.ctx.settings.num_clients), "Hospital Clients (configured)",
                               "Configured in Settings; no experiment has run yet")
            self.m_rounds.set(NOT_RUN, "Training Rounds")
            self.m_records.set(NOT_RUN, "Total Records")
            self.m_auc.set(NOT_RUN, "Test ROC-AUC")
            stale = [r for r in self.ctx.store.list() if not is_current(r)]
            self.show_message("info", "No experiment with the current pipeline yet. Open "
                              "'Federated Training' to run one; results will appear here."
                              + (f" ({len(stale)} superseded experiment(s) are hidden: "
                                 f"{STALE_NOTE})" if stale else ""))
            for c, msg in ((self.pie, "No experiment yet"), (self.roc, "No experiment yet"),
                           (self.trust_chart, "No experiment yet")):
                charts.empty(c.reset()[0], msg)
                c.finish()
            self.latest_text.setText("No experiment has been run with the current pipeline.")
            return
        self.clear_message()
        cfg = rec["config"]
        tm = rec.get("test_metrics") or {}
        self.m_clients.set(str(cfg["num_clients"]), "Hospital Clients")
        self.m_rounds.set(f"{rec.get('rounds_completed', 0)}/{cfg['num_rounds']}",
                          "Training Rounds", "completed / configured")
        self.m_records.set(str(rec.get("dataset", {}).get("rows", "—")), "Total Records",
                           rec.get("dataset", {}).get("name", ""))
        self.m_auc.set(metric(tm.get("roc_auc")), "Test ROC-AUC",
                       f"Held-out test split, n = {tm.get('n')}")
        sizes = [(c["name"], c["num_records"]) for c in rec.get("clients", [])]
        sp = rec.get("splits", {})
        charts.plot_distribution_pie(self.pie.reset()[0], sizes,
                                     sp.get("validation", 0) + sp.get("test", 0),
                                     "Server (validation + test)")
        self.pie.finish()
        charts.plot_roc(self.roc.reset()[0], tm.get("roc_curve"), tm.get("roc_auc"))
        self.roc.finish()
        final = rec.get("final_trust", {})
        names = [c["name"] + (" ⚠" if c["configured_attacker"] else "") for c in rec.get("clients", [])]
        charts.plot_bar_metrics(self.trust_chart.reset()[0], names,
                                [final.get(c["cid"]) for c in rec.get("clients", [])],
                                "Final trust per client (⚠ = configured attacker)"
                                + ("" if cfg["strategy"] == "trust" else " – monitored only"))
        self.trust_chart.finish()
        last = rec["rounds"][-1] if rec.get("rounds") else {"clients": []}
        n_valid = sum(1 for c in last["clients"] if c["valid"])
        n_zero = sum(1 for c in last["clients"] if c["excluded"])
        trusts = [t for t in final.values() if t is not None]
        self.latest_text.setText(
            f"<b>Dataset:</b> {rec.get('dataset', {}).get('name')}<br>"
            f"<b>Strategy:</b> {rec.get('strategy_label')} · <b>Attack:</b> {attack_summary(cfg)}<br>"
            f"<b>Status:</b> {status_label(rec.get('status'))} – "
            f"{rec.get('rounds_completed', 0)} of {cfg['num_rounds']} rounds<br>"
            f"<b>Last round participation:</b> {len(last['clients'])} of {cfg['num_clients']} clients "
            f"reported, {n_valid} valid updates, {n_zero} with zero weight<br>"
            f"<b>Test (held-out, n = {tm.get('n')}, prevalence {100 * tm.get('prevalence', 0):.1f}%):</b> "
            f"ROC-AUC {metric(tm.get('roc_auc'))}, accuracy {metric(tm.get('accuracy'))}, "
            f"recall {metric(tm.get('recall'))}, F1 {metric(tm.get('f1'))}<br>"
            f"<b>Trust:</b> " + (f"min {min(trusts):.2f}, max {max(trusts):.2f}" if trusts else "N/A")
            + ("" if cfg["strategy"] == "trust" else " (monitored only, not used for weighting)"))

    def _render_tables(self) -> None:
        recs = self.ctx.store.list()
        rows = []
        for r in recs[:12]:
            tm = r.get("test_metrics") or {}
            current = is_current(r)
            kind = {"completed": "ok", "stopped": "warn", "failed": "bad"}.get(r.get("status"), "info")
            rows.append([table_item(r.get("created", "")[:16].replace("T", " "), data=r["id"]),
                         r.get("dataset", {}).get("name", r["config"]["dataset_key"]),
                         r.get("strategy_label", r["config"]["strategy"]),
                         attack_summary(r["config"]),
                         r.get("rounds_completed", "—"),
                         badge(status_label(r.get("status")) if current else "Superseded",
                               kind if current else "neutral"),
                         metric(tm.get("roc_auc"), missing="—") if current else "—"])
        fill_table(self.recent, rows)
        avail = self.ctx.registry.all_availability()
        counts: dict[str, int] = {}
        for r in recs:
            if r.get("artifact_dir") and is_current(r):
                counts[r["config"]["dataset_key"]] = counts.get(r["config"]["dataset_key"], 0) + 1
        fill_table(self.datasets, [[table_item(a.spec.name), badge("available" if a.available else "missing",
                                                       "ok" if a.available else "warn"),
                                    counts.get(a.spec.key, 0)] for a in avail])
        self.datasets.resizeRowsToContents()
        for r, a in enumerate(avail):
            it = self.datasets.item(r, 0)
            if it is not None:
                it.setToolTip(a.message)
        missing = [a.spec.name for a in avail if not a.available]
        self.state.setText(
            f"Selected dataset: {self.ctx.registry.get(self.ctx.dataset_key).name}."
            + (f" Missing files can be imported from Settings → Dataset availability "
               f"({', '.join(missing)})." if missing else ""))

    def _open_experiment(self, row: int, _col: int) -> None:
        it = self.recent.item(row, 0)
        if it is None:
            return
        self.ctx.select_experiment(it.data(Qt.ItemDataRole.UserRole))
        if self._navigate:
            self._navigate("results")

