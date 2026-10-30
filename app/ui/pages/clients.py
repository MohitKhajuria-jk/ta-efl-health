"""Hospital clients: partition sizes, trust and status for an experiment."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLineEdit, QVBoxLayout

from app.services import charts, reference_data as ref
from app.services.context import AppContext
from app.ui import icons
from app.ui.labels import attack_label
from app.ui.widgets.chart_canvas import ChartCanvas
from app.ui.widgets.common import (
    Card,
    ExperimentPicker,
    Page,
    StatCard,
    badge,
    button,
    fill_table,
    fmt,
    hbox,
    label,
    make_table,
)

ROW_ALERT = "#FDECEC"


def client_status(rec: dict, client: dict) -> tuple[str, str]:
    """Status text + badge kind from genuine experiment data (configured vs detected)."""
    last = rec["rounds"][-1] if rec.get("rounds") else None
    flagged = False
    if last:
        for c in last["clients"]:
            if c["cid"] == client["cid"]:
                flagged = c["excluded"]
    attacker = client.get("configured_attacker", False)
    if attacker and flagged:
        return "Configured attacker · zero weight", "bad"
    if attacker:
        return "Configured attacker", "bad"
    if flagged:
        return "Zero weight (low trust)", "warn"
    return "Active", "ok"


class ClientDialog(QDialog):
    def __init__(self, parent, rec: dict, client: dict) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{client['name']} – details")
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        status, _ = client_status(rec, client)
        info = (f"<b>{client['name']}</b> ({client['cid']}) — status: {status}<br>"
                f"Records held locally: {client['num_records']} "
                f"(training {client['num_train']}, local evaluation {client['num_eval']}); "
                f"positive: {client['positive']}, negative: {client['negative']}<br>"
                f"Experiment: {rec['id']}")
        if client.get("configured_attacker"):
            info += (f"<br><span style='color:#B91C1C'>Configured attacker in this robustness "
                     f"experiment ({attack_label(rec['config'].get('attack_type'))}): "
                     f"{rec.get('attack_description', '')}</span>")
        lab = label(info, wrap=True)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lab)
        t = make_table(["Round", "Valid", "Val AUC", "Perf P", "Sim S", "Instant r", "Trust T", "Weight"])
        rows = []
        for rnd in rec.get("rounds", []):
            for c in rnd["clients"]:
                if c["cid"] == client["cid"]:
                    rows.append([rnd["round"], "yes" if c["valid"] else f"no: {c['reason']}",
                                 fmt(c["val_auc"]), fmt(c["performance"]), fmt(c["similarity"]),
                                 fmt(c["instant"]), fmt(c["trust"]),
                                 "median" if c["weight"] is None else fmt(c["weight"])])
        fill_table(t, rows, lambda _r, row: ROW_ALERT if row[1] != "yes" else None)
        lay.addWidget(t, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)


class ClientsPage(Page):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__("Hospital Clients", "View dataset details, trust scores and client status")
        self.ctx = ctx
        self.picker = ExperimentPicker()
        self.picker.currentIndexChanged.connect(self._picked)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search client...")
        self.search.addAction(icons.icon("search", "#64748B", 32), QLineEdit.ActionPosition.LeadingPosition)
        self.search.textChanged.connect(self._filter)
        self.search.setMaximumWidth(220)
        refresh = button("", None, "refresh", self.refresh)
        refresh.setToolTip("Refresh")
        self.header.add_right(self.picker)
        self.header.add_right(self.search)
        self.header.add_right(refresh)

        self.card = Card()
        self.table = make_table(["Client ID", "Dataset Size", "Trust Score", "Status",
                                 "Last Round", "Action"])
        self.table.setMinimumHeight(260)
        self.card.add(self.table)
        self.note = label("", "Muted", wrap=True)
        self.card.add(self.note)
        self.root.addWidget(self.card)

        self.s_clients = StatCard("people", "Total Client Records")
        self.s_server = StatCard("shield", "Server/Test Records")
        self.s_total = StatCard("database", "Total Records", "#7C3AED")
        self.root.addWidget(hbox(self.s_clients, self.s_server, self.s_total, stretch=[1, 1, 1]))

        self.mix_card = Card("Class mix per client (local data stays on each client)")
        self.mix = ChartCanvas(6, 2.6, 220)
        self.mix_card.add(self.mix)
        self.root.addWidget(self.mix_card)
        self.root.addStretch(1)
        self._rec: dict | None = None

    def refresh(self) -> None:
        if self.ctx.reference_mode:
            self._render_reference()
            return
        self.reference_banner.hide()
        records = self.ctx.store.list_current()
        self.picker.populate(records, self.ctx.selected_experiment_id)
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
        self._rec = rec
        if not rec or not rec.get("clients"):
            self.table.setRowCount(0)
            for s in (self.s_clients, self.s_server, self.s_total):
                s.set("—")
            self.show_message("info", "No experiment to show. Clients and their partitions are "
                              "created when a federated training run starts.")
            charts.empty(self.mix.reset()[0], "No experiment yet")
            self.mix.finish()
            return
        self.clear_message()
        last_round = rec["rounds"][-1]["round"] if rec.get("rounds") else "—"
        final = rec.get("final_trust") or {}
        if not final and rec.get("rounds"):
            final = {c["cid"]: c["trust"] for c in rec["rounds"][-1]["clients"]}
        rows, alert = [], set()
        for i, c in enumerate(rec["clients"]):
            status, kind = client_status(rec, c)
            if kind == "bad" or kind == "warn":
                alert.add(i)
            view = button("View", None, on_click=lambda c=c: ClientDialog(self, rec, c).exec())
            view.setStyleSheet("color: #1D4ED8; padding: 3px 10px;")
            rows.append([c["name"], c["num_records"], fmt(final.get(c["cid"]), 2),
                         badge(status, kind), last_round, view])
        fill_table(self.table, rows, lambda r, _row: ROW_ALERT if r in alert else None)
        n_clients = sum(c["num_records"] for c in rec["clients"])
        sp = rec.get("splits", {})
        server = sp.get("validation", 0) + sp.get("test", 0)
        self.s_clients.set(str(n_clients))
        self.s_server.set(str(server))
        self.s_total.set(str(n_clients + server))
        self.note.setText(
            "Trust = historical trust T_k after the last round (a policy output, not proof of "
            "malicious behaviour)"
            + ("" if rec["config"]["strategy"] == "trust" else " (monitored only; not used for "
               "weighting by this strategy)")
            + ". 'Configured attacker' is the experiment setting; 'zero weight' means the server "
              "excluded the update (trust below threshold or malformed update).")
        ax = self.mix.reset()[0]
        charts.style_axes(ax)
        names = [c["name"] for c in rec["clients"]]
        neg = [c["negative"] for c in rec["clients"]]
        pos = [c["positive"] for c in rec["clients"]]
        ax.bar(names, neg, color="#93C5FD", label="negative")
        ax.bar(names, pos, bottom=neg, color="#EF4444", label="positive")
        ax.legend(fontsize=7)
        ax.set_ylabel("records", fontsize=8)
        self.mix.finish()
        self._filter(self.search.text())

    def _render_reference(self) -> None:
        self.reference_banner.show()
        self.clear_message()
        rows = []
        for name, size, trust, status, last in ref.CLIENTS:
            rows.append([name, size, f"{trust:.2f}", badge(status, "bad" if status == "Malicious" else "ok"),
                         last, button("View", None, on_click=lambda: self.show_message(
                             "info", "Reference rows have no underlying experiment to inspect."))])
        fill_table(self.table, rows, lambda r, row: ROW_ALERT if ref.CLIENTS[r][3] == "Malicious" else None)
        t = ref.CLIENT_TOTALS
        self.s_clients.set(str(t["client_records"]))
        self.s_server.set(str(t["server_test_records"]))
        self.s_total.set(str(t["total_records"]))
        self.note.setText("")
        charts.empty(self.mix.reset()[0], "Class mix is not shown in the design image")
        self.mix.finish()

    def _filter(self, text: str) -> None:
        text = text.strip().lower()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            self.table.setRowHidden(r, bool(text) and (it is None or text not in it.text().lower()))
