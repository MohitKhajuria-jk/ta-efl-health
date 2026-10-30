import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type {
  PersonalizationReport,
  PersonalizationReportMeta,
  PersonalizationRun,
  PersonalizationRunDetail,
} from "../types";
import { useData, fmt, percent, strategyName, date } from "../api";
import {
  Badge,
  Card,
  CardHead,
  Empty,
  Field,
  Intro,
  Loading,
  Notice,
  RouteError,
  Status,
} from "../components/ui";

const axis = { fontSize: 11, fill: "var(--text-muted)" };
const ms = (v: [number | null, number | null, number]) =>
  v[0] == null ? "—" : `${v[0].toFixed(3)} ± ${(v[1] ?? 0).toFixed(3)}`;
const partitionText = (p: {
  partition: string;
  alpha?: number;
  dirichlet_alpha?: number;
}) =>
  p.partition === "dirichlet"
    ? `Non-IID Dirichlet α = ${p.alpha ?? p.dirichlet_alpha}`
    : "IID";

function ReportView({ name }: { name: string }) {
  const report = useData<PersonalizationReport>(
    `/personalization/reports/${name}`,
  );
  if (report.error) return <RouteError error={report.error} />;
  if (report.isPending) return <Loading />;
  const r = report.data;
  const chart = r.coverage.map((c) => {
    const g3 = r.groups.find(
      (g) => g.dataset === c.dataset && g.group === "G3",
    );
    const g4 = r.groups.find(
      (g) => g.dataset === c.dataset && g.group === "G4",
    );
    return {
      dataset: c.dataset,
      "G3 Brier": g3?.brier[0],
      "G4 Brier": g4?.brier[0],
      "G3 ECE": g3?.ece[0],
      "G4 ECE": g4?.ece[0],
    };
  });
  return (
    <>
      <Card>
        <CardHead
          title="Calibration error: trust-aware global (G3) vs hospital-calibrated (G4)"
          sub={`Mean over seeds ${r.meta.seeds.join(", ")} on matched local test rows · lower is better`}
          action={<Badge>{partitionText(r.meta)}</Badge>}
        />
        <div
          className="chart"
          role="img"
          aria-label="Brier score and expected calibration error by dataset for G3 and G4"
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chart}
              margin={{ top: 16, right: 8, bottom: 4, left: -16 }}
            >
              <CartesianGrid
                stroke="var(--border)"
                strokeDasharray="3 5"
                vertical={false}
              />
              <XAxis
                dataKey="dataset"
                tick={axis}
                tickLine={false}
                axisLine={false}
              />
              <YAxis tick={axis} tickLine={false} axisLine={false} />
              <Tooltip
                formatter={(v) =>
                  typeof v === "number" ? v.toFixed(3) : String(v)
                }
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar
                dataKey="G3 Brier"
                fill="var(--chart-1)"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="G4 Brier"
                fill="var(--chart-2)"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="G3 ECE"
                fill="var(--chart-3)"
                radius={[3, 3, 0, 0]}
              />
              <Bar
                dataKey="G4 ECE"
                fill="var(--chart-4)"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
      <Card>
        <CardHead
          title="G1–G4 on identical hospital test rows"
          sub="Mean ± SD over seeds of pooled metrics. Rows: hospitals that received a calibrator. Threshold fixed at 0.5."
        />
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Group</th>
                <th>ROC-AUC</th>
                <th>F1</th>
                <th>Recall</th>
                <th>Brier</th>
                <th>Log loss</th>
                <th>ECE</th>
              </tr>
            </thead>
            <tbody>
              {r.groups
                .filter((g) => g.group !== "S")
                .map((g) => (
                  <tr key={`${g.dataset}-${g.group}`}>
                    <td>{g.dataset}</td>
                    <td>
                      {g.group}: {g.label}
                    </td>
                    <td>{ms(g.roc_auc)}</td>
                    <td>{ms(g.f1)}</td>
                    <td>{ms(g.recall)}</td>
                    <td>{ms(g.brier)}</td>
                    <td>{ms(g.log_loss)}</td>
                    <td>{ms(g.ece)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <div className="table-scroll" style={{ marginTop: 16 }}>
          <table>
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Hospitals calibrated</th>
                <th>Seeds contributing</th>
              </tr>
            </thead>
            <tbody>
              {r.coverage.map((c) => (
                <tr key={c.dataset}>
                  <td>{c.dataset}</td>
                  <td>
                    {c.hospitals_personalised}/{c.hospitals_total}
                  </td>
                  <td>
                    {c.seeds_with_personalised_hospital.length}/{c.seeds.length}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted" style={{ marginTop: 12 }}>
          Hospitals without enough local calibration rows keep the global model
          and are excluded from these comparisons. Within one hospital ROC-AUC
          cannot change (the calibration is strictly increasing); pooled AUC
          across hospitals can.
        </p>
      </Card>
    </>
  );
}

function RunDetail({ id }: { id: string }) {
  const run = useData<PersonalizationRunDetail>(
    `/personalization/experiments/${id}`,
  );
  if (run.error) return <RouteError error={run.error} />;
  if (run.isPending) return <Loading />;
  const r = run.data;
  return (
    <Card>
      <CardHead
        title="Per-hospital results"
        sub={`${r.dataset.name} · ${strategyName(r.global_experiment.strategy)} global model ${r.global_experiment.id}`}
        action={
          <Badge
            tone={r.integrity.global_model_unchanged ? "success" : "error"}
          >
            {r.integrity.global_model_unchanged
              ? "Global model unchanged"
              : "Integrity check failed"}
          </Badge>
        }
      />
      <p className="small muted">
        {r.formula}. {r.description}
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Hospital</th>
              <th>Status</th>
              <th>Prevalence</th>
              <th>Cal / test rows</th>
              <th>a</th>
              <th>b</th>
              <th>ROC-AUC</th>
              <th>F1</th>
              <th>Brier</th>
              <th>Log loss</th>
              <th>ECE</th>
            </tr>
          </thead>
          <tbody>
            {r.hospitals.map((h) => {
              const pair = (
                k: "roc_auc" | "f1" | "brier" | "log_loss" | "ece",
              ) =>
                h.personalized
                  ? `${fmt(h.global[k])} → ${fmt(h.personalized[k])}`
                  : fmt(h.global[k]);
              return (
                <tr key={h.hospital_id}>
                  <td>{h.hospital_id.replace("_", " ")}</td>
                  <td title={h.reason}>
                    <Status value={h.status} />
                  </td>
                  <td>{percent(h.prevalence)}</td>
                  <td>
                    {h.n_calibration} ({h.calibration_positives}+) / {h.n_test}{" "}
                    ({h.test_positives}+)
                  </td>
                  <td>{h.calibrator ? fmt(h.calibrator.a) : "—"}</td>
                  <td>{h.calibrator ? fmt(h.calibrator.b) : "—"}</td>
                  <td>{pair("roc_auc")}</td>
                  <td>{pair("f1")}</td>
                  <td>{pair("brier")}</td>
                  <td>{pair("log_loss")}</td>
                  <td>{pair("ece")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {r.hospitals
        .filter((h) => h.status !== "personalized")
        .map((h) => (
          <Notice key={h.hospital_id}>
            {h.hospital_id.replace("_", " ")} was not calibrated: {h.reason}
          </Notice>
        ))}
    </Card>
  );
}

export default function PersonalizationPage() {
  const reports = useData<PersonalizationReportMeta[]>(
    "/personalization/reports",
  );
  const runs = useData<PersonalizationRun[]>("/personalization/experiments");
  const [report, setReport] = useState("");
  const [dataset, setDataset] = useState("");
  const [selected, setSelected] = useState("");
  if (reports.error || runs.error)
    return <RouteError error={(reports.error ?? runs.error)!} />;
  const chosenReport = report || reports.data?.[0]?.name;
  const list = (runs.data ?? []).filter(
    (r) => !dataset || r.dataset.key === dataset,
  );
  const datasets = [
    ...new Set((runs.data ?? []).map((r) => r.dataset.key)),
  ].sort();
  return (
    <>
      <Intro
        eyebrow="HOSPITAL-SPECIFIC PERSONALISATION"
        title="Hospital calibration"
        description="A frozen global model, recalibrated per hospital: P = sigmoid(a·z + b) fitted on each hospital's own calibration rows."
      />
      <Notice>
        Local probability calibration only: the global model, its features and
        its SHAP explanations are unchanged. This is not local fine-tuning or
        personalised feature learning. Hospitals are simulated partitions of
        each model's held-out test split.
      </Notice>
      {reports.isPending || runs.isPending ? (
        <Loading />
      ) : !runs.data?.length ? (
        <Empty title="No personalisation runs yet">
          Run scripts/run_personalization.py to calibrate existing global
          models.
        </Empty>
      ) : (
        <>
          {!!reports.data?.length && (
            <div className="filters">
              <Field label="Study report">
                <select
                  value={chosenReport}
                  onChange={(e) => setReport(e.target.value)}
                >
                  {reports.data.map((r) => (
                    <option key={r.name} value={r.name}>
                      {partitionText(r.meta)} · {r.meta.hospitals} hospitals ·
                      seeds {r.meta.seeds.join(", ")} · {r.name}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          )}
          {chosenReport && (
            <ReportView key={chosenReport} name={chosenReport} />
          )}
          <Card>
            <CardHead
              title="Personalisation experiment history"
              sub={`${list.length} saved runs · stored separately from historical experiments`}
            />
            <div className="filters">
              <Field label="Dataset">
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="">All datasets</option>
                  {datasets.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Created</th>
                    <th>Global model</th>
                    <th>Partition</th>
                    <th>Calibrated</th>
                    <th>Brier (global → calibrated)</th>
                    <th>ECE (global → calibrated)</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((r) => (
                    <tr key={r.id} aria-selected={selected === r.id}>
                      <td>
                        <code>{r.id}</code>
                      </td>
                      <td>{date(r.created)}</td>
                      <td>
                        {strategyName(r.global_experiment.strategy)} · seed{" "}
                        {r.global_experiment.seed}
                      </td>
                      <td>{partitionText(r.data_protocol)}</td>
                      <td>
                        {r.n_personalized}/{r.n_hospitals}
                      </td>
                      <td>
                        {r.matched_global && r.matched_personalized
                          ? `${fmt(r.matched_global.brier)} → ${fmt(r.matched_personalized.brier)}`
                          : "—"}
                      </td>
                      <td>
                        {r.matched_global && r.matched_personalized
                          ? `${fmt(r.matched_global.ece)} → ${fmt(r.matched_personalized.ece)}`
                          : "—"}
                      </td>
                      <td>
                        <button
                          className="button"
                          onClick={() => setSelected(r.id)}
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          {selected && <RunDetail key={selected} id={selected} />}
        </>
      )}
    </>
  );
}
