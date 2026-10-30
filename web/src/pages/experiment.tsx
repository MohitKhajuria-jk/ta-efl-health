import { useParams } from "react-router-dom";
import { Download, Target, BarChart3, ScanLine, Users } from "lucide-react";
import { useData, fmt, percent, strategyName, date, label } from "../api";
import type { Experiment } from "../types";
import {
  Card,
  CardHead,
  Intro,
  Loading,
  RouteError,
  Metric,
  Enter,
  Notice,
  Badge,
} from "../components/ui";
import { TrainingChart, RocChart, TrustChart } from "../components/charts";
import { TrustTable } from "./trust";
export function ResultCharts({ record }: { record: Experiment }) {
  const m = record.test_metrics;
  return (
    <div className="split">
      <Card>
        <CardHead
          title="Receiver operating characteristic"
          sub={`Held-out test · ${m.n} records`}
        />
        {m.roc_curve ? (
          <RocChart experiments={[record]} />
        ) : (
          <Notice>ROC curve is unavailable for this evaluation.</Notice>
        )}
      </Card>
      <Card>
        <CardHead
          title="Confusion matrix"
          sub={`Predicted class at the saved threshold ${m.threshold}`}
        />
        <div className="confusion-grid">
          {[
            "True negative",
            "False positive",
            "False negative",
            "True positive",
          ].map((name, i) => (
            <div
              key={name}
              className={`confusion-cell ${i === 1 || i === 2 ? "error-cell" : ""}`}
            >
              <strong>{m.confusion_matrix?.flat()[i] ?? "—"}</strong>
              <span>{name}</span>
            </div>
          ))}
        </div>
        <div className="inline small muted" style={{ marginTop: 20 }}>
          Test prevalence {percent(m.prevalence)} · Brier {fmt(m.brier)} · ECE{" "}
          {fmt(m.ece)}
        </div>
      </Card>
    </div>
  );
}
export default function ExperimentDetail() {
  const { id } = useParams();
  const query = useData<Experiment>(`/experiments/${id}`);
  if (query.isPending) return <Loading />;
  if (query.error) return <RouteError error={query.error} />;
  const e = query.data!;
  const m = e.test_metrics;
  return (
    <>
      <Intro
        eyebrow="A TRACEABLE RESEARCH RESULT"
        title={e.config.label || e.dataset.name}
        description={`${strategyName(e.config.strategy)} · ${date(e.created)} · ${label(e.origin)}`}
        action={
          <Badge tone={e.status === "completed" ? "success" : "neutral"}>
            {e.status}
          </Badge>
        }
      />
      <p className="detail-header-code">
        <code>{e.id}</code>
      </p>
      {!e.current ? (
        <Notice error>
          This result uses a superseded pipeline. Retrain before using its
          metrics or model.
        </Notice>
      ) : e.status === "failed" || !m?.confusion_matrix ? (
        <Card>
          <Notice error>
            This experiment did not produce a complete evaluation. No model
            metrics are available for use.
          </Notice>
          <CardHead title="Recorded configuration" />
          <pre className="mono-block">{JSON.stringify(e.config, null, 2)}</pre>
        </Card>
      ) : (
        <div className="stack">
          <Enter className="metrics-grid">
            <Metric
              title="Test ROC-AUC"
              value={fmt(m.roc_auc)}
              note="Held-out evaluation"
              icon={<Target size={18} />}
            />
            <Metric
              title="Precision"
              value={percent(m.precision)}
              note="Positive predictive value"
              icon={<ScanLine size={18} />}
            />
            <Metric
              title="Recall"
              value={percent(m.recall)}
              note="Sensitivity on test rows"
              icon={<Users size={18} />}
            />
            <Metric
              title="F1-score"
              value={fmt(m.f1)}
              note={`${m.n} test records`}
              icon={<BarChart3 size={18} />}
            />
          </Enter>
          <ResultCharts record={e} />
          <div className="split">
            <Card>
              <CardHead
                title="Validation learning curve"
                sub="Monitoring data; separate from the test split"
              />
              <TrainingChart data={e.validation_history ?? []} />
            </Card>
            <Card>
              <CardHead title="Research provenance" />
              <div className="key-values">
                {[
                  ["Pipeline", e.pipeline_version],
                  ["Dataset rows", e.dataset.rows],
                  ["Seed", e.config.seed],
                  [
                    "Training rounds",
                    `${e.rounds_completed} / ${e.config.num_rounds}`,
                  ],
                  ["Clients", e.config.num_clients],
                  ["Attack", label(e.config.attack_type)],
                  [
                    "Preprocessing",
                    e.preprocessing?.fitted_on ?? "Unavailable",
                  ],
                  [
                    "Model artifact",
                    e.artifact_dir ? "Present in record" : "Unavailable",
                  ],
                ].map(([k, v]) => (
                  <div key={k}>
                    <span>{k}</span>
                    <strong>{v}</strong>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          {e.rounds?.length ? (
            <>
              <Card>
                <CardHead
                  title="Trust history"
                  sub="Measured client signals from the saved run"
                />
                <TrustChart rounds={e.rounds} />
              </Card>
              <Card>
                <CardHead title="Final client contributions" />
                <TrustTable round={e.rounds.at(-1)!} />
              </Card>
            </>
          ) : null}
          <Card>
            <CardHead
              title="Export & reproduce"
              sub="Reports are generated by the existing Python report service."
            />
            <div className="inline">
              {["pdf", "html", "json"].map((format) => (
                <a
                  className="button secondary"
                  key={format}
                  href={`/api/reports/${e.id}?format=${format}`}
                  download
                >
                  <Download size={15} />
                  {format.toUpperCase()} report
                </a>
              ))}
            </div>
            <details style={{ marginTop: 15 }}>
              <summary>Complete experiment configuration</summary>
              <pre className="mono-block">
                {JSON.stringify(e.config, null, 2)}
              </pre>
            </details>
            <Notice>
              These are saved results, not independently reproduced
              measurements. Model scores are not clinically calibrated risk
              estimates.
            </Notice>
          </Card>
        </div>
      )}
    </>
  );
}
