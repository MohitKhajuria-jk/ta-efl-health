import { Link } from "react-router-dom";
import {
  ArrowRight,
  FlaskConical,
  Database,
  Layers,
  Target,
  Plus,
  ArrowUpRight,
} from "lucide-react";
import { useData, fmt, date, strategyName } from "../api";
import type { DashboardData, Experiment } from "../types";
import {
  Card,
  CardHead,
  Intro,
  Metric,
  Enter,
  Loading,
  RouteError,
  Status,
  Badge,
  TextLink,
  Empty,
} from "../components/ui";
import { TrainingChart } from "../components/charts";
import { Network } from "../components/network";
export default function Dashboard() {
  const query = useData<DashboardData>("/dashboard", 5000);
  const latest = query.data?.latest;
  const detail = useData<Experiment>(
    `/experiments/${latest?.id}`,
    false,
    !!latest,
  );
  if (query.isPending) return <Loading />;
  if (query.error)
    return <RouteError error={query.error} retry={() => query.refetch()} />;
  const d = query.data!;
  const round = detail.data?.rounds?.at(-1);
  return (
    <>
      <Intro
        eyebrow="YOUR RESEARCH, IN PERSPECTIVE"
        title="Research overview"
        description="A clear view of your experiments, models, and collaborative learning."
        action={
          <Link className="button primary" to="/app/simulation">
            <Plus size={17} /> New experiment
          </Link>
        }
      />
      <Enter className="metrics-grid">
        <Metric
          title="Experiments"
          value={d.total_experiments.toLocaleString()}
          note="Preserved research history"
          icon={<FlaskConical size={19} />}
        />
        <Metric
          title="Validated datasets"
          value={
            <>
              {d.validated_datasets}
              <span className="metric-denom"> / {d.datasets.length}</span>
            </>
          }
          note="Checked against the local schemas"
          icon={<Database size={19} />}
        />
        <Metric
          title="Saved model references"
          value={d.total_models.toLocaleString()}
          note="Linked to experiment records"
          icon={<Layers size={19} />}
        />
        <Metric
          title="Latest test ROC-AUC"
          value={fmt(latest?.test_metrics?.roc_auc)}
          note={
            latest
              ? `${latest.dataset.name} · held-out test`
              : "Run an experiment to evaluate"
          }
          icon={<Target size={19} />}
        />
      </Enter>
      <div className="dashboard-middle">
        <Card>
          <CardHead
            title="Learning trajectory"
            sub={
              latest
                ? `${strategyName(latest.config.strategy)} · ${latest.dataset.name}`
                : "Validation performance by round"
            }
            action={<Badge>Saved evaluation</Badge>}
          />
          {detail.data ? (
            <>
              <div className="chart-summary">
                <strong>{fmt(latest?.test_metrics?.roc_auc)}</strong>
                <span>
                  test ROC-AUC <small>Validation history shown below</small>
                </span>
              </div>
              <TrainingChart data={detail.data.validation_history ?? []} />
              <div className="card-bottom">
                <span>{detail.data.rounds_completed} completed rounds</span>
                <TextLink to={`/app/experiments/${latest!.id}`}>
                  View experiment
                </TextLink>
              </div>
            </>
          ) : (
            <Empty />
          )}
        </Card>
        <Card className="network-card">
          <CardHead
            title="Federation at a glance"
            sub={
              d.active_job
                ? "Current simulation"
                : "Most recent saved simulation"
            }
            action={
              <Badge tone={d.active_job ? "info" : "neutral"}>
                {d.active_job ? "Running" : "Historical"}
              </Badge>
            }
          />
          <Network
            count={
              d.active_job?.config.num_clients ??
              latest?.config.num_clients ??
              5
            }
            job={d.active_job}
            round={round}
          />
          <div className="card-bottom">
            <span>
              {strategyName(
                d.active_job?.config.strategy ??
                  latest?.config.strategy ??
                  "trust",
              )}
            </span>
            <TextLink to="/app/network">Inspect network</TextLink>
          </div>
        </Card>
      </div>
      <div className="dashboard-bottom">
        <Card>
          <CardHead
            title="Recent experiments"
            sub="Each result is linked to its configuration and model."
            action={<TextLink to="/app/history">View all</TextLink>}
          />
          {d.recent.length ? (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Experiment / dataset</th>
                    <th>Strategy</th>
                    <th>Test AUC</th>
                    <th>Status</th>
                    <th>
                      <span className="sr-only">Open</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {d.recent.slice(0, 5).map((e) => (
                    <tr key={e.id}>
                      <td>
                        <Link
                          className="table-title"
                          to={`/app/experiments/${e.id}`}
                        >
                          {e.dataset.name}
                        </Link>
                        <small className="muted">
                          {date(e.created)} · {e.id.slice(-4)}
                        </small>
                      </td>
                      <td>{strategyName(e.config.strategy)}</td>
                      <td className="numeric">{fmt(e.test_metrics.roc_auc)}</td>
                      <td>
                        <Status value={e.status} />
                      </td>
                      <td>
                        <Link
                          className="icon-button"
                          aria-label={`Open ${e.id}`}
                          to={`/app/experiments/${e.id}`}
                        >
                          <ArrowUpRight size={17} />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty />
          )}
        </Card>
        <Card className="next-card">
          <div className="small-icon">
            <FlaskConical />
          </div>
          <p className="eyebrow">FROM QUESTION TO EVIDENCE</p>
          <h2>
            Your next
            <br />
            experiment starts here.
          </h2>
          <p>
            Configure a cohort, choose an aggregation strategy, and explore how
            trust changes the result.
          </p>
          <Link className="button primary" to="/app/simulation">
            Build an experiment <ArrowRight size={17} />
          </Link>
          <div className="next-note">
            Same Python engine.
            <br />A new perspective on the research.
          </div>
        </Card>
      </div>
      <div className="provenance-strip">
        <span>Latest model</span>
        <code>{latest?.id ?? "No saved model"}</code>
        <Badge>Pipeline v{latest?.pipeline_version ?? 2}</Badge>
      </div>
    </>
  );
}
