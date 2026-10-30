import { useState } from "react";
import { Link } from "react-router-dom";
import {
  GitCompareArrows,
  Target,
  ScanLine,
  Users,
  ChartNoAxesCombined,
} from "lucide-react";
import { useData, fmt, percent, strategyName, label } from "../api";
import type { Experiment, ExperimentPage, Dataset } from "../types";
import {
  Intro,
  Card,
  CardHead,
  Field,
  Loading,
  RouteError,
  Metric,
  Notice,
  Empty,
} from "../components/ui";
import { ResultCharts } from "./experiment";
import { Comparison } from "../components/comparison";
export default function Analytics() {
  const [dataset, setDataset] = useState("heart");
  const [attack, setAttack] = useState("");
  const [seed, setSeed] = useState("42");
  const [id, setId] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [compare, setCompare] = useState(false);
  const datasets = useData<Dataset[]>("/datasets");
  const list = useData<ExperimentPage>(
    `/experiments?dataset=${dataset}&attack=${attack}&page_size=100`,
  );
  const runs =
    list.data?.items.filter(
      (e) =>
        e.current &&
        e.status === "completed" &&
        (!seed || String(e.config.seed) === seed),
    ) ?? [];
  const chosen = runs.find((e) => e.id === id) ?? runs[0];
  const detail = useData<Experiment>(
    `/experiments/${chosen?.id}`,
    false,
    !!chosen,
  );
  function clear() {
    setId("");
    setSelected([]);
  }
  return (
    <>
      <Intro
        eyebrow="EVIDENCE, WITH CONTEXT"
        title="Research analytics"
        description="Explore held-out results without losing the configuration behind them."
      />
      <div className="filters">
        <Field label="Dataset">
          <select
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value);
              clear();
            }}
          >
            {datasets.data?.map((d) => (
              <option key={d.key} value={d.key}>
                {d.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Attack scenario">
          <select
            value={attack}
            onChange={(e) => {
              setAttack(e.target.value);
              clear();
            }}
          >
            <option value="">All scenarios</option>
            {[
              "none",
              "sign_flip",
              "scaling",
              "gaussian_noise",
              "label_flip",
              "invalid_update",
            ].map((a) => (
              <option key={a} value={a}>
                {label(a)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Split seed">
          <input
            type="number"
            value={seed}
            onChange={(e) => {
              setSeed(e.target.value);
              clear();
            }}
            placeholder="Any seed"
          />
        </Field>
        <Field label="Inspect saved run">
          <select
            value={chosen?.id ?? ""}
            onChange={(e) => setId(e.target.value)}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id} · {strategyName(r.config.strategy)}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {list.error || detail.error ? (
        <RouteError error={(list.error ?? detail.error)!} />
      ) : list.isPending || (chosen && detail.isPending) ? (
        <Loading />
      ) : detail.data && chosen ? (
        <div className="stack">
          <div className="metrics-grid">
            <Metric
              title="Test ROC-AUC"
              value={fmt(chosen.test_metrics.roc_auc)}
              note={chosen.id.slice(-13)}
              icon={<Target size={18} />}
            />
            <Metric
              title="Precision"
              value={percent(chosen.test_metrics.precision)}
              note="Same saved test evaluation"
              icon={<ScanLine size={18} />}
            />
            <Metric
              title="Recall"
              value={percent(chosen.test_metrics.recall)}
              note="Same saved test evaluation"
              icon={<Users size={18} />}
            />
            <Metric
              title="F1-score"
              value={fmt(chosen.test_metrics.f1)}
              note={`Threshold ${chosen.test_metrics.threshold}`}
              icon={<ChartNoAxesCombined size={18} />}
            />
          </div>
          <ResultCharts record={detail.data} />
          <Card>
            <CardHead
              title="Strategy & attack comparisons"
              sub="Select 2–8 runs. The API verifies compatible evaluation protocols before comparing."
              action={
                <button
                  className="button secondary"
                  disabled={selected.length < 2}
                  onClick={() => setCompare(true)}
                >
                  <GitCompareArrows size={16} />
                  Compare ({selected.length})
                </button>
              }
            />
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>
                      <span className="sr-only">Select</span>
                    </th>
                    <th>Run</th>
                    <th>Strategy</th>
                    <th>Attack</th>
                    <th>Seed</th>
                    <th>AUC</th>
                    <th>F1</th>
                    <th>Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`Compare ${r.id}`}
                          checked={selected.includes(r.id)}
                          disabled={
                            !selected.includes(r.id) && selected.length >= 8
                          }
                          onChange={(e) =>
                            setSelected(
                              e.target.checked
                                ? [...selected, r.id]
                                : selected.filter((x) => x !== r.id),
                            )
                          }
                        />
                      </td>
                      <td>
                        <Link
                          className="text-link"
                          to={`/app/experiments/${r.id}`}
                        >
                          {r.id}
                        </Link>
                      </td>
                      <td>{strategyName(r.config.strategy)}</td>
                      <td>{label(r.config.attack_type)}</td>
                      <td>{r.config.seed}</td>
                      <td>{fmt(r.test_metrics.roc_auc)}</td>
                      <td>{fmt(r.test_metrics.f1)}</td>
                      <td>{fmt(r.test_metrics.recall)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Notice>
              Showing up to 100 most recent matching records; use Experiment
              history for full pagination. Individual saved runs are shown
              without pooling test splits. No confidence interval or independent
              replication claim is inferred.
            </Notice>
          </Card>
        </div>
      ) : (
        <Empty title="No matching evaluation">
          Change the dataset, attack or seed filter to inspect available
          results.
        </Empty>
      )}
      <Comparison
        ids={selected}
        open={compare}
        onClose={() => setCompare(false)}
      />
    </>
  );
}
