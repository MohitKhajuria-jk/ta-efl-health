import { useState } from "react";
import { useData, fmt, percent, strategyName } from "../api";
import type { Experiment, ExperimentPage, Round } from "../types";
import {
  Card,
  CardHead,
  Intro,
  Loading,
  RouteError,
  Notice,
  Empty,
  Field,
  Badge,
} from "../components/ui";
import { TrustChart } from "../components/charts";
export function TrustTable({ round }: { round: Round }) {
  const weights = round.clients
    .filter((c) => c.weight != null)
    .map((c) => c.weight!);
  const sum = weights.reduce((a, b) => a + b, 0);
  return (
    <>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Client</th>
              <th>Trust</th>
              <th>Weight</th>
              <th>Training rows</th>
              <th>Val. AUC</th>
              <th>Performance</th>
              <th>Similarity</th>
              <th>Update</th>
            </tr>
          </thead>
          <tbody>
            {round.clients.map((c) => (
              <tr key={c.cid}>
                <td>Client {c.cid.split("_")[1]}</td>
                <td className="numeric">{fmt(c.trust)}</td>
                <td>
                  <div className="weight-track">
                    {c.weight != null && (
                      <div>
                        <span style={{ width: `${c.weight * 100}%` }} />
                      </div>
                    )}
                    {c.weight == null ? "N/A · median" : percent(c.weight)}
                  </div>
                </td>
                <td>{c.num_examples}</td>
                <td>{fmt(c.val_auc)}</td>
                <td>{fmt(c.performance)}</td>
                <td>{fmt(c.similarity)}</td>
                <td>
                  <Badge
                    tone={
                      !c.valid ? "error" : c.excluded ? "warning" : "success"
                    }
                  >
                    {!c.valid
                      ? "Rejected"
                      : c.excluded
                        ? "Excluded"
                        : "Accepted"}
                  </Badge>
                  {c.reason && <small>{c.reason}</small>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 15 }}>
        {weights.length
          ? `Weight sum: ${sum.toFixed(6)}. ${Math.abs(sum - 1) < 1e-6 ? "Normalized across contributing clients." : !round.model_updated && sum === 0 ? "No eligible update; previous model retained." : "Normalization requires review."}`
          : "Coordinate-wise median has no per-client aggregation weights."}{" "}
        {round.trust_used_for_weighting
          ? "Trust influences the aggregation."
          : "Trust is monitored only for this baseline."}
      </p>
    </>
  );
}
export default function Trust() {
  const [dataset, setDataset] = useState("");
  const [id, setId] = useState("");
  const [roundIndex, setRoundIndex] = useState(-1);
  const list = useData<ExperimentPage>(
    `/experiments?page_size=100&dataset=${dataset}`,
  );
  const selected = id || list.data?.items.find((e) => e.current)?.id;
  const detail = useData<Experiment>(
    `/experiments/${selected}`,
    false,
    !!selected,
  );
  const rounds = detail.data?.rounds ?? [];
  const round = rounds[roundIndex] ?? rounds.at(-1);
  return (
    <>
      <Intro
        eyebrow="THE SIGNALS BEHIND THE CONTRIBUTION"
        title="Trust monitoring"
        description="Inspect how validation performance and update similarity shape client influence."
      />
      <div className="filters">
        <Field label="Dataset">
          <select
            value={dataset}
            onChange={(e) => {
              setDataset(e.target.value);
              setId("");
              setRoundIndex(-1);
            }}
          >
            <option value="">All datasets</option>
            {["heart", "diabetes", "breast_cancer", "framingham", "stroke"].map(
              (k) => (
                <option key={k} value={k}>
                  {k.replaceAll("_", " ")}
                </option>
              ),
            )}
          </select>
        </Field>
        <Field label="Saved experiment">
          <select
            value={selected ?? ""}
            onChange={(e) => {
              setId(e.target.value);
              setRoundIndex(-1);
            }}
          >
            {list.data?.items
              .filter((e) => e.current)
              .map((e) => (
                <option value={e.id} key={e.id}>
                  {e.id} · {strategyName(e.config.strategy)}
                </option>
              ))}
          </select>
        </Field>
        <Field label="Aggregation round">
          <select
            value={round?.round ?? ""}
            onChange={(e) =>
              setRoundIndex(
                rounds.findIndex((r) => r.round === Number(e.target.value)),
              )
            }
          >
            {rounds.map((r) => (
              <option value={r.round} key={r.round}>
                Round {r.round}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {list.error || detail.error ? (
        <RouteError error={(list.error ?? detail.error)!} />
      ) : list.isPending || (selected && detail.isPending) ? (
        <Loading />
      ) : round ? (
        <div className="stack">
          <Card>
            <CardHead
              title="Trust through the rounds"
              sub={`${detail.data!.dataset.name} · ${strategyName(detail.data!.config.strategy)} · saved history`}
            />
            <TrustChart rounds={rounds} />
          </Card>
          <Card>
            <CardHead
              title={`Round ${round.round} contributions`}
              sub="Low trust is a policy output, not proof of malicious behaviour."
            />
            <TrustTable round={round} />
          </Card>
          <Card>
            <CardHead title="The trust policy" />
            <pre className="mono-block">{detail.data!.trust_formula}</pre>
            <Notice>
              The policy assumes an honest majority. Scaling attacks may retain
              high trust. Validation data informs trust; held-out test data does
              not.
            </Notice>
          </Card>
        </div>
      ) : (
        <Empty />
      )}
    </>
  );
}
