import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
  ReferenceLine,
  Bar,
  BarChart,
  Cell,
} from "recharts";
import { useReducedMotion } from "motion/react";
import type { Experiment, Round } from "../types";
const colors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];
const axis = { fontSize: 11, fill: "var(--text-muted)" };
export function TrainingChart({
  data,
}: {
  data: { round: number; auc: number }[];
}) {
  const reduced = useReducedMotion();
  return (
    <div
      className="chart"
      role="img"
      aria-label="Validation ROC-AUC by completed training round"
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 20, right: 10, bottom: 5, left: -20 }}
        >
          <defs>
            <linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.2} />
              <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            stroke="var(--border)"
            strokeDasharray="3 5"
            vertical={false}
          />
          <XAxis
            dataKey="round"
            tick={axis}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tick={axis}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(v) => Number(v).toFixed(3)}
            labelFormatter={(v) => `Round ${v}`}
          />
          <Area
            type="monotone"
            dataKey="auc"
            name="Validation AUC"
            stroke="var(--chart-1)"
            fill="url(#aucFill)"
            strokeWidth={2.5}
            animationDuration={300}
            isAnimationActive={!reduced}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
export function TrustChart({ rounds }: { rounds: Round[] }) {
  const reduced = useReducedMotion();
  const ids = [...new Set(rounds.flatMap((r) => r.clients.map((c) => c.cid)))];
  const data = rounds.map((r) =>
    Object.assign(
      { round: r.round },
      Object.fromEntries(r.clients.map((c) => [c.cid, c.trust])),
    ),
  );
  return (
    <div className="chart" role="img" aria-label="Per-client trust history">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 12, left: -20, bottom: 0 }}
        >
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis dataKey="round" tick={axis} />
          <YAxis domain={[0, 1]} tick={axis} />
          <Tooltip formatter={(v) => Number(v).toFixed(3)} />
          <Legend />
          {ids.map((id, i) => (
            <Line
              key={id}
              name={`Client ${id.split("_")[1]}`}
              dataKey={id}
              stroke={colors[i % colors.length]}
              strokeDasharray={i >= 5 ? "4 3" : undefined}
              dot={false}
              strokeWidth={2}
              connectNulls={false}
              animationDuration={300}
              isAnimationActive={!reduced}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
export function RocChart({ experiments }: { experiments: Experiment[] }) {
  const reduced = useReducedMotion();
  return (
    <div className="chart" role="img" aria-label="Held-out test ROC curves">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 12, right: 20, left: -10, bottom: 18 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            type="number"
            dataKey="fpr"
            domain={[0, 1]}
            tick={axis}
            label={{
              value: "False positive rate",
              position: "insideBottom",
              offset: -12,
              fill: "var(--text-muted)",
              fontSize: 11,
            }}
          />
          <YAxis type="number" dataKey="tpr" domain={[0, 1]} tick={axis} />
          <Tooltip formatter={(v) => Number(v).toFixed(3)} />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="var(--text-muted)"
            strokeDasharray="4 4"
          />
          {experiments.map((e, i) => (
            <Line
              key={e.id}
              name={`${e.config.strategy} · ${e.id.slice(-4)}`}
              data={
                e.test_metrics.roc_curve?.fpr.map((v, j) => ({
                  fpr: v,
                  tpr: e.test_metrics.roc_curve!.tpr[j],
                })) ?? []
              }
              dataKey="tpr"
              stroke={colors[i % 5]}
              dot={false}
              strokeWidth={2.5}
              animationDuration={300}
              isAnimationActive={!reduced}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
export function ShapChart({
  data,
}: {
  data: { label: string; shap: number }[];
}) {
  const reduced = useReducedMotion();
  return (
    <div
      className="chart shap-chart"
      role="img"
      aria-label="SHAP contributions in log-odds; orange increases the positive-class score, teal decreases it"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data.slice(0, 10)}
          margin={{ left: 0, right: 30, top: 10, bottom: 10 }}
        >
          <XAxis type="number" tick={axis} />
          <YAxis
            dataKey="label"
            type="category"
            width={140}
            tick={axis}
            tickLine={false}
          />
          <Tooltip formatter={(v) => Number(v).toFixed(3)} />
          <ReferenceLine x={0} stroke="var(--text-muted)" />
          <Bar
            dataKey="shap"
            name="Log-odds contribution"
            radius={3}
            animationDuration={300}
            isAnimationActive={!reduced}
          >
            {data.slice(0, 10).map((d, i) => (
              <Cell
                key={i}
                fill={d.shap >= 0 ? "var(--chart-3)" : "var(--chart-1)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
