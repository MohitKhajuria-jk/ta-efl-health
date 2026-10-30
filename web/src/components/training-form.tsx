import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Play, Save } from "lucide-react";
import type { Settings, Dataset } from "../types";
import { Field, Notice } from "./ui";
import { label, strategyName } from "../api";
export const trainingSchema = z.object({
  dataset_key: z.string(),
  strategy: z.enum(["trust", "fedavg", "median"]),
  num_clients: z.number().int().min(2).max(20),
  num_rounds: z.number().int().min(1).max(500),
  local_epochs: z.number().int().min(1).max(50),
  learning_rate: z.number().min(0.00001).max(1),
  batch_size: z.number().int().min(1).max(4096),
  l2: z.number().min(0).max(1),
  class_weight: z.enum(["none", "balanced"]),
  partition: z.enum(["iid", "dirichlet"]),
  dirichlet_alpha: z.number().min(0.01).max(100),
  fraction_fit: z.number().min(0.1).max(1),
  seed: z.number().int().min(0).max(2147483647),
  decision_threshold: z.number().min(0.05).max(0.95),
  trust_alpha: z.number().min(0).lt(1),
  trust_initial: z.number().min(0).max(1),
  trust_performance_weight: z.number().min(0).max(1),
  trust_exclusion_threshold: z.number().min(0).max(1),
  trust_performance_mode: z.enum(["relative", "absolute"]),
  malicious_clients: z.array(z.number().int()),
  attack_type: z.enum([
    "none",
    "sign_flip",
    "scaling",
    "gaussian_noise",
    "label_flip",
    "invalid_update",
  ]),
  attack_scale: z.number().positive().max(100),
  attack_noise_std: z.number().positive().max(100),
  label: z.string().max(100),
});
export type TrainingValues = z.infer<typeof trainingSchema>;
const numericFields: {
  key: keyof TrainingValues;
  label: string;
  step?: number;
}[] = [
  { key: "learning_rate", label: "Learning rate", step: 0.001 },
  { key: "batch_size", label: "Batch size" },
  { key: "l2", label: "L2 regularization", step: 0.0001 },
  { key: "fraction_fit", label: "Participation fraction", step: 0.1 },
  { key: "dirichlet_alpha", label: "Dirichlet alpha", step: 0.1 },
  { key: "decision_threshold", label: "Decision threshold", step: 0.05 },
  { key: "trust_alpha", label: "Trust history α", step: 0.05 },
  { key: "trust_initial", label: "Initial trust", step: 0.05 },
  {
    key: "trust_performance_weight",
    label: "Performance weight λ",
    step: 0.05,
  },
  {
    key: "trust_exclusion_threshold",
    label: "Exclusion threshold τ",
    step: 0.05,
  },
  { key: "attack_scale", label: "Attack scale", step: 0.5 },
  { key: "attack_noise_std", label: "Noise standard deviation", step: 0.1 },
];
export function TrainingForm({
  settings,
  datasets,
  onSubmit,
  busy = false,
  error,
  save = false,
}: {
  settings: Settings;
  datasets: Dataset[];
  onSubmit: (v: TrainingValues) => void;
  busy?: boolean;
  error?: string;
  save?: boolean;
}) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors },
  } = useForm<TrainingValues>({
    resolver: zodResolver(trainingSchema),
    defaultValues: {
      ...settings.defaults,
      label: settings.defaults.label ?? "",
    } as TrainingValues,
  });
  useEffect(() => {
    reset({
      ...settings.defaults,
      label: settings.defaults.label ?? "",
    } as TrainingValues);
  }, [settings.defaults, reset]);
  const attack = watch("attack_type");
  const num = watch("num_clients");
  const attackers = watch("malicious_clients") ?? [];
  const key = watch("dataset_key");
  const data = datasets.find((d) => d.key === key);
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div className="form-grid">
        <Field label="Dataset">
          <select {...register("dataset_key")}>
            {datasets.map((d) => (
              <option
                value={d.key}
                key={d.key}
                disabled={!d.status.startsWith("valid")}
              >
                {d.name}
                {!d.status.startsWith("valid") ? " · unavailable" : ""}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Aggregation strategy">
          <select {...register("strategy")}>
            {Object.keys(settings.strategies).map((s) => (
              <option key={s} value={s}>
                {strategyName(s)}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Simulated clients">
          <input
            type="number"
            min="2"
            max="20"
            {...register("num_clients", { valueAsNumber: true })}
          />
        </Field>
        <Field label="Federated rounds">
          <input
            type="number"
            min="1"
            max="500"
            {...register("num_rounds", { valueAsNumber: true })}
          />
        </Field>
        <Field label="Local epochs">
          <input
            type="number"
            min="1"
            max="50"
            {...register("local_epochs", { valueAsNumber: true })}
          />
        </Field>
        <Field label="Reproducibility seed">
          <input
            type="number"
            min="0"
            {...register("seed", { valueAsNumber: true })}
          />
        </Field>
        <Field label="Attack scenario">
          <select
            {...register("attack_type", {
              onChange: (e) => {
                if (e.target.value === "none")
                  setValue("malicious_clients", []);
                else if (!attackers.length)
                  setValue("malicious_clients", [Math.min(num, 5)]);
              },
            })}
          >
            {Object.keys(settings.attacks).map((s) => (
              <option key={s} value={s}>
                {label(s)}
              </option>
            ))}
          </select>
        </Field>
        <Field
          label="Experiment label"
          hint="Optional, stored with the configuration"
        >
          <input
            {...register("label")}
            placeholder="A question worth investigating"
          />
        </Field>
      </div>
      {attack !== "none" && (
        <div className="form-section">
          <p className="small muted">{settings.attacks[attack]}</p>
          <div className="inline" style={{ marginTop: 12 }}>
            {Array.from(
              { length: Math.min(num || 2, 20) },
              (_, i) => i + 1,
            ).map((n) => (
              <label key={n} className="small inline">
                <input
                  type="checkbox"
                  checked={attackers.includes(n)}
                  onChange={(e) =>
                    setValue(
                      "malicious_clients",
                      e.target.checked
                        ? [...attackers, n]
                        : attackers.filter((x) => x !== n),
                    )
                  }
                />
                Client {n}
              </label>
            ))}
          </div>
        </div>
      )}
      <details className="form-section">
        <summary>Advanced training & trust parameters</summary>
        <div className="form-grid">
          <Field label="Data partition">
            <select {...register("partition")}>
              <option value="iid">IID</option>
              <option value="dirichlet">Dirichlet non-IID</option>
            </select>
          </Field>
          <Field label="Class weighting">
            <select {...register("class_weight")}>
              <option value="none">None</option>
              <option value="balanced">Balanced</option>
            </select>
          </Field>
          {numericFields.map((f) => (
            <Field key={f.key} label={f.label}>
              <input
                type="number"
                step={f.step ?? 1}
                {...register(f.key, { valueAsNumber: true })}
              />
            </Field>
          ))}
          <Field label="Trust performance signal">
            <select {...register("trust_performance_mode")}>
              <option value="relative">Relative to round median</option>
              <option value="absolute">Absolute</option>
            </select>
          </Field>
        </div>
        <p className="small muted" style={{ marginTop: 12 }}>
          The decision threshold is fixed before training. It is a model
          classification setting, not a clinical threshold.
        </p>
      </details>
      {Object.values(errors).length > 0 && (
        <Notice error>
          {Object.entries(errors)
            .map(([key, e]) => `${label(key)}: ${e.message}`)
            .join(" · ")}
        </Notice>
      )}
      {error && <Notice error>{error}</Notice>}
      <div className="form-actions">
        <button
          className="button primary"
          disabled={busy || !data?.status.startsWith("valid")}
        >
          {save ? <Save size={16} /> : <Play size={16} />}{" "}
          {busy ? "Working…" : save ? "Save defaults" : "Start simulation"}
        </button>
        <span className="small muted">
          {data?.rows?.toLocaleString()} rows · {data?.n_features} features
        </span>
      </div>
    </form>
  );
}
