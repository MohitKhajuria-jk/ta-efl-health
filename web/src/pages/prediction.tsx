import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ScanLine, ShieldCheck } from "lucide-react";
import type {
  Dataset,
  Experiment,
  Prediction,
  PersonalizationRun,
  PersonalizedPrediction,
} from "../types";
import { useData, post, percent, fmt } from "../api";
import {
  Card,
  CardHead,
  Intro,
  Field,
  Notice,
  Loading,
  RouteError,
  Empty,
  Badge,
} from "../components/ui";
import { ShapChart } from "../components/charts";
type Result = Prediction & { personal?: PersonalizedPrediction };
function CalibrationCard({ p }: { p: PersonalizedPrediction }) {
  const c = p.calibration;
  const t = c.hospital_local_test;
  return (
    <Card>
      <CardHead
        title="Global vs hospital-calibrated probability"
        sub={`${p.hospital_id.replace("_", " ")} · run ${p.personalization_id}`}
        action={<Badge tone="info">Local calibration</Badge>}
      />
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th></th>
              <th>Global model</th>
              <th>Personalised</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Probability ({p.positive_label})</td>
              <td>{percent(p.global.probability_positive)}</td>
              <td>
                <strong>{percent(p.personalized.probability_positive)}</strong>
              </td>
            </tr>
            <tr>
              <td>Classification (threshold {fmt(p.threshold, 2)})</td>
              <td>{p.global.predicted_label}</td>
              <td>{p.personalized.predicted_label}</td>
            </tr>
            <tr>
              <td>Log-odds</td>
              <td>{fmt(p.global.log_odds)}</td>
              <td>{fmt(p.personalized.log_odds)}</td>
            </tr>
            <tr>
              <td>Hospital local-test Brier</td>
              <td>{fmt(t.global.brier)}</td>
              <td>{fmt(t.personalized.brier)}</td>
            </tr>
            <tr>
              <td>Hospital local-test ECE</td>
              <td>{fmt(t.global.ece)}</td>
              <td>{fmt(t.personalized.ece)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <Notice>
        Calibration adjustment: sigmoid({fmt(c.a)} × z {c.b >= 0 ? "+" : "−"}{" "}
        {fmt(Math.abs(c.b))}) · log-odds change{" "}
        {c.log_odds_adjustment >= 0 ? "+" : ""}
        {fmt(c.log_odds_adjustment)} · probability change{" "}
        {c.probability_change >= 0 ? "+" : ""}
        {(c.probability_change * 100).toFixed(1)} points. Fitted on{" "}
        {c.n_calibration} local calibration rows of this simulated hospital.
      </Notice>
      <p className="small muted">{p.explanation_scope}</p>
    </Card>
  );
}
function PatientForm({
  dataset,
  model,
  onResult,
  personal,
}: {
  dataset: Dataset;
  model: string;
  onResult: (r: Result | null) => void;
  personal?: { run: string; hospital: string } | null;
}) {
  const [error, setError] = useState("");
  const schema = useMemo(
    () =>
      z
        .object(
          Object.fromEntries(
            dataset.features.map((f) => [
              f.name,
              z.string().trim().min(1, `${f.label} is required`),
            ]),
          ),
        )
        .superRefine((values, ctx) => {
          for (const f of dataset.features) {
            const v = values[f.name];
            if (!v) continue;
            let message = "";
            if (f.kind === "categorical") {
              if (!f.categories.includes(v))
                message = "Select a supported category";
            } else {
              const n = Number(v);
              if (!Number.isFinite(n)) message = "Enter a finite number";
              else if (f.kind === "binary" && n !== 0 && n !== 1)
                message = "Select 0 or 1";
              else if (f.integer && !Number.isInteger(n))
                message = "Enter a whole number";
              else if (f.min_value != null && n < f.min_value)
                message = `Minimum ${f.min_value}`;
              else if (f.max_value != null && n > f.max_value)
                message = `Maximum ${f.max_value}`;
            }
            if (message)
              ctx.addIssue({ code: "custom", path: [f.name], message });
          }
        }),
    [dataset],
  );
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<Record<string, string>>({ resolver: zodResolver(schema) });
  async function submit(values: Record<string, string>) {
    setError("");
    onResult(null);
    try {
      if (personal) {
        const r = await post<PersonalizedPrediction>(
          "/personalization/predictions",
          {
            experiment_id: model,
            personalization_id: personal.run,
            hospital_id: personal.hospital,
            values,
            explain: true,
          },
        );
        onResult({
          experiment_id: r.experiment_id,
          dataset_key: r.dataset_key,
          probability_positive: r.global.probability_positive,
          threshold: r.threshold,
          predicted_class: Number(r.global.probability_positive >= r.threshold),
          predicted_label: r.global.predicted_label,
          positive_label: r.positive_label,
          notes: r.notes,
          explanation: r.explanation,
          explanation_error: r.explanation_error,
          personal: r,
        });
      } else {
        onResult(
          await post<Prediction>("/predictions", {
            experiment_id: model,
            values,
            explain: true,
          }),
        );
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <form onSubmit={handleSubmit(submit)} noValidate>
      <div className="form-grid">
        {dataset.features.map((f) => (
          <Field
            key={f.name}
            label={`${f.label}${f.unit ? ` (${f.unit})` : ""}`}
            hint={f.help || undefined}
          >
            {f.kind === "categorical" || f.kind === "binary" ? (
              <select
                {...register(f.name)}
                defaultValue=""
                aria-invalid={!!errors[f.name]}
              >
                <option value="" disabled>
                  Select value
                </option>
                {(f.kind === "binary" ? ["0", "1"] : f.categories).map(
                  (value, i) => (
                    <option value={value} key={value}>
                      {f.category_labels[i] ?? value}
                    </option>
                  ),
                )}
              </select>
            ) : (
              <input
                type="number"
                step={f.integer ? "1" : "any"}
                min={f.min_value ?? undefined}
                max={f.max_value ?? undefined}
                placeholder={`${f.min_value ?? "Min"} – ${f.max_value ?? "Max"}`}
                {...register(f.name)}
                aria-invalid={!!errors[f.name]}
              />
            )}
            {errors[f.name] && (
              <span className="field-error">{errors[f.name]?.message}</span>
            )}
          </Field>
        ))}
      </div>
      {error && <Notice error>{error}</Notice>}
      <div className="form-actions">
        <button className="button primary" disabled={isSubmitting}>
          <ScanLine size={17} />
          {isSubmitting ? "Computing prediction & SHAP…" : "Predict & explain"}
        </button>
      </div>
      <p className="prediction-note">
        Inputs are processed in memory and are not saved to prediction history.
        Use de-identified research inputs. The saved model’s exact preprocessing
        and threshold are applied.
      </p>
    </form>
  );
}
export default function PredictionPage() {
  const data = useData<Dataset[]>("/datasets");
  const [datasetKey, setDatasetKey] = useState("heart");
  const models = useData<Experiment[]>(`/models?dataset=${datasetKey}`);
  const [model, setModel] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [mode, setMode] = useState<"global" | "personalized">("global");
  const [runId, setRunId] = useState("");
  const [hospital, setHospital] = useState("");
  const dataset = data.data?.find((d) => d.key === datasetKey);
  const selected = model || models.data?.[0]?.id;
  const runs = useData<PersonalizationRun[]>(
    `/personalization/experiments?global_experiment_id=${selected ?? ""}`,
    false,
    !!selected,
  );
  const usable = (runs.data ?? []).filter((r) => r.n_personalized > 0);
  const datasetRuns = useData<PersonalizationRun[]>(
    `/personalization/experiments?dataset=${datasetKey}`,
  );
  const calibrated = new Set(
    (datasetRuns.data ?? [])
      .filter((r) => r.n_personalized > 0)
      .map((r) => r.global_experiment.id),
  );
  const run = usable.find((r) => r.id === runId) ?? usable[0];
  const hospitals = run?.hospitals ?? [];
  const chosenHospital =
    hospitals.find(
      (h) => h.hospital_id === hospital && h.status === "personalized",
    )?.hospital_id ??
    hospitals.find((h) => h.status === "personalized")?.hospital_id;
  const personal =
    mode === "personalized" && run && chosenHospital
      ? { run: run.id, hospital: chosenHospital }
      : null;
  if (data.error || models.error)
    return <RouteError error={(data.error ?? models.error)!} />;
  return (
    <>
      <Intro
        eyebrow="UNDERSTAND THE MODEL’S PERSPECTIVE"
        title="Risk prediction"
        description="Validated inputs. Saved preprocessing. Explanations computed from the actual model."
      />
      <Notice>
        Research outputs, not diagnoses. These probabilities are model estimates
        and are not clinically calibrated risk forecasts.
      </Notice>
      <div className="filters">
        <Field label="Prediction dataset">
          <select
            value={datasetKey}
            onChange={(e) => {
              setDatasetKey(e.target.value);
              setModel("");
              setResult(null);
            }}
          >
            {data.data?.map((d) => (
              <option key={d.key} value={d.key}>
                {d.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Saved model">
          <select
            value={selected ?? ""}
            onChange={(e) => {
              setModel(e.target.value);
              setResult(null);
              setRunId("");
              setHospital("");
            }}
          >
            {models.data?.map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
                {calibrated.has(m.id)
                  ? " · hospital calibration available"
                  : ""}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Prediction mode">
          <select
            value={personal ? "personalized" : "global"}
            onChange={(e) => {
              setMode(e.target.value as "global" | "personalized");
              setResult(null);
            }}
          >
            <option value="global">Global model</option>
            <option value="personalized" disabled={!usable.length}>
              Personalised (hospital calibration)
            </option>
          </select>
        </Field>
        {personal && run && (
          <>
            <Field label="Calibration run">
              <select
                value={run.id}
                onChange={(e) => {
                  setRunId(e.target.value);
                  setHospital("");
                  setResult(null);
                }}
              >
                {usable.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Hospital">
              <select
                value={chosenHospital}
                onChange={(e) => {
                  setHospital(e.target.value);
                  setResult(null);
                }}
              >
                {hospitals.map((h) => (
                  <option
                    key={h.hospital_id}
                    value={h.hospital_id}
                    disabled={h.status !== "personalized"}
                  >
                    {h.hospital_id.replace("_", " ")}
                    {h.status !== "personalized" ? " — not calibrated" : ""}
                  </option>
                ))}
              </select>
            </Field>
          </>
        )}
      </div>
      {selected && runs.isSuccess && !usable.length && (
        <Notice>
          Personalised mode needs a hospital calibration for the selected model,
          and this model has none.{" "}
          {calibrated.size
            ? `Choose a saved model marked “hospital calibration available” (${calibrated.size} for this dataset).`
            : "No model for this dataset has been calibrated yet; run scripts/run_personalization.py."}
        </Notice>
      )}
      {data.isPending || models.isPending ? (
        <Loading />
      ) : !selected ? (
        <Empty title="No compatible saved model">
          Train an experiment on this dataset before predicting.
        </Empty>
      ) : (
        <div className="prediction-layout">
          <Card>
            <CardHead
              title="Research input profile"
              sub={`${dataset?.n_features} features · ${dataset?.name}`}
              action={<Badge>Schema validated</Badge>}
            />
            {dataset && (
              <PatientForm
                key={`${datasetKey}-${selected}-${personal?.run ?? ""}-${personal?.hospital ?? ""}`}
                dataset={dataset}
                model={selected}
                onResult={setResult}
                personal={personal}
              />
            )}
          </Card>
          <div className="stack">
            {result ? (
              <>
                <Card>
                  <CardHead
                    title={
                      result.personal ? "Global model output" : "Model output"
                    }
                    sub={`Positive class: ${result.positive_label}`}
                    action={<Badge>Research estimate</Badge>}
                  />
                  <div className="probability">
                    <strong>{percent(result.probability_positive)}</strong>
                    <span>
                      model probability
                      <small>
                        Positive class · threshold {fmt(result.threshold, 2)}
                      </small>
                    </span>
                  </div>
                  <p className="result-label">
                    Classification: {result.predicted_label}
                  </p>
                  <p className="small muted">{result.notes.join(" ")}</p>
                  <p className="prediction-note">
                    Model <code>{result.experiment_id}</code>
                  </p>
                </Card>
                {result.personal && <CalibrationCard p={result.personal} />}
                <Card>
                  <CardHead
                    title="Why this prediction?"
                    sub={
                      result.personal
                        ? "SHAP contributions to the global model's log-odds. They do not explain the hospital calibration step."
                        : "SHAP contributions toward the positive class, in log-odds."
                    }
                  />
                  {result.explanation ? (
                    <>
                      <ShapChart data={result.explanation.contributions} />
                      <div className="inline small">
                        <span>− Teal: decreases score</span>
                        <span>+ Amber: increases score</span>
                      </div>
                      <Notice>
                        Baseline {fmt(result.explanation.base_value)} +
                        contributions = model output{" "}
                        {fmt(result.explanation.output_value)} log-odds.
                        Additivity is checked in Python.
                      </Notice>
                      <div className="table-scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>Feature</th>
                              <th>Input</th>
                              <th>Contribution</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.explanation.contributions.map((c) => (
                              <tr key={c.feature}>
                                <td>{c.label}</td>
                                <td>{c.value}</td>
                                <td>
                                  {c.shap >= 0 ? "+" : ""}
                                  {fmt(c.shap)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="small muted" style={{ marginTop: 16 }}>
                        {result.explanation.caveats.join(" ")}
                      </p>
                    </>
                  ) : (
                    <Notice error>
                      {result.explanation_error ?? "No explanation requested."}
                    </Notice>
                  )}
                </Card>
              </>
            ) : (
              <Card>
                <Empty title="An explanation begins with an input">
                  Complete the research profile and run the saved model to see
                  its prediction and feature contributions.
                </Empty>
                <ul className="checks">
                  <li>
                    <ShieldCheck size={16} />
                    Schema-validated patient features
                  </li>
                  <li>
                    <ShieldCheck size={16} />
                    Training-fitted preprocessing preserved
                  </li>
                  <li>
                    <ShieldCheck size={16} />
                    Actual SHAP values, computed locally
                  </li>
                </ul>
              </Card>
            )}
          </div>
        </div>
      )}
    </>
  );
}
