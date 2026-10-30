import { useState } from "react";
import { Link } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Square, ArrowUpRight } from "lucide-react";
import { useData, post, label, date } from "../api";
import type { Dataset, Settings, Job } from "../types";
import {
  Card,
  CardHead,
  Intro,
  Loading,
  RouteError,
  Notice,
  Empty,
  Status,
  Field,
} from "../components/ui";
import { Network } from "../components/network";
import { TrainingForm, type TrainingValues } from "../components/training-form";
import { TrustTable } from "./trust";
export default function Simulation({
  networkOnly = false,
}: {
  networkOnly?: boolean;
}) {
  const data = useData<Dataset[]>("/datasets");
  const settings = useData<Settings>("/settings");
  const jobs = useData<Job[]>("/federation", 1000);
  const [chosen, setChosen] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const qc = useQueryClient();
  const selected = jobs.data?.find((j) => j.id === chosen) ?? jobs.data?.[0];
  const detail = useData<Job>(
    `/federation/${selected?.id}`,
    (value) => (value?.status === "running" ? 1000 : false),
    !!selected,
  );
  if (data.isPending || settings.isPending || jobs.isPending)
    return <Loading />;
  const failure = data.error ?? settings.error ?? jobs.error ?? detail.error;
  if (failure)
    return (
      <RouteError
        error={failure}
        retry={() => {
          data.refetch();
          settings.refetch();
          jobs.refetch();
          detail.refetch();
        }}
      />
    );
  const job = detail.data ?? selected;
  const active = jobs.data?.find((j) => j.status === "running");
  async function start(v: TrainingValues) {
    setBusy(true);
    setError("");
    try {
      const j = await post<Job>("/federation", v);
      setChosen(j.id);
      await qc.invalidateQueries({ queryKey: ["/federation"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function cancel() {
    if (!job) return;
    setError("");
    try {
      await post(`/federation/${job.id}/cancel`);
      jobs.refetch();
      detail.refetch();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  return (
    <>
      <Intro
        eyebrow={
          networkOnly
            ? "COLLABORATIVE INTELLIGENCE"
            : "FROM CONFIGURATION TO EVIDENCE"
        }
        title={networkOnly ? "Hospital network" : "Federated learning"}
        description="A genuine Flower simulation, running within one Python process."
      />
      <div className={networkOnly ? "stack" : "split"}>
        {!networkOnly && (
          <Card>
            <CardHead
              title="Configure an experiment"
              sub="Every control is passed to the Python research engine."
            />
            <TrainingForm
              settings={settings.data!}
              datasets={data.data!}
              onSubmit={start}
              busy={busy || !!active}
              error={error}
            />
            {active && (
              <Notice>
                One experiment can run at a time. Cancel it or wait for
                completion before starting another.
              </Notice>
            )}
          </Card>
        )}
        <Card>
          <CardHead
            title="Simulation monitor"
            sub="Progress comes from actual engine callbacks."
            action={job && <Status value={job.status} />}
          />
          {jobs.data!.length > 1 && (
            <Field label="Monitored job">
              <select
                value={job?.id}
                onChange={(e) => setChosen(e.target.value)}
              >
                {jobs.data!.map((j) => (
                  <option key={j.id} value={j.id}>
                    {date(j.created)} · {j.config.dataset_key} · {j.status}
                  </option>
                ))}
              </select>
            </Field>
          )}
          {job ? (
            <>
              <div className="job-banner">
                <div>
                  <h3>{label(job.stage)}</h3>
                  <p>
                    {job.completed_rounds} / {job.config.num_rounds} rounds
                    aggregated · seed {job.config.seed}
                  </p>
                </div>
                {job.status === "running" && (
                  <button
                    className="button danger"
                    disabled={job.cancel_requested}
                    onClick={cancel}
                  >
                    <Square size={13} />
                    {job.cancel_requested ? "Stopping…" : "Cancel"}
                  </button>
                )}
              </div>
              <div
                className="progress-track"
                role="progressbar"
                aria-label="Aggregated rounds"
                aria-valuenow={job.completed_rounds}
                aria-valuemin={0}
                aria-valuemax={job.config.num_rounds}
              >
                <div
                  style={{
                    width: `${(job.completed_rounds / job.config.num_rounds) * 100}%`,
                  }}
                />
              </div>
              <div className="phase-list">
                {[
                  "initializing",
                  "local_training",
                  "aggregating",
                  "evaluating",
                  job.status === "running" ? "completed" : job.status,
                ].map((s, i) => (
                  <span
                    key={`${s}-${i}`}
                    className={job.stage === s ? "active" : ""}
                  >
                    {label(s)}
                  </span>
                ))}
              </div>
              {job.cancel_requested && job.status === "running" && (
                <Notice>
                  Cancellation is requested. The engine will stop at the next
                  safe round boundary; completed rounds can retain a partial
                  model.
                </Notice>
              )}
              {job.error && <Notice error>{job.error}</Notice>}
              <Network
                count={job.config.num_clients}
                job={job}
                round={job.rounds.at(-1)}
              />
              {job.experiment_id && (
                <Link
                  className="button secondary full"
                  to={`/app/experiments/${job.experiment_id}`}
                >
                  Inspect saved result <ArrowUpRight size={16} />
                </Link>
              )}
            </>
          ) : (
            <>
              <Network count={settings.data!.defaults.num_clients} />
              <Empty title="Ready for your first web run">
                Historical experiments remain in Experiment history. Start a
                simulation to monitor actual client events.
              </Empty>
            </>
          )}
          {networkOnly && error && <Notice error>{error}</Notice>}
        </Card>
      </div>
      {job && (
        <div className="stack" style={{ marginTop: 22 }}>
          {job.rounds.length > 0 && (
            <Card>
              <CardHead
                title="Latest client contributions"
                sub={`Round ${job.rounds.at(-1)!.round} · captured from the real aggregation strategy`}
              />
              <TrustTable round={job.rounds.at(-1)!} />
            </Card>
          )}
          <Card>
            <CardHead
              title="Engine event stream"
              sub="Timestamped backend events; a bounded live log of the most recent activity."
            />
            <div className="events" aria-label="Recent simulation events">
              {job.events
                .slice(-35)
                .reverse()
                .map((e) => (
                  <div key={e.sequence}>
                    <time>{new Date(e.received_at).toLocaleTimeString()}</time>
                    <span>
                      {e.message ??
                        `${label(e.type)}${e.round != null ? ` · round ${e.round}` : ""}${e.cid ? ` · ${e.cid}` : ""}`}
                    </span>
                  </div>
                ))}
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
