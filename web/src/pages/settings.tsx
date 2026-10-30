import { useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { api, useData } from "../api";
import type { Settings, Dataset } from "../types";
import {
  Intro,
  Card,
  CardHead,
  Loading,
  RouteError,
  Notice,
} from "../components/ui";
import { TrainingForm, type TrainingValues } from "../components/training-form";
import { ThemeOptions } from "../components/theme-control";
export default function SettingsPage() {
  const query = useData<Settings>("/settings");
  const data = useData<Dataset[]>("/datasets");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  async function save(defaults: TrainingValues) {
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      await api("/settings", {
        method: "PUT",
        body: JSON.stringify({ defaults }),
      });
      await query.refetch();
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (query.isPending || data.isPending) return <Loading />;
  if (query.error || data.error)
    return <RouteError error={(query.error ?? data.error)!} />;
  const s = query.data!;
  return (
    <>
      <Intro
        eyebrow="YOUR RESEARCH ENVIRONMENT"
        title="Settings & system"
        description="Reusable experiment defaults and a clear view of the local execution environment."
      />
      <div className="split">
        <Card>
          <CardHead
            title="Web experiment defaults"
            sub="Saved separately from historical configurations and desktop settings."
          />
          {saved && (
            <Notice>
              Defaults saved. New web experiments will use this configuration.
            </Notice>
          )}
          <TrainingForm
            settings={s}
            datasets={data.data!}
            onSubmit={save}
            busy={busy}
            error={error}
            save
          />
        </Card>
        <div className="stack">
          <Card>
            <CardHead
              title="Appearance"
              sub="Choose a comfortable view for your workspace."
            />
            <ThemeOptions />
            <p className="small muted theme-help">
              Applied immediately and remembered in this browser. System follows
              your device's appearance.
            </p>
          </Card>
          <Card>
            <CardHead title="System information" />
            <div className="key-values">
              {[
                ["Python", s.python],
                ["Research pipeline", `v${s.pipeline_version}`],
                ["Execution", s.execution],
                ["Data", s.data_location],
                ["Artifacts", s.storage_location],
                ["Sessions", s.sessions],
              ].map(([k, v]) => (
                <div key={k}>
                  <span>{k}</span>
                  <strong>{v}</strong>
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <CardHead title="Research integrity" />
            <ul className="checks">
              <li>
                <CheckCircle2 size={16} />
                Training-only preprocessing statistics
              </li>
              <li>
                <CheckCircle2 size={16} />
                Fixed threshold before evaluation
              </li>
              <li>
                <CheckCircle2 size={16} />
                Test data excluded from trust scoring
              </li>
              <li>
                <CheckCircle2 size={16} />
                Saved model and schema verified at prediction
              </li>
              <li>
                <CheckCircle2 size={16} />
                Historical records preserved without relabeling
              </li>
            </ul>
            <Notice>{s.privacy}</Notice>
            <p className="small muted">
              The trust policy assumes an honest majority and does not detect
              every attack. This application is a single-workstation research
              tool with shared local accounts, not a clinical system or
              distributed hospital deployment.
            </p>
          </Card>
          <Card>
            <CardHead title="Diagnostics & account recovery" />
            <p className="small muted">
              Simulation events and failures are available on Federated
              learning. Detailed diagnostics are recorded in the local Python
              server output. Existing local accounts are reused; account
              recovery remains an explicit local workstation operation. No
              password reset or filesystem browser is exposed over HTTP.
            </p>
          </Card>
        </div>
      </div>
    </>
  );
}
