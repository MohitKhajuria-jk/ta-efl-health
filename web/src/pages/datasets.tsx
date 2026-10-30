import { Database, ArrowUpRight } from "lucide-react";
import { useData, percent } from "../api";
import type { Dataset } from "../types";
import {
  Intro,
  Card,
  CardHead,
  Badge,
  Loading,
  RouteError,
  Notice,
} from "../components/ui";
export default function Datasets() {
  const query = useData<Dataset[]>("/datasets");
  return (
    <>
      <Intro
        eyebrow="KNOW THE DATA BEFORE THE MODEL"
        title="Dataset library"
        description="Local validation, documented provenance, and transparent preprocessing."
      />
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <RouteError error={query.error} />
      ) : (
        <div className="dataset-grid">
          {query.data!.map((d) => (
            <Card className="dataset-card" key={d.key}>
              <CardHead
                title={d.name}
                action={
                  <Badge
                    tone={d.status.startsWith("valid") ? "success" : "error"}
                  >
                    {d.status}
                  </Badge>
                }
              />
              <p className="dataset-description">{d.description}</p>
              <div className="dataset-numbers">
                <div>
                  <strong>{d.rows?.toLocaleString() ?? "—"}</strong>
                  <span>Validated rows</span>
                </div>
                <div>
                  <strong>{d.n_features}</strong>
                  <span>Input features</span>
                </div>
                <div>
                  <strong>{percent(d.positive_rate)}</strong>
                  <span>Positive class</span>
                </div>
              </div>
              <div className="key-values">
                <div>
                  <span>Target</span>
                  <strong>{d.target}</strong>
                </div>
                <div>
                  <span>Positive label</span>
                  <strong>{d.positive_label}</strong>
                </div>
                <div>
                  <span>Duplicate rows</span>
                  <strong>{d.duplicates ?? "—"}</strong>
                </div>
              </div>
              <a
                className="text-link"
                href={d.source_url}
                target="_blank"
                rel="noreferrer"
                style={{ marginTop: 16 }}
              >
                Dataset source <ArrowUpRight size={14} />
              </a>
              <details>
                <summary>Quality, schema & provenance</summary>
                <p className="small muted">
                  {d.source} · {d.licence}
                </p>
                <ul className="notes">
                  {[...d.notes, ...(d.warnings ?? [])].map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
                <p className="small muted">
                  SHA-256:{" "}
                  <code>{d.sha256 ?? "Bundled with scikit-learn"}</code>
                </p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Type</th>
                        <th>Missing rows</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.features.map((f) => (
                        <tr key={f.name}>
                          <td>{f.label}</td>
                          <td>{f.kind}</td>
                          <td>{d.missing?.[f.name] ?? 0}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            </Card>
          ))}
        </div>
      )}
      <Notice>
        <Database size={16} /> Preprocessing is fitted from client training
        statistics only: numeric mean imputation and standardization,
        categorical mode imputation and schema-based one-hot encoding. Import or
        replace source files through the existing local dataset tools; the web
        API does not expose arbitrary filesystem writes.
      </Notice>
    </>
  );
}
