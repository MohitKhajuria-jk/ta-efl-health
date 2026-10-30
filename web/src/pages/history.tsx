import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ChevronLeft,
  ChevronRight,
  ArrowUpRight,
  GitCompareArrows,
} from "lucide-react";
import { useData, fmt, date, strategyName, label } from "../api";
import type { ExperimentPage, Dataset } from "../types";
import {
  Card,
  Intro,
  Field,
  Loading,
  RouteError,
  Status,
  Empty,
  Badge,
} from "../components/ui";
import { Comparison } from "../components/comparison";
export default function HistoryPage() {
  const [filters, setFilters] = useState({
    q: "",
    dataset: "",
    strategy: "",
    attack: "",
    sort: "newest",
  });
  const [page, setPage] = useState(1);
  const [ids, setIds] = useState<string[]>([]);
  const [compare, setCompare] = useState(false);
  const datasets = useData<Dataset[]>("/datasets");
  const params = new URLSearchParams({
    ...filters,
    page: String(page),
    page_size: "15",
  });
  const query = useData<ExperimentPage>(`/experiments?${params}`);
  function filter(k: keyof typeof filters, v: string) {
    setFilters((s) => ({ ...s, [k]: v }));
    setPage(1);
  }
  return (
    <>
      <Intro
        eyebrow="EVERY RUN, A RESEARCH RECORD"
        title="Experiment history"
        description="Search preserved experiments and inspect the exact model, configuration and evaluation."
      />
      <Card>
        <div className="filters">
          <Field label="Search run ID or label">
            <input
              placeholder="Search experiments…"
              value={filters.q}
              onChange={(e) => filter("q", e.target.value)}
            />
          </Field>
          <Field label="Dataset">
            <select
              value={filters.dataset}
              onChange={(e) => filter("dataset", e.target.value)}
            >
              <option value="">All datasets</option>
              {datasets.data?.map((d) => (
                <option key={d.key} value={d.key}>
                  {d.name}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Strategy">
            <select
              value={filters.strategy}
              onChange={(e) => filter("strategy", e.target.value)}
            >
              <option value="">All strategies</option>
              {["trust", "fedavg", "median"].map((s) => (
                <option key={s} value={s}>
                  {strategyName(s)}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Attack scenario">
            <select
              value={filters.attack}
              onChange={(e) => filter("attack", e.target.value)}
            >
              <option value="">All scenarios</option>
              {[
                "none",
                "sign_flip",
                "scaling",
                "gaussian_noise",
                "label_flip",
                "invalid_update",
              ].map((s) => (
                <option key={s} value={s}>
                  {label(s)}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Sort by">
            <select
              value={filters.sort}
              onChange={(e) => filter("sort", e.target.value)}
            >
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="auc">Test ROC-AUC</option>
            </select>
          </Field>
        </div>
        {query.isPending ? (
          <Loading />
        ) : query.error ? (
          <RouteError error={query.error} />
        ) : query.data!.items.length ? (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>
                      <span className="sr-only">Compare</span>
                    </th>
                    <th>Experiment</th>
                    <th>Dataset / strategy</th>
                    <th>Attack</th>
                    <th>Test AUC</th>
                    <th>Status</th>
                    <th>Origin</th>
                    <th>
                      <span className="sr-only">Open</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {query.data!.items.map((e) => (
                    <tr key={e.id}>
                      <td>
                        <input
                          type="checkbox"
                          aria-label={`Compare ${e.id}`}
                          checked={ids.includes(e.id)}
                          disabled={
                            !e.current ||
                            (!ids.includes(e.id) && ids.length >= 8)
                          }
                          onChange={(event) =>
                            setIds(
                              event.target.checked
                                ? [...ids, e.id]
                                : ids.filter((x) => x !== e.id),
                            )
                          }
                        />
                      </td>
                      <td>
                        <Link
                          className="table-title"
                          to={`/app/experiments/${e.id}`}
                        >
                          {e.config.label || e.id}
                        </Link>
                        <small>
                          {date(e.created)} · seed {e.config.seed}
                        </small>
                      </td>
                      <td>
                        {e.dataset.name}
                        <small>{strategyName(e.config.strategy)}</small>
                      </td>
                      <td>{label(e.config.attack_type)}</td>
                      <td>
                        {e.current ? fmt(e.test_metrics.roc_auc) : "Superseded"}
                      </td>
                      <td>
                        <Status value={e.status} />
                      </td>
                      <td>
                        <Badge>
                          {e.origin === "web" ? "Web run" : "Historical"}
                        </Badge>
                      </td>
                      <td>
                        <Link
                          to={`/app/experiments/${e.id}`}
                          className="icon-button"
                          aria-label={`Inspect ${e.id}`}
                        >
                          <ArrowUpRight size={17} />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <span>
                {query.data!.total.toLocaleString()} experiments · Page {page}{" "}
                of {Math.max(1, Math.ceil(query.data!.total / 15))}
              </span>
              <div className="inline">
                <button
                  className="button secondary"
                  disabled={page === 1}
                  aria-label="Previous page"
                  onClick={() => setPage((p) => p - 1)}
                >
                  <ChevronLeft size={15} />
                </button>
                <button
                  className="button secondary"
                  disabled={page * 15 >= query.data!.total}
                  aria-label="Next page"
                  onClick={() => setPage((p) => p + 1)}
                >
                  <ChevronRight size={15} />
                </button>
              </div>
            </div>
          </>
        ) : (
          <Empty title="No matching experiments">
            Change the filters or search term to find another run.
          </Empty>
        )}
      </Card>
      {ids.length > 0 && (
        <div className="comparison-bar">
          <GitCompareArrows size={18} />
          <span>{ids.length} selected · same split required</span>
          <button className="button glass" onClick={() => setIds([])}>
            Clear
          </button>
          <button
            className="button mint"
            disabled={ids.length < 2}
            onClick={() => setCompare(true)}
          >
            Compare results
          </button>
        </div>
      )}
      <Comparison ids={ids} open={compare} onClose={() => setCompare(false)} />
    </>
  );
}
