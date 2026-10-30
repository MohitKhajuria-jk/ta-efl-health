import { useQuery } from "@tanstack/react-query";
import { post, fmt, strategyName } from "../api";
import type { Experiment } from "../types";
import { Modal, Loading, Notice } from "./ui";
import { RocChart } from "./charts";
export function Comparison({
  ids,
  open,
  onClose,
}: {
  ids: string[];
  open: boolean;
  onClose: () => void;
}) {
  const query = useQuery({
    queryKey: ["comparison", ids],
    queryFn: () =>
      post<{ items: Experiment[]; note: string }>("/experiments/compare", {
        ids,
      }),
    enabled: open && ids.length >= 2,
    retry: false,
  });
  return (
    <Modal
      open={open}
      onOpenChange={(v) => {
        if (!v) onClose();
      }}
      title="Compare experiments"
      description="Only runs with identical dataset versions and held-out split protocols can be compared."
    >
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <Notice error>{query.error.message}</Notice>
      ) : (
        query.data && (
          <>
            <RocChart experiments={query.data.items} />
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Run / strategy</th>
                    <th>ROC-AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {query.data.items.map((e) => (
                    <tr key={e.id}>
                      <td>
                        {strategyName(e.config.strategy)}
                        <small>{e.id}</small>
                      </td>
                      {(["roc_auc", "precision", "recall", "f1"] as const).map(
                        (k) => (
                          <td key={k}>{fmt(e.test_metrics[k])}</td>
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Notice>
              {query.data.note} Repeated-run uncertainty is not inferred from a
              few selected runs.
            </Notice>
          </>
        )
      )}
    </Modal>
  );
}
