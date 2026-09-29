import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { api, type RunRecord } from "../lib/api";
import { useApi } from "../lib/useApi";

function groupByWorkload(runs: RunRecord[]): Map<string, RunRecord[]> {
  const groups = new Map<string, RunRecord[]>();
  for (const run of runs) {
    const list = groups.get(run.workload_name) ?? [];
    list.push(run);
    groups.set(run.workload_name, list);
  }
  return groups;
}

export function Benchmarks() {
  const state = useApi(() => api.runs({ limit: 500 }), []);

  if (state.status === "loading") return <LoadingState label="Loading benchmarks" />;
  if (state.status === "error") return <ErrorState message={state.message} />;
  if (state.data.length === 0) {
    return <EmptyState title="No benchmark data yet" detail="Runs accumulate here as you use the CLI or API." />;
  }

  const groups = groupByWorkload(state.data);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Benchmarks</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Grouped by workload, most recent first. Cross-GPU and cross-optimization comparison views land with Phase
          33.
        </p>
      </div>

      {[...groups.entries()].map(([workload, runs]) => (
        <Panel key={workload} title={workload}>
          <div className="flex flex-col gap-1.5">
            {runs.slice(0, 10).map((run) => {
              const firstMetric = Object.values(run.metrics)[0];
              return (
                <div key={run.run_id} className="flex items-center justify-between border-b border-[var(--color-hairline)] py-1.5 text-sm last:border-b-0">
                  <span className="font-mono text-xs text-[var(--color-text-faint)]">{run.run_id.slice(0, 8)}</span>
                  <span className="text-xs text-[var(--color-text-dim)]">{run.device}</span>
                  {firstMetric && (
                    <span className="font-mono text-xs">
                      {firstMetric.metric_name}: {firstMetric.mean.toPrecision(4)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </Panel>
      ))}
    </div>
  );
}
