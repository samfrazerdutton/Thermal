import { Link } from "react-router-dom";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

function formatTimestamp(ns: number): string {
  return new Date(ns / 1e6).toLocaleString();
}

export function Runs() {
  const state = useApi(() => api.runs({ limit: 100 }), []);

  if (state.status === "loading") return <LoadingState label="Loading runs" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  if (state.data.length === 0) {
    return (
      <EmptyState
        title="No runs yet"
        detail="Run a workload from the CLI: thermal workload run matmul"
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg text-[var(--color-text)]">Runs</h1>
      <div className="overflow-x-auto border border-[var(--color-hairline)]" style={{ borderRadius: "var(--radius-panel)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-hairline)] text-left text-xs text-[var(--color-text-dim)]">
              <th className="px-3 py-2 font-normal">Workload</th>
              <th className="px-3 py-2 font-normal">Device</th>
              <th className="px-3 py-2 font-normal">Bottleneck</th>
              <th className="px-3 py-2 font-normal">n</th>
              <th className="px-3 py-2 font-normal">When</th>
            </tr>
          </thead>
          <tbody>
            {state.data.map((run) => (
              <tr key={run.run_id} className="border-b border-[var(--color-hairline)] last:border-b-0 hover:bg-[var(--color-surface-raised)]">
                <td className="px-3 py-2">
                  <Link to={`/runs/${run.run_id}`} className="text-[var(--color-signal)] hover:underline">
                    {run.workload_name}
                  </Link>
                </td>
                <td className="px-3 py-2 font-mono text-xs text-[var(--color-text-dim)]">{run.device}</td>
                <td className="px-3 py-2 font-mono text-xs">
                  {run.diagnosis?.bottleneck.toLowerCase().replace(/_/g, " ") ?? "—"}
                </td>
                <td className="px-3 py-2 font-mono text-xs tabular-nums">{run.measurement_iterations}</td>
                <td className="px-3 py-2 text-xs text-[var(--color-text-faint)]">{formatTimestamp(run.created_at_ns)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
