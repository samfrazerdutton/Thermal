import { Link } from "react-router-dom";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { VerdictBadge } from "../components/VerdictBadge";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export function Experiments() {
  const state = useApi(() => api.experiments({ limit: 100 }), []);

  if (state.status === "loading") return <LoadingState label="Loading experiments" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  if (state.data.length === 0) {
    return (
      <EmptyState
        title="No experiments yet"
        detail="Run one from the CLI: thermal experiment run matmul --baseline-param size=256 --treatment-param size=512 --metric gflops"
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg text-[var(--color-text)]">Experiments</h1>
      <div className="overflow-x-auto border border-[var(--color-hairline)]" style={{ borderRadius: "var(--radius-panel)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-hairline)] text-left text-xs text-[var(--color-text-dim)]">
              <th className="px-3 py-2 font-normal">Workload</th>
              <th className="px-3 py-2 font-normal">Metric</th>
              <th className="px-3 py-2 font-normal">Change</th>
              <th className="px-3 py-2 font-normal">Verdict</th>
              <th className="px-3 py-2 font-normal">n</th>
            </tr>
          </thead>
          <tbody>
            {state.data.map((exp) => (
              <tr key={exp.experiment_id} className="border-b border-[var(--color-hairline)] last:border-b-0 hover:bg-[var(--color-surface-raised)]">
                <td className="px-3 py-2">
                  <Link to={`/experiments/${exp.experiment_id}`} className="text-[var(--color-signal)] hover:underline">
                    {exp.workload_name}
                  </Link>
                </td>
                <td className="px-3 py-2 font-mono text-xs text-[var(--color-text-dim)]">{exp.metric_name}</td>
                <td className="px-3 py-2 font-mono text-xs tabular-nums">
                  {exp.comparison.percent_change >= 0 ? "+" : ""}
                  {exp.comparison.percent_change.toFixed(1)}%
                </td>
                <td className="px-3 py-2">
                  <VerdictBadge verdict={exp.verdict} />
                </td>
                <td className="px-3 py-2 font-mono text-xs tabular-nums">{exp.repetitions}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
