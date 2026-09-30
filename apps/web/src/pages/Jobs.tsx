import { Link } from "react-router-dom";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

const STATUS_COLOR: Record<string, string> = {
  completed: "var(--color-verified)",
  failed: "var(--color-regression)",
  running: "var(--color-caution)",
  pending: "var(--color-text-faint)",
};

export function Jobs() {
  const state = useApi(() => api.jobs({ limit: 100 }), []);

  if (state.status === "loading") return <LoadingState label="Loading jobs" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  if (state.data.length === 0) {
    return (
      <div className="flex flex-col gap-6">
        <Header />
        <EmptyState
          title="No background jobs yet"
          detail={'Submit one without waiting for it: POST /api/jobs/workloads/run, or run a workload from the Workloads page and watch it stream live instead.'}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Header />
      <div className="overflow-x-auto border border-[var(--color-hairline)]" style={{ borderRadius: "var(--radius-panel)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-hairline)] text-left text-xs text-[var(--color-text-dim)]">
              <th className="px-3 py-2 font-normal">Kind</th>
              <th className="px-3 py-2 font-normal">Status</th>
              <th className="px-3 py-2 font-normal">Detail</th>
              <th className="px-3 py-2 font-normal">Job id</th>
            </tr>
          </thead>
          <tbody>
            {state.data.map((job) => (
              <tr key={job.job_id} className="border-b border-[var(--color-hairline)] last:border-b-0">
                <td className="px-3 py-2 font-mono text-xs">{job.kind}</td>
                <td className="px-3 py-2">
                  <span className="inline-flex items-center gap-1.5 font-mono text-xs" style={{ color: STATUS_COLOR[job.status] }}>
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: STATUS_COLOR[job.status] }} />
                    {job.status}
                  </span>
                </td>
                <td className="px-3 py-2 text-xs text-[var(--color-text-dim)]">
                  {job.status === "failed" && job.error}
                  {job.status === "completed" && job.result && "run_id" in job.result && (
                    <Link to={`/runs/${job.result.run_id}`} className="text-[var(--color-signal)] hover:underline">
                      view run
                    </Link>
                  )}
                  {job.status === "completed" && job.result && "experiment_id" in job.result && (
                    <Link to={`/experiments/${job.result.experiment_id}`} className="text-[var(--color-signal)] hover:underline">
                      view experiment
                    </Link>
                  )}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-[var(--color-text-faint)]">{job.job_id.slice(0, 8)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Header() {
  return (
    <div>
      <h1 className="text-lg text-[var(--color-text)]">Jobs</h1>
      <p className="mt-1 text-sm text-[var(--color-text-dim)]">
        Runs submitted through the background job queue (POST /api/jobs/...) — these continue on the server even if
        the browser that started them is closed. A run started from the Workloads page instead streams live over a
        WebSocket and won&apos;t show up here.
      </p>
    </div>
  );
}
