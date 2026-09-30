import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type Intervention } from "../lib/api";

// One counterfactual hypothesis (thermal/counterfactual.py) with a real,
// ready-to-submit experiment attached -- or, if the workload has no matching
// parameter, an honest explanation of why not, never a button that would fail.
export function InterventionCard({ intervention }: { intervention: Intervention }) {
  const [state, setState] = useState<{ status: "idle" | "submitting" | "polling" | "done" | "error"; jobId?: string; result?: Record<string, unknown>; message?: string }>({
    status: "idle",
  });

  async function runExperiment() {
    if (!intervention.experiment_request) return;
    setState({ status: "submitting" });
    try {
      const { job_id } = await api.submitExperimentJob(intervention.experiment_request);
      setState({ status: "polling", jobId: job_id });
      poll(job_id);
    } catch (err) {
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  function poll(jobId: string) {
    const interval = setInterval(async () => {
      const job = await api.job(jobId);
      if (job.status === "completed") {
        clearInterval(interval);
        setState({ status: "done", jobId, result: job.result ?? undefined });
      } else if (job.status === "failed") {
        clearInterval(interval);
        setState({ status: "error", jobId, message: job.error ?? "job failed" });
      }
    }, 500);
  }

  return (
    <div className="border border-[var(--color-hairline)] p-3" style={{ borderRadius: "var(--radius-panel)" }}>
      <p className="text-sm text-[var(--color-text)]">{intervention.description}</p>

      {!intervention.actionable && (
        <p className="mt-2 text-xs text-[var(--color-text-faint)]">Not testable here: {intervention.reason_not_actionable}</p>
      )}

      {intervention.actionable && intervention.experiment_request && (
        <div className="mt-3">
          <p className="mb-2 font-mono text-xs text-[var(--color-text-dim)]">
            {intervention.independent_variable}: {JSON.stringify(intervention.experiment_request.baseline_params[intervention.independent_variable])}
            {" -> "}
            {JSON.stringify(intervention.proposed_treatment_params?.[intervention.independent_variable])}
          </p>

          {state.status === "idle" && (
            <button
              onClick={runExperiment}
              className="border border-[var(--color-hairline-strong)] px-3 py-1 text-xs text-[var(--color-text-dim)] hover:border-[var(--color-signal)] hover:text-[var(--color-signal)]"
              style={{ borderRadius: "var(--radius-panel)" }}
            >
              run this experiment
            </button>
          )}

          {(state.status === "submitting" || state.status === "polling") && (
            <p className="text-xs text-[var(--color-caution)]">
              <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: "var(--color-caution)" }} />{" "}
              {state.status === "submitting" ? "submitting..." : "running..."}
            </p>
          )}

          {state.status === "error" && <p className="text-xs text-[var(--color-regression)]">{state.message}</p>}

          {state.status === "done" && state.result && (
            <div className="text-xs">
              <span
                className="font-mono"
                style={{
                  color:
                    state.result.verdict === "IMPROVED"
                      ? "var(--color-verified)"
                      : state.result.verdict === "REGRESSED"
                        ? "var(--color-regression)"
                        : "var(--color-caution)",
                }}
              >
                {String(state.result.verdict).toLowerCase()}
              </span>{" "}
              <Link to={`/experiments/${state.result.experiment_id}`} className="text-[var(--color-signal)] hover:underline">
                view result
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
