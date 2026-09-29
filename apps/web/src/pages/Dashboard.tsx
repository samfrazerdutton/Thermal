import { Link } from "react-router-dom";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { EvidenceChain, type EvidenceStep } from "../components/EvidenceChain";
import { Panel } from "../components/Panel";
import { VerdictBadge } from "../components/VerdictBadge";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export function Dashboard() {
  const runs = useApi(() => api.runs({ limit: 1 }), []);
  const experiments = useApi(() => api.experiments({ limit: 5 }), []);

  const loading = runs.status === "loading" || experiments.status === "loading";
  if (loading) return <LoadingState label="Loading latest evidence" />;
  if (runs.status === "error") return <ErrorState message={runs.message} />;
  if (experiments.status === "error") return <ErrorState message={experiments.message} />;

  const latestRun = runs.data[0];
  const latestExperiment = experiments.data[0];

  if (!latestRun) {
    return (
      <div className="flex flex-col gap-6">
        <Header />
        <EmptyState
          title="No evidence yet"
          detail="Run your first workload: thermal workload run matmul"
        />
      </div>
    );
  }

  const steps: EvidenceStep[] = [
    { label: "Workload", status: "done", detail: latestRun.workload_name },
    { label: "Baseline", status: Object.keys(latestRun.metrics).length ? "done" : "pending", detail: `n=${latestRun.measurement_iterations}` },
    {
      label: "Observation",
      status: latestRun.diagnosis ? "done" : "pending",
      detail: latestRun.diagnosis?.bottleneck.toLowerCase().replace(/_/g, " "),
    },
    { label: "Hypothesis", status: latestExperiment?.hypothesis ? "done" : "pending", detail: latestExperiment?.hypothesis ? "stated" : undefined },
    { label: "Intervention", status: latestExperiment ? "done" : "pending", detail: latestExperiment?.workload_name },
    { label: "Experiment", status: latestExperiment ? "done" : "pending", detail: latestExperiment ? `n=${latestExperiment.repetitions}` : undefined },
    {
      label: "Measurement",
      status: latestExperiment ? "done" : "pending",
      detail: latestExperiment ? `${latestExperiment.comparison.percent_change >= 0 ? "+" : ""}${latestExperiment.comparison.percent_change.toFixed(1)}%` : undefined,
    },
    {
      label: "Statistics",
      status: latestExperiment ? "done" : "pending",
      detail: latestExperiment ? `p=${latestExperiment.comparison.welch_p_value.toExponential(1)}` : undefined,
    },
    {
      label: "Verification",
      status: latestExperiment && latestExperiment.verdict !== "INCONCLUSIVE" ? "done" : "active",
      detail: latestExperiment?.verdict.toLowerCase(),
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      <Header />

      <Panel title="Latest evidence chain">
        <div className="overflow-x-auto pb-1">
          <EvidenceChain steps={steps} />
        </div>
      </Panel>

      <div className="grid grid-cols-2 gap-4">
        <Panel
          title="Most recent run"
          action={
            <Link to={`/runs/${latestRun.run_id}`} className="text-xs text-[var(--color-signal)] hover:underline">
              view
            </Link>
          }
        >
          <p className="font-mono text-sm">{latestRun.workload_name}</p>
          <p className="mt-1 text-xs text-[var(--color-text-dim)]">
            {latestRun.diagnosis ? latestRun.diagnosis.rationale : "no diagnosis recorded"}
          </p>
        </Panel>

        <Panel title="Recent experiments">
          {experiments.data.length === 0 ? (
            <p className="text-sm text-[var(--color-text-dim)]">None yet.</p>
          ) : (
            <div className="flex flex-col gap-2">
              {experiments.data.map((exp) => (
                <Link
                  key={exp.experiment_id}
                  to={`/experiments/${exp.experiment_id}`}
                  className="flex items-center justify-between border-b border-[var(--color-hairline)] py-1.5 text-sm last:border-b-0 hover:text-[var(--color-signal)]"
                >
                  <span>
                    {exp.workload_name} / {exp.metric_name}
                  </span>
                  <VerdictBadge verdict={exp.verdict} />
                </Link>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Header() {
  return (
    <div>
      <h1 className="text-lg text-[var(--color-text)]">Find the bottleneck. Prove it. Fix it.</h1>
      <p className="mt-1 text-sm text-[var(--color-text-dim)]">
        Every step below is real: it only shows &quot;done&quot; once this machine actually produced that evidence.
      </p>
    </div>
  );
}
