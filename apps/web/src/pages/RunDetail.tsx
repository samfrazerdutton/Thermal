import { useParams } from "react-router-dom";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { StatValue } from "../components/StatValue";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export function RunDetail({ focusDiagnosis = false }: { focusDiagnosis?: boolean }) {
  const { id } = useParams<{ id: string }>();
  const state = useApi(() => api.run(id!), [id]);

  if (state.status === "loading") return <LoadingState label="Loading run" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const run = state.data;
  const diagnosis = run.diagnosis;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">{run.workload_name}</h1>
        <p className="mt-1 font-mono text-xs text-[var(--color-text-faint)]">
          run {run.run_id} on {run.device}
          {run.git_commit && ` · commit ${run.git_commit.slice(0, 7)}`}
        </p>
      </div>

      {diagnosis && (
        <Panel title="Root cause">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-xl" style={{ color: "var(--color-caution)" }}>
              {diagnosis.bottleneck.replace(/_/g, " ").toLowerCase()}
            </span>
            <span className="text-sm text-[var(--color-text-dim)]">{Math.round(diagnosis.confidence * 100)}% confidence</span>
          </div>

          {diagnosis.evidence.length > 0 && (
            <div className="mt-4">
              <p className="text-xs text-[var(--color-text-dim)]">Evidence</p>
              <div className="mt-1.5 flex flex-col gap-1">
                {diagnosis.evidence.map((ev) => (
                  <div key={ev.feature} className="flex items-center justify-between border-b border-[var(--color-hairline)] py-1.5 text-sm last:border-b-0">
                    <span className="text-[var(--color-text-dim)]">{ev.feature.replace(/_/g, " ")}</span>
                    <span className="font-mono text-xs">
                      {ev.value.toFixed(2)} ({ev.comparison} {ev.threshold})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <p className="text-xs text-[var(--color-text-dim)]">Why?</p>
            <p className="mt-1 text-sm leading-relaxed">{diagnosis.rationale}</p>
          </div>

          {diagnosis.missing_features.length > 0 && (
            <p className="mt-3 text-xs text-[var(--color-text-faint)]">
              Not measurable on this run: {diagnosis.missing_features.join(", ").replace(/_/g, " ")}
            </p>
          )}
        </Panel>
      )}

      <div className="grid grid-cols-3 gap-4">
        {Object.values(run.metrics).map((stats) => (
          <Panel key={stats.metric_name} title={stats.metric_name.replace(/_/g, " ")}>
            <StatValue
              label="mean"
              value={stats.mean.toPrecision(4)}
              tone={stats.high_variance_warning ? "caution" : "default"}
            />
            <dl className="mt-3 grid grid-cols-2 gap-y-1 text-xs">
              <dt className="text-[var(--color-text-faint)]">median</dt>
              <dd className="font-mono">{stats.median.toPrecision(4)}</dd>
              <dt className="text-[var(--color-text-faint)]">p95</dt>
              <dd className="font-mono">{stats.p95.toPrecision(4)}</dd>
              <dt className="text-[var(--color-text-faint)]">CV</dt>
              <dd className="font-mono">{(stats.coefficient_of_variation * 100).toFixed(1)}%</dd>
              <dt className="text-[var(--color-text-faint)]">n</dt>
              <dd className="font-mono">{stats.n}</dd>
            </dl>
            {stats.high_variance_warning && (
              <p className="mt-2 text-xs" style={{ color: "var(--color-caution)" }}>
                high variance — treat cautiously
              </p>
            )}
          </Panel>
        ))}
      </div>

      <Panel title="Configuration">
        <pre className="overflow-x-auto font-mono text-xs text-[var(--color-text-dim)]">
          {JSON.stringify(run.configuration, null, 2)}
        </pre>
      </Panel>

      {!focusDiagnosis && run.telemetry_path && (
        <p className="text-xs text-[var(--color-text-faint)]">Raw telemetry: {run.telemetry_path}</p>
      )}
    </div>
  );
}
