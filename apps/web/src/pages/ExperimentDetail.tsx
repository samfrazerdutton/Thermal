import { useParams } from "react-router-dom";
import { CartesianGrid, Legend, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { StatValue } from "../components/StatValue";
import { VerdictBadge } from "../components/VerdictBadge";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export function ExperimentDetail() {
  const { id } = useParams<{ id: string }>();
  const state = useApi(() => api.experiment(id!), [id]);

  if (state.status === "loading") return <LoadingState label="Loading experiment" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const exp = state.data;
  const cmp = exp.comparison;

  const baselinePoints = exp.baseline_values.map((v, i) => ({ i, value: v }));
  const treatmentPoints = exp.treatment_values.map((v, i) => ({ i, value: v }));

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">{exp.workload_name}</h1>
        {exp.hypothesis && <p className="mt-1 text-sm text-[var(--color-text-dim)]">{exp.hypothesis}</p>}
      </div>

      <div className="grid grid-cols-4 gap-4">
        <Panel>
          <StatValue label="Control" value={cmp.baseline_mean.toPrecision(4)} />
        </Panel>
        <Panel>
          <StatValue label="Treatment" value={cmp.treatment_mean.toPrecision(4)} />
        </Panel>
        <Panel>
          <StatValue
            label="Change"
            value={`${cmp.percent_change >= 0 ? "+" : ""}${cmp.percent_change.toFixed(1)}%`}
            tone={cmp.verdict === "IMPROVED" ? "verified" : cmp.verdict === "REGRESSED" ? "regression" : "default"}
          />
          {cmp.percent_change_ci_low !== null && (
            <p className="mt-1 text-xs text-[var(--color-text-faint)]">
              {Math.round(cmp.confidence_level * 100)}% CI [{cmp.percent_change_ci_low.toFixed(1)}%, {cmp.percent_change_ci_high!.toFixed(1)}%]
            </p>
          )}
        </Panel>
        <Panel>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-dim)]">Verdict</span>
            <VerdictBadge verdict={cmp.verdict} />
          </div>
        </Panel>
      </div>

      <Panel title="Raw samples">
        <ScatterChart width={640} height={280} margin={{ left: 8, right: 8 }}>
          <CartesianGrid stroke="var(--color-hairline)" />
          <XAxis
            type="number"
            dataKey="i"
            name="trial"
            allowDuplicatedCategory={false}
            stroke="var(--color-text-faint)"
            tick={{ fontSize: 11, fill: "var(--color-text-dim)" }}
          />
          <YAxis
            type="number"
            dataKey="value"
            name={exp.metric_name}
            stroke="var(--color-text-faint)"
            tick={{ fontSize: 11, fill: "var(--color-text-dim)" }}
          />
          <Tooltip
            contentStyle={{ background: "var(--color-surface-raised)", border: "1px solid var(--color-hairline)", fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Scatter name="control" data={baselinePoints} fill="var(--color-signal)" />
          <Scatter name="treatment" data={treatmentPoints} fill="var(--color-verified)" />
        </ScatterChart>
      </Panel>

      <div className="grid grid-cols-2 gap-4">
        <Panel title="Statistics">
          <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
            <dt className="text-[var(--color-text-dim)]">Welch&apos;s t-test p-value</dt>
            <dd className="font-mono">{cmp.welch_p_value.toExponential(2)}</dd>
            {cmp.mannwhitney_p_value !== null && (
              <>
                <dt className="text-[var(--color-text-dim)]">Mann-Whitney U p-value</dt>
                <dd className="font-mono">{cmp.mannwhitney_p_value.toExponential(2)}</dd>
              </>
            )}
            <dt className="text-[var(--color-text-dim)]">Effect size (Cohen&apos;s d)</dt>
            <dd className="font-mono">{cmp.cohens_d.toFixed(2)}</dd>
            <dt className="text-[var(--color-text-dim)]">n</dt>
            <dd className="font-mono">
              {cmp.baseline_n} control, {cmp.treatment_n} treatment
            </dd>
          </dl>
          {cmp.warnings.length > 0 && (
            <div className="mt-3 flex flex-col gap-1">
              {cmp.warnings.map((w) => (
                <p key={w} className="text-xs" style={{ color: "var(--color-caution)" }}>
                  {w}
                </p>
              ))}
            </div>
          )}
        </Panel>

        <Panel title="Configuration">
          <p className="text-xs text-[var(--color-text-dim)]">Baseline</p>
          <pre className="mb-3 font-mono text-xs">{JSON.stringify(exp.baseline_config, null, 2)}</pre>
          <p className="text-xs text-[var(--color-text-dim)]">Treatment</p>
          <pre className="font-mono text-xs">{JSON.stringify(exp.treatment_config, null, 2)}</pre>
        </Panel>
      </div>
    </div>
  );
}
