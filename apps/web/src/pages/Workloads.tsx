import { ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export function Workloads() {
  const state = useApi(api.workloads, []);

  if (state.status === "loading") return <LoadingState label="Loading workloads" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Workloads</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Run one from the CLI to generate a baseline: <code className="font-mono text-[var(--color-signal)]">thermal workload run &lt;name&gt;</code>
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {state.data.map((spec) => (
          <Panel key={spec.name} title={spec.name}>
            <p className="text-sm text-[var(--color-text-dim)]">{spec.description}</p>
            <dl className="mt-3 grid grid-cols-2 gap-y-1 text-xs">
              <dt className="text-[var(--color-text-faint)]">version</dt>
              <dd className="font-mono">{spec.version}</dd>
              <dt className="text-[var(--color-text-faint)]">warmup</dt>
              <dd className="font-mono">{spec.warmup_iterations} iterations</dd>
              <dt className="text-[var(--color-text-faint)]">measured</dt>
              <dd className="font-mono">{spec.measurement_iterations} iterations</dd>
              <dt className="text-[var(--color-text-faint)]">default params</dt>
              <dd className="font-mono">{JSON.stringify(spec.default_parameters)}</dd>
            </dl>
          </Panel>
        ))}
      </div>
    </div>
  );
}
