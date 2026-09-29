import { ErrorState, LoadingState } from "../components/AsyncState";
import { LiveRunPanel } from "../components/LiveRunPanel";
import { Panel } from "../components/Panel";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useWorkloadRunStream } from "../lib/useWorkloadRunStream";

export function Workloads() {
  const state = useApi(api.workloads, []);
  const stream = useWorkloadRunStream();

  if (state.status === "loading") return <LoadingState label="Loading workloads" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Workloads</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Run one live below, or from the CLI: <code className="font-mono text-[var(--color-signal)]">thermal workload run &lt;name&gt;</code>
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {state.data.map((spec) => (
          <Panel
            key={spec.name}
            title={spec.name}
            action={
              <button
                onClick={() => stream.start({ workload_name: spec.name, params: spec.default_parameters })}
                disabled={stream.status === "connecting" || stream.status === "running"}
                className="border border-[var(--color-hairline-strong)] px-2 py-1 text-xs text-[var(--color-text-dim)] hover:border-[var(--color-signal)] hover:text-[var(--color-signal)] disabled:opacity-40"
                style={{ borderRadius: "var(--radius-panel)" }}
              >
                run
              </button>
            }
          >
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

      <LiveRunPanel status={stream.status} events={stream.events} result={stream.result} error={stream.error} />
    </div>
  );
}
