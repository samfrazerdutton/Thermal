import { ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { StatValue } from "../components/StatValue";
import { api, type Capability } from "../lib/api";
import { useApi } from "../lib/useApi";

function formatCapabilityValue(value: unknown): string {
  if (value === null || value === undefined) return "yes";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function CapabilityRow({ capability }: { capability: Capability }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-[var(--color-hairline)] py-2 text-sm last:border-b-0">
      <span className="shrink-0 text-[var(--color-text-dim)]">{capability.name.replace(/_/g, " ")}</span>
      <span
        className="truncate text-right font-mono"
        style={{ color: capability.available ? "var(--color-verified)" : "var(--color-text-faint)" }}
      >
        {capability.available ? formatCapabilityValue(capability.value) : `unavailable — ${capability.reason}`}
      </span>
    </div>
  );
}

export function Hardware() {
  const state = useApi(api.hardware, []);

  if (state.status === "loading") return <LoadingState label="Detecting hardware" />;
  if (state.status === "error") return <ErrorState message={state.message} />;

  const report = state.data;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Hardware</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Detected live on this machine. Nothing here is estimated — a counter that can&apos;t be read is marked
          unavailable with a reason, not filled in.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <Panel>
          <StatValue label="CPU" value={report.cpu.logical_cores ?? "?"} unit="threads" />
          <p className="mt-2 text-xs text-[var(--color-text-faint)]">{report.cpu.name}</p>
        </Panel>
        <Panel>
          {report.gpu.available ? (
            <>
              <StatValue label="GPU memory" value={Math.round(report.gpu.memory_total_mb ?? 0)} unit="MB" />
              <p className="mt-2 text-xs text-[var(--color-text-faint)]">{report.gpu.name}</p>
            </>
          ) : (
            <StatValue label="GPU" value="none detected" tone="caution" />
          )}
        </Panel>
        <Panel>
          <StatValue label="CUDA driver" value={report.gpu.cuda_driver_version ?? "n/a"} />
          <p className="mt-2 text-xs text-[var(--color-text-faint)]">driver {report.gpu.driver_version ?? "n/a"}</p>
        </Panel>
      </div>

      {report.gpu.available && (
        <Panel title="GPU telemetry">
          {Object.values(report.gpu.telemetry).map((cap) => (
            <CapabilityRow key={cap.name} capability={cap} />
          ))}
        </Panel>
      )}

      <Panel title="Toolchain">
        {Object.values(report.toolchain).map((cap) => (
          <CapabilityRow key={cap.name} capability={cap} />
        ))}
      </Panel>

      <Panel title="Software">
        <CapabilityRow capability={{ name: "python", available: true, value: report.software.python_version, reason: null }} />
        <CapabilityRow capability={report.software.torch_installed} />
        <CapabilityRow capability={report.software.torch_cuda} />
      </Panel>
    </div>
  );
}
