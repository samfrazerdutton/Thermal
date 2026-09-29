import { Link } from "react-router-dom";
import { Panel } from "./Panel";
import type { StreamEvent } from "../lib/useWorkloadRunStream";

const EVENT_LABEL: Record<string, string> = {
  run_started: "started",
  warmup_started: "warmup started",
  warmup_iteration: "warmup",
  measurement_started: "measuring",
  iteration_completed: "iteration",
  telemetry_update: "telemetry",
  diagnosis_updated: "diagnosis computed",
  run_completed: "completed",
};

interface Props {
  status: "idle" | "connecting" | "running" | "done" | "error";
  events: StreamEvent[];
  result: { run_id: string; diagnosis: string } | null;
  error: string | null;
}

export function LiveRunPanel({ status, events, result, error }: Props) {
  if (status === "idle") return null;

  const latestTelemetry = [...events].reverse().find((e) => e.event === "telemetry_update");
  const iterationsDone = events.filter((e) => e.event === "iteration_completed").length;

  return (
    <Panel
      title={status === "running" || status === "connecting" ? "Running" : status === "done" ? "Completed" : "Failed"}
    >
      <div className="flex items-center gap-2">
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{
            background:
              status === "done"
                ? "var(--color-verified)"
                : status === "error"
                  ? "var(--color-regression)"
                  : "var(--color-caution)",
          }}
        />
        <span className="text-sm">
          {status === "connecting" && "connecting..."}
          {status === "running" && `${iterationsDone} iterations completed`}
          {status === "done" && result && (
            <>
              done — bottleneck: <span className="font-mono">{result.diagnosis.toLowerCase().replace(/_/g, " ")}</span>
            </>
          )}
          {status === "error" && (error ?? "unknown error")}
        </span>
      </div>

      {latestTelemetry && (
        <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
          {latestTelemetry.cpu_utilization_percent !== null && (
            <span className="font-mono">cpu {(latestTelemetry.cpu_utilization_percent as number).toFixed(0)}%</span>
          )}
          {latestTelemetry.gpu_utilization_percent !== null && latestTelemetry.gpu_utilization_percent !== undefined && (
            <span className="font-mono">gpu {(latestTelemetry.gpu_utilization_percent as number).toFixed(0)}%</span>
          )}
          {latestTelemetry.gpu_temperature_c !== null && latestTelemetry.gpu_temperature_c !== undefined && (
            <span className="font-mono">{(latestTelemetry.gpu_temperature_c as number).toFixed(0)}C</span>
          )}
        </div>
      )}

      {status === "done" && result && (
        <Link to={`/runs/${result.run_id}`} className="mt-3 inline-block text-sm text-[var(--color-signal)] hover:underline">
          view evidence
        </Link>
      )}

      {events.length > 0 && (
        <div className="mt-3 flex flex-col gap-0.5 border-t border-[var(--color-hairline)] pt-3">
          {events
            .filter((e) => e.event !== "telemetry_update")
            .slice(-6)
            .map((e, i) => (
              <span key={i} className="font-mono text-xs text-[var(--color-text-faint)]">
                {EVENT_LABEL[e.event] ?? e.event}
              </span>
            ))}
        </div>
      )}
    </Panel>
  );
}
