import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ErrorState, LoadingState } from "../components/AsyncState";
import { InterventionCard } from "../components/InterventionCard";
import { Panel } from "../components/Panel";
import { ParameterForm } from "../components/ParameterForm";
import { api, type RunRecord, type InterventionsResponse } from "../lib/api";
import { useApi } from "../lib/useApi";
import { useWorkloadRunStream } from "../lib/useWorkloadRunStream";

const EVENT_LABEL: Record<string, string> = {
  run_started: "run started",
  warmup_started: "warmup started",
  warmup_iteration: "warmup iteration",
  measurement_started: "measuring",
  iteration_completed: "iteration completed",
  telemetry_update: "telemetry sample",
  diagnosis_updated: "diagnosis computed",
  run_completed: "run completed",
};

function formatClock(ns: unknown): string {
  if (typeof ns !== "number") return "";
  return new Date(ns / 1e6).toLocaleTimeString(undefined, { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function Investigate() {
  const workloadsState = useApi(api.workloads, []);
  const hardwareState = useApi(api.hardware, []);
  const stream = useWorkloadRunStream();

  const [selected, setSelected] = useState<string>("");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [samples, setSamples] = useState(10);
  const [warmup, setWarmup] = useState(3);

  const [runDetail, setRunDetail] = useState<RunRecord | null>(null);
  const [interventions, setInterventions] = useState<InterventionsResponse | null>(null);
  const [postRunError, setPostRunError] = useState<string | null>(null);

  useEffect(() => {
    if (workloadsState.status === "ready" && !selected) {
      const first = workloadsState.data[0];
      if (first) {
        setSelected(first.name);
        setParams(first.default_parameters);
        setWarmup(first.warmup_iterations);
        setSamples(first.measurement_iterations);
      }
    }
  }, [workloadsState, selected]);

  useEffect(() => {
    if (stream.status !== "done" || !stream.result) return;
    setPostRunError(null);
    api
      .run(stream.result.run_id)
      .then(setRunDetail)
      .catch((err) => setPostRunError(err instanceof Error ? err.message : String(err)));
    api
      .interventions(stream.result.run_id)
      .then(setInterventions)
      .catch((err) => setPostRunError(err instanceof Error ? err.message : String(err)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.status, stream.result]);

  function handleWorkloadChange(name: string) {
    setSelected(name);
    const spec = workloadsState.status === "ready" ? workloadsState.data.find((w) => w.name === name) : undefined;
    if (spec) {
      setParams(spec.default_parameters);
      setWarmup(spec.warmup_iterations);
      setSamples(spec.measurement_iterations);
    }
    setRunDetail(null);
    setInterventions(null);
  }

  function handleRun() {
    setRunDetail(null);
    setInterventions(null);
    setPostRunError(null);
    stream.start({ workload_name: selected, params, samples, warmup });
  }

  if (workloadsState.status === "loading" || hardwareState.status === "loading") return <LoadingState label="Loading investigation workspace" />;
  if (workloadsState.status === "error") return <ErrorState message={workloadsState.message} />;
  if (hardwareState.status === "error") return <ErrorState message={hardwareState.message} />;

  const gpuAvailable = hardwareState.data.gpu.available;
  const running = stream.status === "connecting" || stream.status === "running";
  const requestedCuda = params.device === "cuda:0";
  const willFail = requestedCuda && !gpuAvailable;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Investigate</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Configure a workload, run it, and see the bottleneck evidence and a testable intervention in one place.
        </p>
      </div>

      <div className="grid grid-cols-[280px_1fr_360px] gap-4 max-lg:grid-cols-1">
        {/* Left: workload + parameters */}
        <Panel title="Workload">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-dim)]">workload</span>
            <select
              value={selected}
              disabled={running}
              onChange={(e) => handleWorkloadChange(e.target.value)}
              className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
              style={{ borderRadius: "var(--radius-panel)" }}
            >
              {workloadsState.data.map((w) => (
                <option key={w.name} value={w.name}>
                  {w.name}
                </option>
              ))}
            </select>
          </label>

          <div className="mt-3">
            <ParameterForm params={params} onChange={setParams} gpuAvailable={gpuAvailable} disabled={running} />
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-[var(--color-text-dim)]">warmup</span>
              <input
                type="number"
                min={0}
                value={warmup}
                disabled={running}
                onChange={(e) => setWarmup(Number(e.target.value))}
                className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
                style={{ borderRadius: "var(--radius-panel)" }}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-[var(--color-text-dim)]">measured</span>
              <input
                type="number"
                min={1}
                value={samples}
                disabled={running}
                onChange={(e) => setSamples(Number(e.target.value))}
                className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
                style={{ borderRadius: "var(--radius-panel)" }}
              />
            </label>
          </div>

          {willFail && (
            <p className="mt-3 text-xs" style={{ color: "var(--color-regression)" }}>
              This will fail: no CUDA-capable GPU detected on this machine (see Hardware).
            </p>
          )}

          <button
            onClick={handleRun}
            disabled={running || !selected || willFail}
            className="mt-4 w-full border px-3 py-1.5 text-sm disabled:opacity-40"
            style={{
              borderRadius: "var(--radius-panel)",
              borderColor: "var(--color-signal)",
              color: "var(--color-signal)",
            }}
          >
            {running ? "running..." : "run"}
          </button>
        </Panel>

        {/* Center: live execution timeline */}
        <Panel title="Execution timeline">
          {stream.status === "idle" && (
            <p className="text-sm text-[var(--color-text-dim)]">Configure a workload on the left and click run.</p>
          )}
          {stream.status === "error" && <p className="text-sm text-[var(--color-regression)]">{stream.error}</p>}
          {stream.status !== "idle" && (
            <div className="flex max-h-[520px] flex-col gap-1 overflow-y-auto">
              {stream.events.map((event, i) => (
                <div key={i} className="flex items-center gap-3 border-b border-[var(--color-hairline)] py-1.5 text-xs last:border-b-0">
                  <span className="font-mono text-[var(--color-text-faint)]">{formatClock(event.timestamp_ns)}</span>
                  <span className="text-[var(--color-text)]">{EVENT_LABEL[event.event] ?? event.event}</span>
                  {event.event === "telemetry_update" && event.gpu_utilization_percent !== null && event.gpu_utilization_percent !== undefined && (
                    <span className="font-mono text-[var(--color-text-dim)]">gpu {(event.gpu_utilization_percent as number).toFixed(0)}%</span>
                  )}
                  {event.event === "iteration_completed" && (
                    <span className="font-mono text-[var(--color-text-dim)]">#{(event.index as number) + 1}</span>
                  )}
                </div>
              ))}
              {stream.status === "done" && (
                <p className="pt-2 text-xs text-[var(--color-verified)]">
                  completed —{" "}
                  <Link to={`/runs/${stream.result?.run_id}`} className="text-[var(--color-signal)] hover:underline">
                    view full run
                  </Link>
                </p>
              )}
            </div>
          )}
        </Panel>

        {/* Right: evidence + interventions */}
        <div className="flex flex-col gap-4">
          <Panel title="Bottleneck evidence">
            {!runDetail && stream.status !== "done" && <p className="text-sm text-[var(--color-text-dim)]">No result yet.</p>}
            {postRunError && <ErrorState message={postRunError} />}
            {runDetail?.diagnosis && (
              <div>
                <p className="font-mono text-base" style={{ color: "var(--color-caution)" }}>
                  {runDetail.diagnosis.bottleneck.replace(/_/g, " ").toLowerCase()}
                </p>
                <p className="mt-1 text-xs text-[var(--color-text-dim)]">{Math.round(runDetail.diagnosis.confidence * 100)}% confidence</p>
                <p className="mt-2 text-sm leading-relaxed">{runDetail.diagnosis.rationale}</p>
                {runDetail.diagnosis.evidence.length > 0 && (
                  <div className="mt-3 flex flex-col gap-1">
                    {runDetail.diagnosis.evidence.map((ev) => (
                      <div key={ev.feature} className="flex items-center justify-between text-xs">
                        <span className="text-[var(--color-text-dim)]">{ev.feature.replace(/_/g, " ")}</span>
                        <span className="font-mono">
                          {ev.value.toFixed(2)} ({ev.comparison} {ev.threshold})
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {runDetail.diagnosis.missing_features.length > 0 && (
                  <p className="mt-3 text-xs text-[var(--color-text-faint)]">
                    Not measurable: {runDetail.diagnosis.missing_features.join(", ").replace(/_/g, " ")}
                  </p>
                )}
              </div>
            )}
          </Panel>

          <Panel title="Suggested interventions">
            {!interventions && stream.status !== "done" && <p className="text-sm text-[var(--color-text-dim)]">No result yet.</p>}
            {interventions && interventions.interventions.length === 0 && (
              <p className="text-sm text-[var(--color-text-dim)]">No intervention strategy registered for this bottleneck class yet.</p>
            )}
            {interventions && (
              <div className="flex flex-col gap-3">
                {interventions.interventions.map((intervention, i) => (
                  <InterventionCard key={i} intervention={intervention} />
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
