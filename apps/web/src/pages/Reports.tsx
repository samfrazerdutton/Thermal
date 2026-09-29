import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { EmptyState, ErrorState, LoadingState } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { ApiError } from "../lib/api";

async function fetchReport(id: string): Promise<string> {
  const response = await fetch(`/api/reports/${id}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail ?? response.statusText);
  }
  return response.text();
}

export function Reports() {
  const [searchParams] = useSearchParams();
  const [id, setId] = useState(searchParams.get("id") ?? "");
  const [state, setState] = useState<{ status: "idle" | "loading" | "error" | "ready"; report?: string; message?: string }>({
    status: "idle",
  });

  async function handleFetch(targetId: string) {
    if (!targetId.trim()) return;
    setState({ status: "loading" });
    try {
      const report = await fetchReport(targetId.trim());
      setState({ status: "ready", report });
    } catch (err) {
      setState({ status: "error", message: err instanceof ApiError ? err.message : String(err) });
    }
  }

  useEffect(() => {
    const paramId = searchParams.get("id");
    if (paramId) {
      setId(paramId);
      handleFetch(paramId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-lg text-[var(--color-text)]">Reports</h1>
        <p className="mt-1 text-sm text-[var(--color-text-dim)]">
          Enter a run or experiment id (or run <code className="font-mono text-[var(--color-signal)]">thermal report latest</code>{" "}
          from the CLI) to generate a Markdown report from exactly what was recorded — nothing is recomputed here.
        </p>
      </div>

      <div className="flex gap-2">
        <input
          value={id}
          onChange={(e) => setId(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleFetch(id)}
          placeholder="run or experiment id"
          className="flex-1 border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-3 py-1.5 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)]"
          style={{ borderRadius: "var(--radius-panel)" }}
        />
        <button
          onClick={() => handleFetch(id)}
          className="border border-[var(--color-hairline-strong)] px-4 py-1.5 text-sm text-[var(--color-text-dim)] hover:border-[var(--color-signal)] hover:text-[var(--color-signal)]"
          style={{ borderRadius: "var(--radius-panel)" }}
        >
          generate
        </button>
      </div>

      {state.status === "loading" && <LoadingState label="Generating report" />}
      {state.status === "error" && <ErrorState message={state.message ?? "failed"} />}
      {state.status === "idle" && (
        <EmptyState title="No report generated yet" detail="Paste a run or experiment id above." />
      )}
      {state.status === "ready" && state.report && (
        <Panel>
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs leading-relaxed text-[var(--color-text)]">
            {state.report}
          </pre>
        </Panel>
      )}
    </div>
  );
}
