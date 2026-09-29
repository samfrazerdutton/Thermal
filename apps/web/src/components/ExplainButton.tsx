import { useState } from "react";
import { ApiError } from "../lib/api";

// Optional AI layer (Phase 16). Never shown as broken when unconfigured --
// a 503 here is an expected mode (no ANTHROPIC_API_KEY set), not an error.
export function ExplainButton({ id }: { id: string }) {
  const [state, setState] = useState<{ status: "idle" | "loading" | "unavailable" | "error" | "ready"; text?: string }>({
    status: "idle",
  });

  async function handleClick() {
    setState({ status: "loading" });
    try {
      const response = await fetch(`/api/explain/${id}`);
      if (response.status === 503) {
        setState({ status: "unavailable" });
        return;
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new ApiError(response.status, body.detail ?? response.statusText);
      }
      const data = await response.json();
      setState({ status: "ready", text: data.explanation });
    } catch (err) {
      setState({ status: "error", text: err instanceof Error ? err.message : String(err) });
    }
  }

  if (state.status === "idle") {
    return (
      <button onClick={handleClick} className="text-sm text-[var(--color-signal)] hover:underline">
        explain with AI
      </button>
    );
  }

  if (state.status === "loading") {
    return <p className="text-sm text-[var(--color-text-dim)]">asking...</p>;
  }

  if (state.status === "unavailable") {
    return (
      <p className="text-xs text-[var(--color-text-faint)]">
        AI explanation isn&apos;t configured on this server (no ANTHROPIC_API_KEY) — everything else here works without it.
      </p>
    );
  }

  if (state.status === "error") {
    return <p className="text-xs text-[var(--color-regression)]">{state.text}</p>;
  }

  return <p className="text-sm leading-relaxed text-[var(--color-text)]">{state.text}</p>;
}
