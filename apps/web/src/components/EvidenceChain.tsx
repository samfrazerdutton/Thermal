export interface EvidenceStep {
  label: string;
  status: "done" | "active" | "pending";
  detail?: string;
}

// The recurring structural device across THERMAL: every claim traces through
// this exact pipeline (see README "PRIMARY DESIGN PRINCIPLE"). Rendered as a
// real state machine, not a generic stepper -- a step is only "done" when
// this run/experiment actually produced that evidence.
export function EvidenceChain({ steps }: { steps: EvidenceStep[] }) {
  return (
    <div className="flex flex-wrap items-stretch gap-0">
      {steps.map((step, i) => (
        <div key={step.label} className="flex items-stretch">
          <div
            className={`flex min-w-[104px] flex-col gap-1.5 border px-3 py-2.5 ${
              step.status === "done"
                ? "border-[var(--color-signal-dim)] bg-[var(--color-signal)]/[0.06]"
                : step.status === "active"
                  ? "border-[var(--color-caution)] bg-[var(--color-caution)]/[0.06]"
                  : "border-[var(--color-hairline)] bg-transparent"
            }`}
            style={{ borderRadius: "var(--radius-panel)" }}
          >
            <div className="flex items-center gap-1.5">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{
                  background:
                    step.status === "done"
                      ? "var(--color-signal)"
                      : step.status === "active"
                        ? "var(--color-caution)"
                        : "var(--color-hairline-strong)",
                }}
              />
              <span className="text-[11px] text-[var(--color-text-dim)]">{step.label}</span>
            </div>
            <span className="font-mono text-sm text-[var(--color-text)]">{step.detail ?? "—"}</span>
          </div>
          {i < steps.length - 1 && (
            <div className="flex w-6 items-center justify-center">
              <div className="h-px w-full bg-[var(--color-hairline)]" />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
