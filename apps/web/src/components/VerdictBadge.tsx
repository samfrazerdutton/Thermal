const STYLES: Record<string, string> = {
  IMPROVED: "text-[var(--color-verified)] border-[var(--color-verified)]/40 bg-[var(--color-verified)]/10",
  REGRESSED: "text-[var(--color-regression)] border-[var(--color-regression)]/40 bg-[var(--color-regression)]/10",
  INCONCLUSIVE: "text-[var(--color-caution)] border-[var(--color-caution)]/40 bg-[var(--color-caution)]/10",
  NO_CHANGE: "text-[var(--color-text-dim)] border-[var(--color-hairline-strong)] bg-transparent",
};

export function VerdictBadge({ verdict }: { verdict: string }) {
  const style = STYLES[verdict] ?? STYLES.NO_CHANGE;
  return (
    <span className={`inline-flex items-center rounded-[var(--radius-panel)] border px-2 py-0.5 font-mono text-xs ${style}`}>
      {verdict.replace("_", " ").toLowerCase()}
    </span>
  );
}
