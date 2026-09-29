interface StatValueProps {
  label: string;
  value: string | number;
  unit?: string;
  tone?: "default" | "verified" | "caution" | "regression";
}

const TONE_COLOR: Record<string, string> = {
  default: "var(--color-text)",
  verified: "var(--color-verified)",
  caution: "var(--color-caution)",
  regression: "var(--color-regression)",
};

export function StatValue({ label, value, unit, tone = "default" }: StatValueProps) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-[var(--color-text-dim)]">{label}</span>
      <span className="font-mono text-2xl tabular-nums" style={{ color: TONE_COLOR[tone] }}>
        {value}
        {unit && <span className="ml-1 text-sm text-[var(--color-text-dim)]">{unit}</span>}
      </span>
    </div>
  );
}
