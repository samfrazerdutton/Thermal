export function LoadingState({ label = "Loading" }: { label?: string }) {
  return <p className="text-sm text-[var(--color-text-dim)]">{label}...</p>;
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="border px-4 py-3 text-sm"
      style={{
        borderRadius: "var(--radius-panel)",
        borderColor: "var(--color-regression)",
        background: "color-mix(in srgb, var(--color-regression) 10%, transparent)",
        color: "var(--color-text)",
      }}
    >
      {message}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div
      className="border border-dashed border-[var(--color-hairline)] px-4 py-8 text-center"
      style={{ borderRadius: "var(--radius-panel)" }}
    >
      <p className="text-sm text-[var(--color-text)]">{title}</p>
      {detail && <p className="mt-1 text-xs text-[var(--color-text-dim)]">{detail}</p>}
    </div>
  );
}
