import type { ReactNode } from "react";

export function Panel({ title, action, children }: { title?: string; action?: ReactNode; children: ReactNode }) {
  return (
    <div
      className="border border-[var(--color-hairline)] bg-[var(--color-surface)]"
      style={{ borderRadius: "var(--radius-panel)" }}
    >
      {title && (
        <div className="flex items-center justify-between border-b border-[var(--color-hairline)] px-4 py-2.5">
          <h2 className="text-sm text-[var(--color-text-dim)]">{title}</h2>
          {action}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
