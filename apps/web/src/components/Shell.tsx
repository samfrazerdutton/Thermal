import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api, type HardwareReport } from "../lib/api";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/runs", label: "Runs" },
  { to: "/experiments", label: "Experiments" },
  { to: "/workloads", label: "Workloads" },
  { to: "/hardware", label: "Hardware" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/genome", label: "Genome" },
  { to: "/reports", label: "Reports" },
  { to: "/settings", label: "Settings" },
];

export function Shell() {
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .hardware()
      .then(setHw)
      .catch(() => setHw(null));
    api
      .health()
      .then(() => setApiUp(true))
      .catch(() => setApiUp(false));
  }, []);

  return (
    <div className="flex min-h-screen text-[var(--color-text)]">
      <aside className="flex w-[188px] shrink-0 flex-col border-r border-[var(--color-hairline)] bg-[var(--color-surface)]">
        <div className="flex items-center gap-2 border-b border-[var(--color-hairline)] px-4 py-4">
          <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden>
            <circle cx="16" cy="16" r="10" fill="none" stroke="var(--color-signal)" strokeWidth="2" />
            <circle cx="16" cy="16" r="3" fill="var(--color-verified)" />
          </svg>
          <span className="text-sm font-medium tracking-tight">THERMAL</span>
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 px-2 py-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `border-l-2 px-2.5 py-1.5 text-sm transition-colors ${
                  isActive
                    ? "border-[var(--color-signal)] bg-[var(--color-surface-raised)] text-[var(--color-text)]"
                    : "border-transparent text-[var(--color-text-dim)] hover:text-[var(--color-text)]"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-[var(--color-hairline)] px-4 py-3 text-xs text-[var(--color-text-faint)]">
          <div className="flex items-center gap-1.5">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: apiUp ? "var(--color-verified)" : "var(--color-regression)" }}
            />
            <span>{apiUp === null ? "connecting..." : apiUp ? "api connected" : "api unreachable"}</span>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-[var(--color-hairline)] px-6 py-3">
          <span className="text-sm text-[var(--color-text-dim)]">Local mode</span>
          <span className="font-mono text-xs text-[var(--color-text-dim)]">
            {hw?.gpu.available
              ? `${hw.gpu.name} (CUDA ${hw.gpu.cuda_driver_version ?? "unknown"})`
              : hw
                ? "no GPU detected"
                : "reading hardware..."}
          </span>
        </header>
        <main className="flex-1 overflow-y-auto px-6 py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
