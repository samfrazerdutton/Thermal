import { Panel } from "../components/Panel";

export function Settings() {
  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-lg text-[var(--color-text)]">Settings</h1>

      <Panel title="Local mode">
        <p className="text-sm text-[var(--color-text-dim)]">
          This deployment runs entirely on your machine: SQLite at{" "}
          <code className="font-mono text-[var(--color-text)]">~/.thermal/thermal.sqlite3</code>, no authentication,
          no external network calls.
        </p>
      </Panel>

      <Panel title="Not yet configurable here">
        <p className="text-sm text-[var(--color-text-dim)]">
          Server-mode settings (Postgres connection, authentication, the optional AI explanation layer) don&apos;t
          exist yet — see docs/roadmap.md, Phases 12 and 16.
        </p>
      </Panel>
    </div>
  );
}
