interface ParameterFormProps {
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
  gpuAvailable: boolean;
  disabled?: boolean;
}

// Renders one input per parameter, typed from the current value (number,
// boolean, or string) -- "device" and "dtype" get a real <select> constrained
// to what this machine can actually run, everything else is a plain input.
// No parameter is invented: this only ever shows keys already present in the
// workload's default_parameters.
export function ParameterForm({ params, onChange, gpuAvailable, disabled }: ParameterFormProps) {
  function setField(key: string, raw: string, kind: "number" | "boolean" | "string") {
    let value: unknown = raw;
    if (kind === "number") value = raw === "" ? 0 : Number(raw);
    if (kind === "boolean") value = raw === "true";
    onChange({ ...params, [key]: value });
  }

  return (
    <div className="flex flex-col gap-3">
      {Object.entries(params).map(([key, value]) => {
        const kind = typeof value === "number" ? "number" : typeof value === "boolean" ? "boolean" : "string";

        return (
          <label key={key} className="flex flex-col gap-1">
            <span className="text-xs text-[var(--color-text-dim)]">{key.replace(/_/g, " ")}</span>

            {key === "device" ? (
              <select
                value={String(value)}
                disabled={disabled}
                onChange={(e) => setField(key, e.target.value, "string")}
                className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
                style={{ borderRadius: "var(--radius-panel)" }}
              >
                <option value="auto">auto (prefer GPU if available)</option>
                <option value="cpu">cpu (force)</option>
                <option value="cuda:0" disabled={!gpuAvailable}>
                  cuda:0 (force){!gpuAvailable ? " -- no GPU detected" : ""}
                </option>
              </select>
            ) : kind === "boolean" ? (
              <select
                value={String(value)}
                disabled={disabled}
                onChange={(e) => setField(key, e.target.value, "boolean")}
                className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
                style={{ borderRadius: "var(--radius-panel)" }}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            ) : (
              <input
                value={String(value)}
                disabled={disabled}
                onChange={(e) => setField(key, e.target.value, kind)}
                type={kind === "number" ? "number" : "text"}
                className="border border-[var(--color-hairline-strong)] bg-[var(--color-surface)] px-2 py-1 font-mono text-sm text-[var(--color-text)] outline-none focus:border-[var(--color-signal)] disabled:opacity-50"
                style={{ borderRadius: "var(--radius-panel)" }}
              />
            )}
          </label>
        );
      })}
    </div>
  );
}
