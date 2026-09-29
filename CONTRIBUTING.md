# Contributing to THERMAL

THERMAL's core rule: **never fabricate a measurement.** Every PR is reviewed against
this before anything else.

## Ground rules

- If a metric can't be collected on the target hardware/backend, the code must return
  an explicit `available: false` with a `reason`, never a substituted or estimated
  value.
- No optimization claim ships without a controlled experiment (baseline vs. treatment,
  repeated trials, statistical comparison). A single before/after run is not evidence.
- Follow the phase order in [`docs/roadmap.md`](docs/roadmap.md). Don't build a UI for
  a subsystem that doesn't exist yet.
- New workloads live under `python/workloads/` and must declare hardware requirements,
  warmup policy, and measurement policy — see existing workloads for the interface
  once Phase 3 lands.
- New telemetry counters go through the versioned schema in `schemas/`. Do not add
  ad hoc fields to trace records.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit
thermal doctor
```

Native (CUDA/C++) code requires CMake and, on Windows, a Developer Command Prompt for
`cl.exe` — see [`docs/environment-report.md`](docs/environment-report.md).

## Tests

- Unit tests must not depend on GPU hardware unless explicitly marked and skipped with
  a clear reason when no GPU is present.
- Integration tests exercise CLI → runner → telemetry → storage → diagnosis → API.
- Do not fake hardware tests. A skipped test with a reason is honest; a mocked GPU
  metric pretending to be real is not.

## Commit style

Small, phase-scoped commits. A commit that only adds scaffolding should say so.
