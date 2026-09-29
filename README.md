# THERMAL

**Find the bottleneck. Prove it. Fix it.**

THERMAL is an open-source, research-grade performance engineering system for AI and
accelerated computing workloads. It is not a GPU dashboard, not an LLM wrapper, and not
a benchmark chart generator. It is a closed-loop system that runs real workloads,
collects real telemetry, forms falsifiable hypotheses about what is slow, runs
controlled experiments to test them, and reports the statistically verified result.

```text
WORKLOAD → BASELINE → OBSERVATION → HYPOTHESIS → INTERVENTION →
CONTROLLED EXPERIMENT → MEASUREMENT → STATISTICAL ANALYSIS → VERIFICATION
```

## Problem

Performance work in AI infrastructure is usually done by eyeballing `nvidia-smi`,
changing something, running once, and declaring victory. That produces claims that
don't survive a second run. THERMAL exists to replace "I think this is faster" with
an evidence chain that a stranger can reproduce.

## Solution

THERMAL executes real workloads on real hardware, classifies bottlenecks from
measured features (not guesses), proposes concrete interventions, runs them as
controlled A/B experiments with proper statistics, and only reports an optimization
as real once the experiment supports it.

## Architecture

```text
native/     CUDA kernels, C++ runtime, benchmark harness
python/     thermal CLI, hardware/telemetry detection, workload plugins, analysis
core/       runtime, telemetry, profiler, diagnosis, experiments, causal, storage
services/   FastAPI service, background worker, scheduler
apps/web/   React/TypeScript engineering console
schemas/    versioned trace/experiment schemas
```

A full architecture doc lands once there's more than one subsystem to document — see
Roadmap below.

## Status

THERMAL is being built in explicit phases (see [`docs/roadmap.md`](docs/roadmap.md)).
Nothing is claimed to work until it has been run, tested, and committed.

Currently implemented:

- **Phase 0 — Repository foundation**
- **Phase 1 — Hardware detection** (`thermal hardware`, `thermal doctor`)
- **Phase 2 — Telemetry engine** (`thermal profile`, versioned trace schema)
- **Phase 3 — Workload runner** (`thermal workload list/run`; matmul, memory_bandwidth, vector_ops)
- **Phase 4 — Baseline statistics** (mean/median/CV/percentiles/bootstrap CI/outliers)
- **Phase 5 — Storage** (`thermal database list/show`, SQLite local run repository)
- **Phase 6 — Bottleneck classifier** (`thermal diagnose`, deterministic, no AI)
- **Phase 7 — CUDA kernel lab** (`thermal kernel list/run`; vector add, reduction, tiled matmul)
- **Phase 8 — Controlled experiment engine** (`thermal experiment run/list/compare`; ABAB alternation, Welch's t-test, Mann-Whitney U, bootstrap CI, honest INCONCLUSIVE verdicts)
- **Phase 9 — Counterfactual engine** (`thermal diagnose` now proposes a testable hypothesis with the exact `thermal experiment run` command to test it)
- **Phase 10 — Causal graph** (`thermal experiment causal-graph`; edges only come from experiments whose CI excluded zero, tagged EXPERIMENTAL_EVIDENCE — never asserted from correlation alone)
- **Phase 11 — Optimization search** (`thermal optimize`; grid search and coordinate descent, each candidate a real controlled experiment — never picks a candidate whose verdict wasn't IMPROVED)
- **Phase 12 — FastAPI service** (`thermal serve`; `/api/health`, `/api/hardware`, `/api/workloads`, `/api/runs`, `/api/experiments`, `/api/diagnoses`, `/api/causal-graph`, `/api/optimize/grid-search` — every endpoint calls the same `thermal.runner` code the CLI calls, so a run triggered from the API and one from the CLI are computed identically)
- **Phase 13 — Web console** (`apps/web`; React/TypeScript/Vite/Tailwind, dark instrument-panel design, live against the real API — no mock data path)
- **Phase 14 — Live experiment streaming** (`/api/ws/workloads/run`, `/api/ws/experiments/run`; a run started from the Workloads page streams real events — run_started, warmup, iteration_completed, telemetry_update, diagnosis_updated, run_completed — over a WebSocket as they happen, verified end-to-end with a real browser click-through)
- **Phase 15 — Report generation** (`thermal report <run_id|experiment_id>`, `GET /api/reports/:id`, and a Reports page in the web console — a Markdown report built entirely from what was already recorded, never recomputed, including a literal reproduction command)
- **Phase 16 — Optional AI explanation layer** (`thermal explain`, `GET /api/explain/:id`, an "explain with AI" button in the web console — grounded strictly in the stored evidence JSON, instructed never to state a number outside it; works honestly with no API key: everything else in THERMAL requires no LLM at all)

Everything else in the CLI surface exists as an explicit `NOT IMPLEMENTED` stub rather
than a fake success message — see [`docs/roadmap.md`](docs/roadmap.md) for what's next.

## Quick start

```bash
git clone https://github.com/samfrazerdutton/Thermal.git
cd Thermal
python -m pip install -e .

thermal hardware   # real GPU/CPU/toolchain detection on your machine
thermal doctor     # can this machine actually run THERMAL?
```

Example, run on the reference development machine (RTX 2060, CUDA 13.2, Windows):

```text
THERMAL HARDWARE

CPU
  AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD
  cores:   8
  threads: 16

GPU
  NVIDIA GeForce RTX 2060 with Max-Q Design
  VRAM: 6144 MB
  CUDA driver: 13.2
  Compute capability: 7.5
```

Full detected capabilities for this machine are recorded in
[`docs/environment-report.md`](docs/environment-report.md).

## Benchmarks

No benchmark results are published yet — the workload runner, baseline engine, and
experiment engine are later phases (see roadmap). THERMAL will never publish a number
it did not measure. When the first golden experiment (Phase 8) lands, its raw data and
reproduction script will live under `experiments/`.

## Research

THERMAL treats every optimization claim as a scientific claim: baseline vs. treatment,
repeated trials, confidence intervals, explicit statement of when a result is
inconclusive. A `docs/methodology.md` lands with the baseline/experiment engines
(Phase 4/8) once there is a real methodology to document.

## Limitations

Documented honestly, updated as phases land. Current, as of Phase 1:

- Native (CUDA/C++) builds require a Developer Command Prompt on Windows — `cl.exe` is
  not resolved from a plain shell. See `docs/environment-report.md`.
- No workload runner, telemetry engine, diagnosis engine, or experiment engine exists
  yet. `thermal workload`, `thermal profile`, `thermal diagnose`, `thermal experiment`,
  `thermal optimize`, `thermal report`, and `thermal serve` are stubs that print
  `NOT IMPLEMENTED` and exit non-zero.
- GPU-accelerated PyTorch is not installed in the reference environment; transformer
  workloads will not be able to use the GPU until that is added.

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full 18-phase build order.

## License

[MIT](LICENSE)
