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
python/     thermal CLI + engine (hardware, telemetry, workloads, diagnosis,
            experiments, causal graph, optimization, storage, report, AI layer)
services/   FastAPI service (thermal serve) -- calls the exact same python/thermal
            code as the CLI via thermal/runner.py, so results never diverge
apps/web/   React/TypeScript engineering console, live against services/api
schemas/    versioned trace/run-manifest schemas
```

`core/` exists in this tree but is deliberately empty — see
[`core/README.md`](core/README.md) for why the engine lives in `python/thermal/`
instead of the separate directory the original plan sketched.

## Status

THERMAL was built in 18 explicit phases (see [`docs/roadmap.md`](docs/roadmap.md)).
All 18 are implemented, tested, and committed — nothing here is claimed to work
without having been run first. The two deliberately narrower ones: Phase 3 ships 3
of the 8 workloads the original design sketch named (matmul, memory_bandwidth,
vector_ops — the rest are straightforward to add following the same interface), and
Phase 18 implements real multi-process distributed dispatch but not multi-GPU/multi-node
coordination, because this development machine has exactly one GPU and one node to
verify that against.

Implemented, phase by phase:

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
- **Phase 17 — CI** (`.github/workflows/ci.yml`: ruff lint, Python unit+integration tests, a benchmark smoke test through the real CLI, web console lint+build, a native CMake configure smoke test that asserts CUDA is honestly reported unavailable on a CUDA-less runner, a required-docs check, and GPU-dependent tests explicitly labeled as hand-verified on the reference machine rather than silently skipped)
- **Phase 18 — Distributed mode** (`thermal distribute`; real dispatch across OS worker processes via `ProcessPoolExecutor`, each producing its own persisted run — multi-GPU and multi-node coordination are explicitly not implemented, not simulated)

A few CLI surfaces named in the original design sketch but out of scope for the
18 phases above (`thermal similar` / workload genome, `thermal compare` for CI
regression detection, `thermal research init`) remain explicit `NOT IMPLEMENTED`
stubs rather than fake success messages.

## Quick start

```bash
git clone https://github.com/samfrazerdutton/Thermal.git
cd Thermal
python -m pip install -e .

thermal hardware   # real GPU/CPU/toolchain detection on your machine
thermal doctor     # can this machine actually run THERMAL?
```

`pip install -e .` alone pulls the default (CPU-only) PyTorch wheel from PyPI —
`thermal doctor` will show `torch_cuda: unavailable` and workloads will honestly report
`device: cpu`. To exercise an NVIDIA GPU, install a CUDA build afterward from
[pytorch.org](https://pytorch.org/get-started/locally/), e.g. for CUDA 12.8:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
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

No curated benchmark numbers are published in this README — THERMAL will never
publish a number it did not just measure on the machine reading this. Generate your
own instead:

```bash
thermal workload run matmul --param size=1024
thermal experiment run matmul --baseline-param size=512 --treatment-param size=1024 \
    --metric gflops --hypothesis "Larger N improves achieved GFLOP/s."
thermal report latest
```

or open the web console (`thermal serve` + `cd apps/web && npm run dev`) and watch a
run stream live. `experiments/` holds research-mode experiment directories once you
run `thermal research init` (not yet implemented — see Limitations).

## Research

THERMAL treats every optimization claim as a scientific claim: baseline vs. treatment,
run in strict ABAB alternation, repeated trials, Welch's t-test and Mann-Whitney U,
a bootstrap confidence interval on the change, and an explicit INCONCLUSIVE verdict
when the evidence doesn't support a direction. See
[`python/analysis/baseline.py`](python/analysis/baseline.py) and
[`python/analysis/comparison.py`](python/analysis/comparison.py) for the exact
methodology, and [`python/thermal/diagnosis.py`](python/thermal/diagnosis.py) for the
deterministic bottleneck classifier's thresholds and evidence model.

## Limitations

Documented honestly rather than hidden:

- Only 3 of the 8 workloads sketched in the original design exist (matmul,
  memory_bandwidth, vector_ops) — transformer inference, KV-cache stress, CPU/GPU
  transfer, batch-size scaling, and compression-vs-transfer are not implemented.
  New workloads are straightforward: subclass `thermal.workload.Workload` and register
  it (see `python/workloads/matmul.py` for the shortest example).
- ~~Built-in workloads run on CPU~~ **Fixed**: the reference machine now has
  `torch==2.10.0+cu128` installed (`thermal doctor` reports `torch_cuda: 12.8`), so
  `matmul`/`memory_bandwidth`/`vector_ops` execute on `cuda:0` and `thermal hardware`'s
  GPU telemetry reflects real workload activity. One real limitation this surfaced:
  telemetry samples every 200ms (`TelemetryCollector`'s default interval), so a workload
  whose total measured runtime is shorter than that (e.g. a 2048×2048 matmul, ~5ms/iter)
  can finish before a single sample lands mid-kernel, and the classifier honestly reports
  `gpu_utilization_percent = 0.00` even though the GPU did the work — verified by
  re-running the same workload at 8192×8192 (long enough per iteration to be sampled),
  which correctly showed 31% GPU utilization. A future fix would sample at a much
  higher frequency or hold the GPU busy across more back-to-back iterations before
  each sample.
- The bottleneck classifier (`thermal/diagnosis.py`) is first-generation, threshold-based
  heuristics, not validated against the golden experiments the original design sketch
  called for (deliberately-constructed memory-bound/compute-bound/transfer-bound/
  launch-bound workloads) — that validation harness doesn't exist yet.
- Native CUDA builds require a non-default CMake generator on Windows (Ninja, after
  loading the MSVC environment) because this machine's CUDA install has no Visual
  Studio integration — `scripts/build-native.ps1` automates it. See
  `docs/environment-report.md` for the full finding.
- The API's POST endpoints and CLI runs execute synchronously; the WebSocket streaming
  endpoints (Phase 14) don't change that — there is still no background job queue, so
  a run is tied to one open connection/request.
- The optional AI layer (Phase 16) requires `ANTHROPIC_API_KEY`; without it, `thermal
  explain` and `GET /api/explain/:id` say so and exit/return non-success rather than
  silently doing nothing.
- Distributed mode (Phase 18) is real multi-process dispatch only; multi-GPU and
  multi-node coordination are not implemented (one GPU, one node here to test with).
- `thermal similar` (workload genome/similarity search), `thermal compare` (CI
  regression detection across commits), and `thermal research init` are unimplemented
  stubs.
- Server mode (Postgres storage, authentication, multi-user) doesn't exist; local mode
  (SQLite, no auth) is the only supported deployment.

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full 18-phase build order.

## License

[MIT](LICENSE)
