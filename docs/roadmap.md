# THERMAL Roadmap

Built in this exact order. Each phase must be implemented, run, tested, and committed
before the next begins — no phase is "done" because its files exist.

| Phase | Description | Status |
|---|---|---|
| 0 | Repository foundation | Done |
| 1 | Hardware detection (`thermal hardware`, `thermal doctor`) | Done |
| 2 | Telemetry engine (continuous GPU/CPU/process sampling, versioned trace schema) | Done |
| 3 | Workload runner (plugin architecture, first workloads) | Done (3 of 8 workloads: matmul, memory_bandwidth, vector_ops) |
| 4 | Baseline statistics engine (warmup, repeated trials, CI, outlier detection) | Done |
| 5 | Storage (SQLite local / Postgres server, Parquet telemetry, shared repository abstraction) | Done (SQLite local; Postgres deferred to Phase 12) |
| 6 | Bottleneck classifier (deterministic, feature-based, no AI) | Done (first-generation heuristics; refine against Phase 40 golden experiments) |
| 7 | CUDA benchmark suite (native/cuda kernel lab) | Done (vector_add, reduction, matmul: naive + optimized variants) |
| 8 | Experiment engine (controlled A/B, randomization policy) | Done |
| 9 | Counterfactual engine (hypothesis generation from bottleneck class) | Done |
| 10 | Causal graph (evidence-linked, not correlation-only) | Done |
| 11 | Optimization search (grid/binary/coordinate descent, optional Bayesian) | Done (grid search + coordinate descent; Bayesian deferred) |
| 12 | FastAPI service | Done (local mode: no auth, synchronous execution; report endpoint stubbed 501 pending Phase 15) |
| 13 | React web console | Done (Overview, Runs, Experiments, Workloads, Hardware, Benchmarks; Genome/Reports honest stubs) |
| 14 | Live experiment streaming (WebSocket/SSE) | Not started |
| 15 | Report generation (`thermal report`) | Not started |
| 16 | Optional AI explanation layer (evidence-grounded, no fabrication, works without it) | Not started |
| 17 | CI integration (GitHub Actions, GPU-dependent tests clearly labeled) | Not started |
| 18 | Distributed / cluster mode | Not started |

## Why phases, not features

A dashboard showing invented numbers is easy to build in an afternoon and worthless.
The hard, valuable part of THERMAL is Phases 2–11: real telemetry, a bottleneck
classifier that works from measured features, and an experiment engine whose verdicts
are statistically defensible. The web UI (Phase 13) is deliberately late — it has
nothing honest to display until the engine underneath it produces real evidence.

## Current gaps (see `docs/environment-report.md` for detection detail)

- No CUDA-enabled PyTorch installed — blocks GPU-side transformer workloads (Phase 3).
- No C++ compiler on PATH outside a VS Developer Command Prompt — native builds
  (Phase 7) need a build script that locates and invokes `vcvarsall.bat`, or a CMake
  invocation using the Visual Studio generator.
- No Nsight Systems — timeline tracing unavailable until installed; Nsight Compute
  (kernel-level metrics) is available.
