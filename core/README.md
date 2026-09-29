# core/

The analysis and experimentation engine, independent of any particular CLI or web
frontend. Subdirectories map directly to roadmap phases:

| Directory | Phase | Purpose |
|---|---|---|
| `runtime/` | 2–3 | Workload execution runtime shared by CLI and API |
| `telemetry/` | 2 | Continuous hardware/process telemetry sampling |
| `profiler/` | 7 | Kernel/timeline profiling integration (Nsight, CUPTI) |
| `diagnosis/` | 6 | Deterministic bottleneck classifier |
| `experiments/` | 8 | Controlled experiment engine |
| `optimization/` | 11 | Optimization search strategies |
| `causal/` | 10 | Evidence-linked causal graph |
| `storage/` | 5 | Storage repository abstraction (SQLite/Postgres/Parquet) |

Each subdirectory gets real code, tests, and a short README when its phase lands —
none of these exist yet as of Phase 1.
