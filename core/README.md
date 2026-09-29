# core/

Empty by design decision, not by omission. The Phase 0 plan sketched here put the
runtime/telemetry/diagnosis/experiments/optimization/causal/storage engine under
`core/`, separate from the CLI. Once Phase 2 actually started, that split added a
layer of indirection with no real benefit for a single-package Python codebase — the
engine lives in `python/thermal/` instead (see its module list below), used directly
by both the CLI (`thermal/cli.py`) and the API (`services/api/main.py`) through the
shared `thermal/runner.py`.

| What was planned here | Where it actually landed |
|---|---|
| `runtime/` | `python/thermal/workload.py`, `python/thermal/runner.py` |
| `telemetry/` | `python/thermal/telemetry.py` |
| `profiler/` | `native/cuda/bench_main.cu` (CUDA-event timing); no separate Nsight/CUPTI integration |
| `diagnosis/` | `python/thermal/diagnosis.py` |
| `experiments/` | `python/thermal/experiment.py`, `python/analysis/comparison.py` |
| `optimization/` | `python/thermal/optimization.py` |
| `causal/` | `python/thermal/causal.py` |
| `storage/` | `python/thermal/storage.py` |

This directory is kept (rather than deleted) so the discrepancy between the original
plan and the as-built layout stays visible and explained, instead of silently
disappearing.
