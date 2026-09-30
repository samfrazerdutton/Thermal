# tests/

| Directory | Purpose |
|---|---|
| `unit/` | Fast, no-hardware-required tests (telemetry parsing, statistics, classification logic) |
| `integration/` | CLI → runner → telemetry → storage → diagnosis → API |
| `benchmarks/` | Golden experiments validating the bottleneck classifier — real, continuous GPU bursts deliberately constructed to be compute-, memory-, transfer-, or latency-bound (Phase 40). Not in the default `testpaths`: each burst runs for several real seconds, so run them explicitly with `python -m pytest tests/benchmarks`. |
| `reproducibility/` | Verifies a recorded run/experiment can be reproduced from its manifest |

Run: `python -m pytest tests/unit tests/integration` for the fast suite (also the
default `python -m pytest` via `testpaths` in pyproject.toml). GPU-dependent tests are
skipped with an explicit reason when no NVIDIA GPU is present — see `thermal doctor`.

The golden experiments in `benchmarks/` found and fixed two real calibration bugs in
`thermal/diagnosis.py` during development: the MEMORY_BOUND rule required low SM
utilization, which a genuinely memory-bound kernel never showed (NVML's SM utilization
reads high even while a kernel is stalled on memory); and the TRANSFER_BOUND threshold
(70% of the PCIe link ceiling) was never reached by a real pageable-memory transfer,
which plateaus around 58-62%. See the module docstring in
`tests/benchmarks/test_golden_experiments.py` for the full story.
