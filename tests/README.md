# tests/

| Directory | Purpose |
|---|---|
| `unit/` | Fast, no-hardware-required tests (telemetry parsing, statistics, classification logic) |
| `integration/` | CLI → runner → telemetry → storage → diagnosis → API |
| `benchmarks/` | Golden experiments used to validate the bottleneck classifier (Phase 40) |
| `reproducibility/` | Verifies a recorded run/experiment can be reproduced from its manifest |

Run: `python -m pytest tests/unit`. GPU-dependent tests are skipped with an explicit
reason when no NVIDIA GPU is present — see `thermal doctor`.
