# services/

| Directory | Purpose |
|---|---|
| `api/` | FastAPI service — implemented (Phase 12). `thermal serve` runs it. Every endpoint calls the same `thermal.runner`/`thermal.storage` code as the CLI, so results never diverge between the two front ends. Local mode: no auth, SQLite storage, synchronous execution (a `POST /api/workloads/run` blocks until the run finishes — there is no background queue yet). |
| `worker/` | Not implemented. Background/async workload execution for long-running jobs, needed once Phase 14 (live streaming) or true multi-user server mode lands. |
| `scheduler/` | Not implemented. Experiment queue and multi-run scheduling for server/distributed mode (Phase 18). |

Local single-user mode (the CLI) doesn't require `worker/` or `scheduler/` — only the
API does, and only once requests need to run outside the request/response cycle.
