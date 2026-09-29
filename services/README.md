# services/

Server-mode components, built from Phase 12 onward.

| Directory | Purpose |
|---|---|
| `api/` | FastAPI service (`/api/health`, `/api/hardware`, `/api/runs`, `/api/experiments`, ...) |
| `worker/` | Background workload/experiment execution |
| `scheduler/` | Experiment queue and multi-run scheduling |

Local single-user mode (the CLI, Phases 0–11) does not require any of these — they
exist for the web console and multi-user/server deployments.
