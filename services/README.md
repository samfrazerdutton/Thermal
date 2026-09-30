# services/

| Directory | Purpose |
|---|---|
| `api/` | FastAPI service — implemented (Phase 12). `thermal serve` runs it. Every endpoint calls the same `thermal.runner`/`thermal.storage` code as the CLI, so results never diverge between the two front ends. Local mode: no auth, SQLite storage. Three ways to start a run: `POST /api/workloads/run` (blocks until done), `/api/ws/workloads/run` (streams live events, Phase 14), or `POST /api/jobs/workloads/run` (returns a job id immediately; poll `GET /api/jobs/{id}`). |
| `worker/` | No standalone process here — the "worker" backing the job queue is an in-process thread pool (`thermal/jobs.py`'s `JobQueue`, instantiated inside `services/api/main.py`), not a separate service. A real out-of-process worker (its own deployable, pulling from a shared queue) is what would be needed for true multi-node server mode; out of scope for local mode. |
| `scheduler/` | Not implemented. Experiment queue prioritization and multi-run scheduling for server/distributed mode (Phase 18). |

Local single-user mode (the CLI) doesn't need any of `services/` — only the API does.
The job queue (`thermal/jobs.py`) persists job status to SQLite (`thermal database`-style
inspection via `thermal job list` / `thermal job show <id>`, or `GET /api/jobs`), so a
job's status survives a server restart and isn't lost if nobody is watching the
connection that submitted it — but it does not survive the *process* being killed
mid-job (an in-flight job thread has no persistence of its own progress beyond what
`thermal.runner`'s existing event/telemetry hooks already save).
