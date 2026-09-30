"""Background job queue (docs/roadmap.md Phase 12/18 follow-up): lets a
workload or experiment run without tying up the HTTP request or WebSocket
connection that started it for the run's whole duration.

Local mode's "worker" is an in-process thread pool inside the API server
(services/api) rather than a separate process/service -- true distributed
workers (services/worker as a standalone process, a real message queue) are
out of scope for local mode; see services/README.md. Jobs are persisted
(thermal.storage.JobRepository) so their status survives a server restart
and is inspectable the same way runs/experiments are, not a black box that
disappears if nobody is watching the connection that submitted it.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Optional

from thermal.runner import run_and_store_experiment, run_and_store_workload
from thermal.storage import JobRecord, JobRepository, default_db_path


class JobQueue:
    """One JobQueue per process. The API creates a single module-level
    instance (see services/api/main.py).

    `db_path` (if given) only isolates *this queue's own job-status
    tracking* -- the workload/experiment run a job actually executes is
    persisted by thermal.runner's own default_db_path() call inside a worker
    thread, which is independent of this instance. To isolate a job's full
    effect (job status AND the run/experiment it produces) in a test, set the
    THERMAL_DB_PATH environment variable instead -- both paths resolve
    through it (see thermal.storage.default_db_path)."""

    def __init__(self, max_workers: int = 2, db_path=None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="thermal-worker")
        self._db_path_override = db_path

    def _repo(self) -> JobRepository:
        # Re-opened per call rather than held open: sqlite3 connections aren't
        # safe to share across threads, and each worker thread runs in its own.
        # default_db_path() is resolved fresh here (not cached at __init__ time)
        # so tests that monkeypatch thermal.jobs.default_db_path *after*
        # constructing a JobQueue -- the API module builds one at import time --
        # still get an isolated database instead of silently hitting the real one.
        return JobRepository(self._db_path_override or default_db_path())

    def submit_workload_run(
        self,
        workload_name: str,
        params: Optional[dict[str, Any]] = None,
        samples: Optional[int] = None,
        warmup: Optional[int] = None,
    ) -> str:
        job = JobRecord.new(
            kind="workload_run",
            request={"workload_name": workload_name, "params": params, "samples": samples, "warmup": warmup},
        )
        self._repo().save(job)
        self._executor.submit(self._run_workload_job, job.job_id, workload_name, params, samples, warmup)
        return job.job_id

    def submit_experiment_run(
        self,
        workload_name: str,
        baseline_params: dict[str, Any],
        treatment_params: dict[str, Any],
        metric_name: str,
        higher_is_better: bool = True,
        repetitions: int = 15,
        warmup_iterations: int = 3,
        hypothesis: str = "",
    ) -> str:
        request = {
            "workload_name": workload_name,
            "baseline_params": baseline_params,
            "treatment_params": treatment_params,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "repetitions": repetitions,
            "warmup_iterations": warmup_iterations,
            "hypothesis": hypothesis,
        }
        job = JobRecord.new(kind="experiment_run", request=request)
        self._repo().save(job)
        self._executor.submit(self._run_experiment_job, job.job_id, request)
        return job.job_id

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._repo().get(job_id)

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> list[JobRecord]:
        return self._repo().list(status=status, limit=limit)

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)

    # --- worker-thread entry points -----------------------------------

    def _mark_running(self, job_id: str) -> None:
        repo = self._repo()
        job = repo.get(job_id)
        job.status = "running"
        job.started_at_ns = time.time_ns()
        repo.save(job)

    def _mark_done(self, job_id: str, result: dict[str, Any]) -> None:
        repo = self._repo()
        job = repo.get(job_id)
        job.status = "completed"
        job.finished_at_ns = time.time_ns()
        job.result = result
        repo.save(job)

    def _mark_failed(self, job_id: str, error: str) -> None:
        repo = self._repo()
        job = repo.get(job_id)
        job.status = "failed"
        job.finished_at_ns = time.time_ns()
        job.error = error
        repo.save(job)

    def _run_workload_job(
        self, job_id: str, workload_name: str, params: Optional[dict[str, Any]], samples: Optional[int], warmup: Optional[int]
    ) -> None:
        self._mark_running(job_id)
        try:
            outcome = run_and_store_workload(workload_name, params, samples=samples, warmup=warmup)
            self._mark_done(
                job_id,
                {
                    "run_id": outcome.record.run_id,
                    "device": outcome.record.device,
                    "diagnosis": {
                        "bottleneck": outcome.diagnosis.bottleneck.value,
                        "confidence": outcome.diagnosis.confidence,
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 -- a job failure must never crash the worker thread silently
            self._mark_failed(job_id, str(exc))

    def _run_experiment_job(self, job_id: str, request: dict[str, Any]) -> None:
        self._mark_running(job_id)
        try:
            outcome = run_and_store_experiment(
                workload_name=request["workload_name"],
                baseline_params=request["baseline_params"],
                treatment_params=request["treatment_params"],
                metric_name=request["metric_name"],
                higher_is_better=request["higher_is_better"],
                repetitions=request["repetitions"],
                warmup_iterations=request["warmup_iterations"],
                hypothesis=request["hypothesis"],
            )
            self._mark_done(
                job_id,
                {
                    "experiment_id": outcome.record.experiment_id,
                    "verdict": outcome.record.verdict,
                    "comparison": asdict(outcome.result.comparison),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_failed(job_id, str(exc))
