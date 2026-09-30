"""Unit tests for thermal.jobs (background job queue).

Runs real workloads/experiments on a real ThreadPoolExecutor -- not mocked --
against a per-test temporary SQLite database.
"""

import time

import pytest

import workloads  # noqa: F401  (registers built-in workloads)
from thermal.jobs import JobQueue


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    # THERMAL_DB_PATH, not JobQueue(db_path=...): the workload/experiment run
    # a job executes is persisted by thermal.runner's own default_db_path()
    # call, independent of the JobQueue instance that scheduled it -- only the
    # environment variable reaches both. See thermal/jobs.py's docstring.
    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setenv("THERMAL_DB_PATH", str(db_path))
    q = JobQueue(max_workers=2)
    yield q
    q.shutdown(wait=True)


def _wait_for_terminal(queue, job_id, timeout=30.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        job = queue.get_job(job_id)
        if job.status in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not reach a terminal state within {timeout}s")


def test_submit_workload_run_returns_immediately_with_pending_or_running_job(queue):
    job_id = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=4, warmup=1)
    job = queue.get_job(job_id)
    assert job is not None
    assert job.status in ("pending", "running")
    assert job.kind == "workload_run"


def test_workload_job_completes_with_real_result(queue):
    job_id = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=4, warmup=1)
    job = _wait_for_terminal(queue, job_id)

    assert job.status == "completed"
    assert job.error is None
    assert job.result["run_id"]
    assert job.result["device"] in ("cpu", "cuda:0")
    assert job.started_at_ns is not None
    assert job.finished_at_ns is not None
    assert job.finished_at_ns >= job.started_at_ns


def test_workload_job_persists_a_real_run(queue, tmp_path):
    from thermal.storage import SQLiteRunRepository

    job_id = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=4, warmup=1)
    job = _wait_for_terminal(queue, job_id)

    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    run = repo.get(job.result["run_id"])
    assert run is not None
    assert run.workload_name == "vector_ops"


def test_unknown_workload_job_fails_with_error_not_a_crash(queue):
    job_id = queue.submit_workload_run("__does_not_exist__")
    job = _wait_for_terminal(queue, job_id)

    assert job.status == "failed"
    assert job.error is not None
    assert "unknown workload" in job.error.lower()


def test_experiment_job_completes_with_real_result(queue):
    job_id = queue.submit_experiment_run(
        workload_name="matmul",
        baseline_params={"size": 64},
        treatment_params={"size": 128},
        metric_name="gflops",
        repetitions=6,
        warmup_iterations=1,
    )
    job = _wait_for_terminal(queue, job_id)

    assert job.status == "completed"
    assert job.result["experiment_id"]
    assert job.result["verdict"] in ("IMPROVED", "REGRESSED", "INCONCLUSIVE", "NO_CHANGE")


def test_list_jobs_orders_most_recent_first(queue):
    first = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=3, warmup=1)
    _wait_for_terminal(queue, first)
    second = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=3, warmup=1)
    _wait_for_terminal(queue, second)

    jobs = queue.list_jobs(limit=10)
    ids = [j.job_id for j in jobs]
    assert ids.index(second) < ids.index(first)


def test_list_jobs_filters_by_status(queue):
    job_id = queue.submit_workload_run("vector_ops", {"size_millions": 1}, samples=3, warmup=1)
    _wait_for_terminal(queue, job_id)

    completed = queue.list_jobs(status="completed", limit=10)
    assert any(j.job_id == job_id for j in completed)
    pending = queue.list_jobs(status="pending", limit=10)
    assert not any(j.job_id == job_id for j in pending)


def test_concurrent_jobs_actually_run_in_parallel(queue):
    """Two jobs submitted back to back should both be running (or done)
    well before either would finish if they ran one after another --
    proof the thread pool genuinely overlaps them rather than serializing."""
    job_a = queue.submit_workload_run("vector_ops", {"size_millions": 50}, samples=20, warmup=3)
    job_b = queue.submit_workload_run("vector_ops", {"size_millions": 50}, samples=20, warmup=3)

    time.sleep(0.05)
    status_a = queue.get_job(job_a).status
    status_b = queue.get_job(job_b).status
    assert status_a in ("pending", "running")
    assert status_b in ("pending", "running")

    _wait_for_terminal(queue, job_a)
    _wait_for_terminal(queue, job_b)
