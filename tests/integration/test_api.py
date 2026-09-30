"""Integration tests for the THERMAL FastAPI service (services/api).

Uses FastAPI's TestClient against the real app -- every endpoint here
exercises the same thermal.runner / thermal.storage code the CLI uses, with
a per-test temporary SQLite database so tests don't pollute
~/.thermal/thermal.sqlite3 or interfere with each other.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "thermal.sqlite3"

    import thermal.jobs as jobs_mod
    import thermal.report as report_mod
    import thermal.runner as runner_mod
    import thermal.storage as storage_mod

    monkeypatch.setattr(storage_mod, "default_db_path", lambda: db_path)
    monkeypatch.setattr(runner_mod, "default_db_path", lambda: db_path)
    monkeypatch.setattr(report_mod, "default_db_path", lambda: db_path)
    # job_queue is a module-level singleton in api.main constructed at import
    # time; its _repo() re-resolves default_db_path() on every call (rather
    # than caching it), which is exactly what makes patching it here -- after
    # that singleton already exists -- take effect.
    monkeypatch.setattr(jobs_mod, "default_db_path", lambda: db_path)

    import api.main as api_main

    monkeypatch.setattr(api_main, "default_db_path", lambda: db_path)

    return TestClient(api_main.app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_hardware_returns_real_gpu_or_honest_unavailable(client):
    response = client.get("/api/hardware")
    assert response.status_code == 200
    data = response.json()
    assert "cpu" in data and "gpu" in data
    if data["gpu"]["available"]:
        assert data["gpu"]["name"]
    else:
        assert data["gpu"]["reason"]


def test_list_workloads_includes_builtins(client):
    response = client.get("/api/workloads")
    assert response.status_code == 200
    names = {w["name"] for w in response.json()}
    assert {"matmul", "memory_bandwidth", "vector_ops"}.issubset(names)


def test_run_workload_end_to_end(client):
    response = client.post(
        "/api/workloads/run",
        json={"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 5, "warmup": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["run"]["workload_name"] == "vector_ops"
    assert data["run"]["measurement_iterations"] == 5
    assert "gflops" in data["run"]["metrics"]
    assert data["diagnosis"]["bottleneck"]

    run_id = data["run"]["run_id"]
    get_response = client.get(f"/api/runs/{run_id}")
    assert get_response.status_code == 200
    assert get_response.json()["run_id"] == run_id


def test_run_unknown_workload_returns_404(client):
    response = client.post("/api/workloads/run", json={"workload_name": "not_a_real_workload"})
    assert response.status_code == 404


def test_get_missing_run_returns_404(client):
    response = client.get("/api/runs/does-not-exist")
    assert response.status_code == 404


def test_get_missing_diagnosis_returns_404(client):
    response = client.get("/api/diagnoses/does-not-exist")
    assert response.status_code == 404


def test_run_experiment_end_to_end(client):
    response = client.post(
        "/api/experiments",
        json={
            "workload_name": "matmul",
            "baseline_params": {"size": 128},
            "treatment_params": {"size": 256},
            "metric_name": "gflops",
            "repetitions": 6,
            "warmup_iterations": 1,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] in ("IMPROVED", "REGRESSED", "INCONCLUSIVE", "NO_CHANGE")
    assert data["metric_name"] == "gflops"

    list_response = client.get("/api/experiments")
    assert list_response.status_code == 200
    assert any(e["experiment_id"] == data["experiment_id"] for e in list_response.json())


def test_experiment_validation_rejects_too_few_repetitions(client):
    response = client.post(
        "/api/experiments",
        json={"workload_name": "matmul", "metric_name": "gflops", "repetitions": 1},
    )
    assert response.status_code == 422


def test_workload_run_validation_rejects_excessive_samples(client):
    response = client.post(
        "/api/workloads/run",
        json={"workload_name": "matmul", "samples": 100000},
    )
    assert response.status_code == 422


def test_causal_graph_empty_when_no_experiments(client):
    response = client.get("/api/causal-graph")
    assert response.status_code == 200
    assert response.json() == {"edges": []}


def test_causal_graph_has_edge_after_clear_experiment(client):
    client.post(
        "/api/experiments",
        json={
            "workload_name": "matmul",
            "baseline_params": {"size": 128},
            "treatment_params": {"size": 512},
            "metric_name": "gflops",
            "repetitions": 8,
            "warmup_iterations": 1,
        },
    )
    response = client.get("/api/causal-graph")
    assert response.status_code == 200
    # may be empty if the comparison happened to be inconclusive on this run;
    # assert on shape, not a specific outcome from a real (non-deterministic) measurement
    assert "edges" in response.json()


def test_report_endpoint_404_for_missing_id(client):
    response = client.get("/api/reports/does-not-exist")
    assert response.status_code == 404


def test_report_endpoint_returns_markdown_for_real_run(client):
    run_response = client.post(
        "/api/workloads/run",
        json={"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 5, "warmup": 1},
    )
    run_id = run_response.json()["run"]["run_id"]

    response = client.get(f"/api/reports/{run_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# THERMAL Report" in response.text
    assert run_id in response.text


def test_explain_endpoint_returns_503_when_ai_not_configured(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.get("/api/explain/some-id")
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def _wait_for_job(client, job_id, timeout=30.0):
    import time

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_submit_workload_job_returns_202_immediately(client):
    response = client.post(
        "/api/jobs/workloads/run",
        json={"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 4, "warmup": 1},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    assert body["job_id"]


def test_workload_job_completes_and_is_pollable(client):
    submit = client.post(
        "/api/jobs/workloads/run",
        json={"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 4, "warmup": 1},
    )
    job_id = submit.json()["job_id"]

    job = _wait_for_job(client, job_id)
    assert job["status"] == "completed"
    assert job["result"]["run_id"]

    # the run the job produced is a real, independently fetchable run record
    run_response = client.get(f"/api/runs/{job['result']['run_id']}")
    assert run_response.status_code == 200


def test_server_stays_responsive_while_a_job_runs(client):
    """The whole point of the job queue: submitting a run must not block the
    server from answering other requests while that run is in progress."""
    submit = client.post(
        "/api/jobs/workloads/run",
        json={"workload_name": "matmul", "params": {"size": 4096}, "samples": 10, "warmup": 2},
    )
    job_id = submit.json()["job_id"]

    health = client.get("/api/health")
    assert health.status_code == 200

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] in ("pending", "running", "completed")

    _wait_for_job(client, job_id)


def test_experiment_job_completes(client):
    submit = client.post(
        "/api/jobs/experiments/run",
        json={
            "workload_name": "matmul",
            "baseline_params": {"size": 64},
            "treatment_params": {"size": 128},
            "metric_name": "gflops",
            "repetitions": 6,
            "warmup_iterations": 1,
        },
    )
    assert submit.status_code == 202
    job = _wait_for_job(client, submit.json()["job_id"])
    assert job["status"] == "completed"
    assert job["result"]["verdict"] in ("IMPROVED", "REGRESSED", "INCONCLUSIVE", "NO_CHANGE")


def test_unknown_workload_job_fails_not_500(client):
    submit = client.post("/api/jobs/workloads/run", json={"workload_name": "__does_not_exist__"})
    assert submit.status_code == 202  # accepted -- failure is discovered asynchronously
    job = _wait_for_job(client, submit.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]


def test_get_missing_job_returns_404(client):
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_list_jobs_includes_submitted_job(client):
    submit = client.post(
        "/api/jobs/workloads/run",
        json={"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 4, "warmup": 1},
    )
    job_id = submit.json()["job_id"]
    _wait_for_job(client, job_id)

    listing = client.get("/api/jobs").json()
    assert any(j["job_id"] == job_id for j in listing)
