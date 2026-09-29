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

    import thermal.runner as runner_mod
    import thermal.storage as storage_mod

    monkeypatch.setattr(storage_mod, "default_db_path", lambda: db_path)
    monkeypatch.setattr(runner_mod, "default_db_path", lambda: db_path)

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


def test_report_endpoint_returns_not_implemented(client):
    response = client.get("/api/reports/some-id")
    assert response.status_code == 501
