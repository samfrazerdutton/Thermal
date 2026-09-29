"""Integration tests for the THERMAL live-streaming WebSocket endpoints
(Phase 14). Uses FastAPI's TestClient WebSocket support against the real
app and a per-test temporary SQLite database.
"""

import json

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


def test_workload_ws_streams_full_event_sequence(client):
    with client.websocket_connect("/api/ws/workloads/run") as ws:
        ws.send_text(json.dumps({"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 4, "warmup": 1}))
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg["event"])
            if msg["event"] in ("result", "error"):
                final = msg
                break

    assert final["event"] == "result"
    assert final["run"]["workload_name"] == "vector_ops"
    assert "diagnosis" in final

    assert events[0] == "run_started"
    assert "warmup_started" in events
    assert "measurement_started" in events
    assert events.count("iteration_completed") == 4  # matches samples=4
    assert "run_completed" in events
    assert events[-1] == "result"


def test_workload_ws_invalid_payload_returns_error(client):
    with client.websocket_connect("/api/ws/workloads/run") as ws:
        ws.send_text("not valid json")
        msg = ws.receive_json()
    assert msg["event"] == "error"


def test_workload_ws_unknown_workload_returns_error_event(client):
    with client.websocket_connect("/api/ws/workloads/run") as ws:
        ws.send_text(json.dumps({"workload_name": "__does_not_exist__"}))
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg["event"])
            if msg["event"] in ("result", "error"):
                break
    assert events[-1] == "error"


def test_experiment_ws_streams_start_and_finish(client):
    with client.websocket_connect("/api/ws/experiments/run") as ws:
        ws.send_text(
            json.dumps(
                {
                    "workload_name": "matmul",
                    "baseline_params": {"size": 128},
                    "treatment_params": {"size": 256},
                    "metric_name": "gflops",
                    "repetitions": 6,
                    "warmup_iterations": 1,
                }
            )
        )
        events = []
        while True:
            msg = ws.receive_json()
            events.append(msg["event"])
            if msg["event"] in ("result", "error"):
                final = msg
                break

    assert events == ["experiment_started", "experiment_finished", "result"]
    assert final["experiment"]["verdict"] in ("IMPROVED", "REGRESSED", "INCONCLUSIVE", "NO_CHANGE")


def test_run_created_over_websocket_is_persisted(client):
    with client.websocket_connect("/api/ws/workloads/run") as ws:
        ws.send_text(json.dumps({"workload_name": "vector_ops", "params": {"size_millions": 1}, "samples": 4, "warmup": 1}))
        while True:
            msg = ws.receive_json()
            if msg["event"] == "result":
                run_id = msg["run"]["run_id"]
                break

    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["run_id"] == run_id
