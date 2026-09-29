"""Unit tests for thermal.runner (shared execution logic behind CLI + API)."""

import pytest

import workloads  # noqa: F401  (registers matmul/memory_bandwidth/vector_ops)
from thermal.runner import run_and_store_experiment, run_and_store_workload
from thermal.storage import ExperimentRepository, SQLiteRunRepository


def test_run_and_store_workload_persists_and_returns_matching_data(tmp_path, monkeypatch):
    import thermal.runner as runner_mod

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(runner_mod, "default_db_path", lambda: db_path)

    outcome = run_and_store_workload("vector_ops", {"size_millions": 1}, samples=5, warmup=1)

    assert outcome.record.workload_name == "vector_ops"
    assert outcome.record.measurement_iterations == 5
    assert "gflops" in outcome.baselines
    assert outcome.diagnosis is not None
    assert len(outcome.per_iteration_metrics) == 5

    repo = SQLiteRunRepository(db_path)
    fetched = repo.get(outcome.record.run_id)
    assert fetched is not None
    assert fetched.workload_name == "vector_ops"


def test_run_and_store_workload_unknown_name_raises(tmp_path, monkeypatch):
    import thermal.runner as runner_mod

    monkeypatch.setattr(runner_mod, "default_db_path", lambda: tmp_path / "thermal.sqlite3")
    with pytest.raises(KeyError):
        run_and_store_workload("__does_not_exist__", {})


def test_run_and_store_experiment_persists_and_returns_matching_data(tmp_path, monkeypatch):
    import thermal.runner as runner_mod

    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setattr(runner_mod, "default_db_path", lambda: db_path)

    outcome = run_and_store_experiment(
        workload_name="matmul",
        baseline_params={"size": 128},
        treatment_params={"size": 256},
        metric_name="gflops",
        repetitions=6,
        warmup_iterations=1,
    )

    assert outcome.record.workload_name == "matmul"
    assert outcome.result.comparison is not None

    repo = ExperimentRepository(db_path)
    fetched = repo.get(outcome.record.experiment_id)
    assert fetched is not None
    assert fetched.metric_name == "gflops"
