"""Unit tests for thermal.storage.ExperimentRepository."""

from thermal.storage import ExperimentRecord, ExperimentRepository


def _make_experiment(**overrides):
    defaults = dict(
        workload_name="matmul",
        metric_name="gflops",
        higher_is_better=True,
        repetitions=20,
        baseline_config={"size": 256},
        treatment_config={"size": 512},
        baseline_values=[100.0] * 20,
        treatment_values=[150.0] * 20,
        baseline_device="cpu",
        treatment_device="cpu",
        comparison={"verdict": "IMPROVED", "percent_change": 50.0},
        verdict="IMPROVED",
        hypothesis="bigger is faster",
        git_commit="abc123",
    )
    defaults.update(overrides)
    return ExperimentRecord.new(**defaults)


def test_save_and_get_roundtrip(tmp_path):
    repo = ExperimentRepository(tmp_path / "thermal.sqlite3")
    record = _make_experiment()
    repo.save(record)

    fetched = repo.get(record.experiment_id)
    assert fetched is not None
    assert fetched.workload_name == "matmul"
    assert fetched.verdict == "IMPROVED"
    assert fetched.baseline_values == [100.0] * 20
    assert fetched.hypothesis == "bigger is faster"


def test_get_missing_returns_none(tmp_path):
    repo = ExperimentRepository(tmp_path / "thermal.sqlite3")
    assert repo.get("does-not-exist") is None


def test_list_orders_most_recent_first(tmp_path):
    repo = ExperimentRepository(tmp_path / "thermal.sqlite3")
    first = _make_experiment()
    first.created_at_ns = 1000
    second = _make_experiment()
    second.created_at_ns = 2000
    repo.save(first)
    repo.save(second)

    experiments = repo.list()
    assert [e.experiment_id for e in experiments] == [second.experiment_id, first.experiment_id]


def test_list_filters_by_workload(tmp_path):
    repo = ExperimentRepository(tmp_path / "thermal.sqlite3")
    repo.save(_make_experiment(workload_name="matmul"))
    repo.save(_make_experiment(workload_name="vector_ops"))

    experiments = repo.list(workload_name="vector_ops")
    assert len(experiments) == 1
    assert experiments[0].workload_name == "vector_ops"


def test_higher_is_better_roundtrips_as_bool(tmp_path):
    repo = ExperimentRepository(tmp_path / "thermal.sqlite3")
    record = _make_experiment(higher_is_better=False)
    repo.save(record)
    fetched = repo.get(record.experiment_id)
    assert fetched.higher_is_better is False
