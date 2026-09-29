"""Unit tests for thermal.storage (SQLiteRunRepository)."""

from thermal.storage import RunRecord, SQLiteRunRepository


def _make_record(**overrides):
    defaults = dict(
        workload_name="matmul",
        workload_version="1.0.0",
        device="cpu",
        warmup_iterations=3,
        measurement_iterations=10,
        configuration={"size": 256},
        hardware_fingerprint={"cpu_name": "test-cpu", "gpu_name": None},
        metrics={"gflops": {"mean": 100.0, "n": 10}},
        git_commit="abc123",
    )
    defaults.update(overrides)
    return RunRecord.new(**defaults)


def test_save_and_get_roundtrip(tmp_path):
    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    record = _make_record()
    repo.save(record)

    fetched = repo.get(record.run_id)
    assert fetched is not None
    assert fetched.run_id == record.run_id
    assert fetched.workload_name == "matmul"
    assert fetched.configuration == {"size": 256}
    assert fetched.metrics == {"gflops": {"mean": 100.0, "n": 10}}


def test_get_missing_run_returns_none(tmp_path):
    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    assert repo.get("does-not-exist") is None


def test_list_orders_most_recent_first(tmp_path):
    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    first = _make_record()
    first.created_at_ns = 1000
    second = _make_record()
    second.created_at_ns = 2000
    repo.save(first)
    repo.save(second)

    runs = repo.list()
    assert [r.run_id for r in runs] == [second.run_id, first.run_id]


def test_list_filters_by_workload_name(tmp_path):
    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    repo.save(_make_record(workload_name="matmul"))
    repo.save(_make_record(workload_name="vector_ops"))

    runs = repo.list(workload_name="vector_ops")
    assert len(runs) == 1
    assert runs[0].workload_name == "vector_ops"


def test_list_respects_limit(tmp_path):
    repo = SQLiteRunRepository(tmp_path / "thermal.sqlite3")
    for _ in range(5):
        repo.save(_make_record())
    assert len(repo.list(limit=2)) == 2


def test_repository_persists_across_reopen(tmp_path):
    db_path = tmp_path / "thermal.sqlite3"
    record = _make_record()
    SQLiteRunRepository(db_path).save(record)

    reopened = SQLiteRunRepository(db_path)
    fetched = reopened.get(record.run_id)
    assert fetched is not None
    assert fetched.run_id == record.run_id
