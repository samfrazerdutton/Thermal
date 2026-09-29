"""Unit tests for thermal.distributed.

Runs real OS processes (concurrent.futures.ProcessPoolExecutor) -- these
tests are slower than the rest of the suite but exercise the real dispatch
path, not a mock. Each worker persists its own run to a temporary database.

Isolation here uses the THERMAL_DB_PATH environment variable, not
monkeypatch: worker processes are spawned fresh (the default start method on
Windows) and do not inherit the parent's in-memory monkeypatched state, but
they do inherit environment variables -- this is exactly why
thermal.storage.default_db_path() reads that variable.
"""

import pytest

from thermal.distributed import run_workload_multiprocess


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "thermal.sqlite3"
    monkeypatch.setenv("THERMAL_DB_PATH", str(db_path))
    return db_path


def test_rejects_zero_workers():
    with pytest.raises(ValueError):
        run_workload_multiprocess("vector_ops", num_workers=0)


def test_dispatches_across_multiple_real_processes(isolated_db):
    result = run_workload_multiprocess("vector_ops", {"size_millions": 1}, num_workers=3, samples=3, warmup=1)

    assert result.num_workers == 3
    assert len(result.workers) == 3
    # real OS processes -- must be genuinely distinct PIDs, not the same process three times
    assert len({w.worker_pid for w in result.workers}) == 3
    # each worker's run_id must be distinct too
    assert len({w.run_id for w in result.workers}) == 3
    assert result.wall_clock_seconds > 0


def test_each_worker_run_is_actually_persisted(isolated_db):
    from thermal.storage import SQLiteRunRepository

    result = run_workload_multiprocess("vector_ops", {"size_millions": 1}, num_workers=2, samples=3, warmup=1)

    repo = SQLiteRunRepository(isolated_db)
    for worker in result.workers:
        record = repo.get(worker.run_id)
        assert record is not None
        assert record.workload_name == "vector_ops"


def test_actually_multi_gpu_false_when_all_workers_share_a_device(isolated_db):
    result = run_workload_multiprocess("vector_ops", {"size_millions": 1}, num_workers=2, samples=3, warmup=1)
    # on any machine, every worker running the same CPU/GPU-selecting workload
    # lands on the same device unless the machine has multiple GPUs assigned
    # per-worker (not implemented) -- so this must be False here.
    assert len(result.distinct_devices) == 1
    assert result.actually_multi_gpu is False


def test_unknown_workload_propagates_as_keyerror(isolated_db):
    with pytest.raises(KeyError):
        run_workload_multiprocess("__does_not_exist__", num_workers=2, samples=3, warmup=1)
