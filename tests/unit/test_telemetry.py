"""Unit tests for thermal.telemetry."""

import json
import os
import time

import pytest

from thermal.telemetry import SCHEMA_VERSION, TelemetryCollector, validate_sample


@pytest.fixture()
def collector():
    c = TelemetryCollector(pid=os.getpid(), interval_seconds=0.1)
    yield c
    c.stop()


def test_sample_once_matches_schema(collector):
    sample = collector.sample_once()
    d = sample.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    validate_sample(d)  # raises on mismatch


def test_sample_never_fabricates_gpu_when_absent():
    c = TelemetryCollector(interval_seconds=0.1)
    sample = c.sample_once()
    if not c._gpu_source.available:
        assert sample.gpu is None
    else:
        assert sample.gpu is not None
    c.stop()


def test_unavailable_counters_have_reasons(collector):
    sample = collector.sample_once()
    for counter in sample.cpu.unavailable_counters:
        assert counter.reason
    if sample.gpu is not None:
        for counter in sample.gpu.unavailable_counters:
            assert counter.reason


def test_start_stop_collects_multiple_samples(collector):
    collector.start()
    time.sleep(0.55)
    collector.stop()
    assert len(collector.samples) >= 2


def test_start_samples_immediately_for_workloads_shorter_than_interval():
    """Regression test: found via live UI testing. A workload that finishes
    faster than interval_seconds (routine for small/quick workloads) used to
    get zero samples because the collector waited a full interval before its
    first sample_once() call -- so the bottleneck classifier reported UNKNOWN
    for every fast run, not because nothing could be measured, but because of
    this timing bug. The first sample must be available essentially
    immediately after start()."""
    c = TelemetryCollector(interval_seconds=5.0)
    c.start()
    time.sleep(0.05)
    c.stop()
    assert len(c.samples) >= 1


def test_double_start_raises(collector):
    collector.start()
    with pytest.raises(RuntimeError):
        collector.start()
    collector.stop()


def test_write_jsonl_round_trips(collector, tmp_path):
    collector.sample_once()
    collector.sample_once()
    out = tmp_path / "run.jsonl"
    collector.write_jsonl(out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        validate_sample(record)


def test_write_parquet_round_trips(collector, tmp_path):
    pd = pytest.importorskip("pandas")
    collector.sample_once()
    collector.sample_once()
    out = tmp_path / "run.parquet"
    collector.write_parquet(out)
    df = pd.read_parquet(out)
    assert len(df) == 2
    assert "run_id" in df.columns


def test_process_sample_tracks_real_pid(collector):
    sample = collector.sample_once()
    assert sample.process is not None
    assert sample.process.pid == os.getpid()
    assert sample.process.rss_mb is not None and sample.process.rss_mb > 0
