"""Unit tests for thermal.cli._infer_primary_metric (metric direction heuristic)."""

from thermal.cli import _infer_primary_metric


def test_prefers_gflops_over_duration():
    name, higher_is_better = _infer_primary_metric({"duration_seconds": {}, "gflops": {}})
    assert name == "gflops"
    assert higher_is_better is True


def test_falls_back_to_duration_as_lower_is_better():
    name, higher_is_better = _infer_primary_metric({"duration_seconds": {}, "size": {}})
    assert name == "duration_seconds"
    assert higher_is_better is False


def test_bandwidth_metric_is_higher_is_better():
    name, higher_is_better = _infer_primary_metric({"bandwidth_gbps": {}})
    assert name == "bandwidth_gbps"
    assert higher_is_better is True


def test_no_hints_falls_back_to_first_metric():
    name, higher_is_better = _infer_primary_metric({"n": {}, "size": {}})
    assert name == "n"
    assert higher_is_better is True


def test_empty_metrics_returns_none():
    name, higher_is_better = _infer_primary_metric({})
    assert name is None
