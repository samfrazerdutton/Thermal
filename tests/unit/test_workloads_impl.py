"""Integration-flavored unit tests for the built-in workloads.

Runs each with tiny parameters so the suite stays fast, and asserts the
reported device matches what's actually available (never claims GPU
execution on a CPU-only torch build).
"""

import pytest

torch = pytest.importorskip("torch")

import workloads  # noqa: F401  (registers matmul/memory_bandwidth/vector_ops)
from thermal.device import gpu_available
from thermal.workload import WorkloadRegistry, run_workload


@pytest.mark.parametrize(
    "name,params",
    [
        ("matmul", {"size": 32}),
        ("memory_bandwidth", {"size_mb": 1}),
        ("vector_ops", {"size_millions": 1}),
    ],
)
def test_workload_runs_and_reports_real_device(name, params):
    workload_cls = WorkloadRegistry.get(name)
    instance = workload_cls(params)
    instance.spec = instance.spec.__class__(
        **{**instance.spec.__dict__, "warmup_iterations": 1, "measurement_iterations": 2}
    )
    result = run_workload(instance)

    assert len(result.per_iteration_metrics) == 2
    expected_device = "cuda:0" if gpu_available() else "cpu"
    assert result.device == expected_device

    for metric in result.per_iteration_metrics:
        assert metric["duration_seconds"] > 0
