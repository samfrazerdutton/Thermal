"""Integration-flavored unit tests for the built-in workloads.

Runs each with tiny parameters so the suite stays fast, and asserts the
reported device matches what's actually available (never claims GPU
execution on a CPU-only torch build).
"""

import pytest

torch = pytest.importorskip("torch")

import workloads  # noqa: E402, F401  (registers matmul/memory_bandwidth/vector_ops; import deliberately after the torch skip-guard above)
from thermal.device import gpu_available  # noqa: E402
from thermal.workload import WorkloadRegistry, run_workload  # noqa: E402


@pytest.mark.parametrize(
    "name,params",
    [
        ("matmul", {"size": 32}),
        ("memory_bandwidth", {"size_mb": 1}),
        ("vector_ops", {"size_millions": 1}),
        ("transformer_inference", {"batch_size": 2, "seq_length": 16, "hidden_dim": 32, "num_layers": 1, "num_heads": 2}),
        ("kv_cache_stress", {"kv_cache_length": 64, "num_heads": 2, "head_dim": 16}),
        ("batch_size_scaling", {"batch_size": 4, "in_features": 64, "out_features": 64}),
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


@pytest.mark.parametrize("name", ["matmul", "memory_bandwidth", "vector_ops", "batch_size_scaling"])
def test_workload_device_param_forces_cpu_even_when_gpu_available(name):
    workload_cls = WorkloadRegistry.get(name)
    params = {"device": "cpu"}
    if name == "matmul":
        params["size"] = 32
    elif name == "memory_bandwidth":
        params["size_mb"] = 1
    elif name == "vector_ops":
        params["size_millions"] = 1
    elif name == "batch_size_scaling":
        params.update(batch_size=4, in_features=64, out_features=64)

    instance = workload_cls(params)
    instance.spec = instance.spec.__class__(
        **{**instance.spec.__dict__, "warmup_iterations": 1, "measurement_iterations": 2}
    )
    result = run_workload(instance)
    assert result.device == "cpu"


requires_gpu = pytest.mark.skipif(not gpu_available(), reason="cpu_gpu_transfer/compression_vs_transfer need a real CUDA device")


@requires_gpu
def test_workload_device_param_forces_cuda_when_requested():
    workload_cls = WorkloadRegistry.get("matmul")
    instance = workload_cls({"size": 32, "device": "cuda:0"})
    instance.spec = instance.spec.__class__(
        **{**instance.spec.__dict__, "warmup_iterations": 1, "measurement_iterations": 2}
    )
    result = run_workload(instance)
    assert result.device == "cuda:0"


def test_workload_device_param_invalid_raises_value_error():
    workload_cls = WorkloadRegistry.get("matmul")
    instance = workload_cls({"size": 32, "device": "not_a_real_device"})
    with pytest.raises(ValueError):
        instance.setup()


@requires_gpu
@pytest.mark.parametrize("params", [{"size_mb": 1, "pinned": False}, {"size_mb": 1, "pinned": True}])
def test_cpu_gpu_transfer_runs_on_real_gpu(params):
    workload_cls = WorkloadRegistry.get("cpu_gpu_transfer")
    instance = workload_cls(params)
    instance.spec = instance.spec.__class__(
        **{**instance.spec.__dict__, "warmup_iterations": 1, "measurement_iterations": 2}
    )
    result = run_workload(instance)
    assert result.device == "cuda:0"
    for metric in result.per_iteration_metrics:
        assert metric["h2d_bandwidth_gbps"] > 0
        assert metric["d2h_bandwidth_gbps"] > 0


@pytest.mark.parametrize("mode", ["raw", "compressed"])
def test_compression_vs_transfer_requires_gpu_when_unavailable(mode, monkeypatch):
    if gpu_available():
        pytest.skip("this test exercises the no-GPU error path specifically")
    from thermal.workload import UnsupportedHardwareError

    workload_cls = WorkloadRegistry.get("compression_vs_transfer")
    instance = workload_cls({"mode": mode, "size_mb": 1})
    with pytest.raises(UnsupportedHardwareError):
        instance.setup()


@requires_gpu
@pytest.mark.parametrize("mode", ["raw", "compressed"])
def test_compression_vs_transfer_runs_on_real_gpu(mode):
    workload_cls = WorkloadRegistry.get("compression_vs_transfer")
    instance = workload_cls({"mode": mode, "size_mb": 1})
    instance.spec = instance.spec.__class__(
        **{**instance.spec.__dict__, "warmup_iterations": 1, "measurement_iterations": 2}
    )
    result = run_workload(instance)
    assert result.device == "cuda:0"
    for metric in result.per_iteration_metrics:
        assert metric["duration_seconds"] > 0
        assert metric["mode"] == mode


def test_compression_vs_transfer_rejects_invalid_mode():
    # mode is validated before the GPU check, so this raises ValueError
    # regardless of whether this machine has a GPU.
    workload_cls = WorkloadRegistry.get("compression_vs_transfer")
    instance = workload_cls({"mode": "not_a_real_mode", "size_mb": 1})
    with pytest.raises(ValueError):
        instance.setup()
