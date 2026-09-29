"""Unit tests for thermal.native_bench.

The native kernel binary is built separately (scripts/build-native.ps1, see
docs/environment-report.md for why it needs a non-default CMake generator on
Windows) and is not part of `pip install -e .`. Tests that need the real
binary skip with an explicit reason when it hasn't been built -- never faked.
"""

import pytest

from thermal.native_bench import (
    KERNEL_NAMES,
    KernelBenchNotBuiltError,
    achieved_bandwidth_gbps,
    achieved_gflops,
    is_built,
    run_kernel_bench,
)

requires_native_binary = pytest.mark.skipif(
    not is_built(),
    reason="native kernel bench binary not built; run scripts/build-native.ps1 (see docs/environment-report.md)",
)


def test_unknown_kernel_raises_value_error():
    with pytest.raises(ValueError):
        run_kernel_bench("not_a_real_kernel", n=1024)


def test_missing_binary_raises_not_built_error(monkeypatch, tmp_path):
    import thermal.native_bench as native_bench

    monkeypatch.setattr(native_bench, "_DEFAULT_BINARY", tmp_path / "does_not_exist.exe")
    with pytest.raises(KernelBenchNotBuiltError):
        run_kernel_bench("vector_add_naive", n=1024)


@requires_native_binary
@pytest.mark.parametrize("kernel", KERNEL_NAMES)
def test_each_kernel_runs_correctly_on_real_gpu(kernel):
    n = 256 if "matmul" in kernel else 1 << 16
    result = run_kernel_bench(kernel, n=n, iters=3, warmup=1)
    assert result.correct is True
    assert result.kernel == kernel
    assert len(result.iteration_times_ms) == 3
    assert all(t > 0 for t in result.iteration_times_ms)


@requires_native_binary
def test_shared_memory_reduction_faster_than_naive_atomic():
    n = 1 << 22
    naive = run_kernel_bench("reduce_naive_atomic", n=n, iters=5, warmup=2)
    shared = run_kernel_bench("reduce_shared_memory", n=n, iters=5, warmup=2)
    assert min(shared.iteration_times_ms) < min(naive.iteration_times_ms)


@requires_native_binary
def test_achieved_metrics_are_positive():
    result = run_kernel_bench("vector_add_naive", n=1 << 16, iters=3, warmup=1)
    mean_ms = sum(result.iteration_times_ms) / len(result.iteration_times_ms)
    assert achieved_gflops(result, mean_ms) > 0
    assert achieved_bandwidth_gbps(result, mean_ms) > 0


def test_achieved_gflops_none_for_zero_time():
    from thermal.native_bench import KernelBenchResult

    result = KernelBenchResult(
        kernel="vector_add_naive", device="test", n=1, block_size=1,
        correct=True, bytes_moved=100.0, flops=100.0, iteration_times_ms=[0.0],
    )
    assert achieved_gflops(result, 0.0) is None
    assert achieved_bandwidth_gbps(result, 0.0) is None
