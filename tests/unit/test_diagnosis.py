"""Unit tests for thermal.diagnosis (deterministic bottleneck classifier)."""

from thermal.diagnosis import BottleneckClass, WorkloadFeatures, classify, features_from_telemetry


def test_memory_bound_when_bandwidth_saturated_and_compute_idle():
    features = WorkloadFeatures(
        gpu_utilization_percent=37.0,
        gpu_memory_bandwidth_utilization_percent=91.0,
        arithmetic_intensity_flops_per_byte=2.0,
    )
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.MEMORY_BOUND
    assert diagnosis.confidence > 0.9
    assert any(e.feature == "gpu_memory_bandwidth_utilization_percent" for e in diagnosis.evidence)


def test_compute_bound_when_compute_high_bandwidth_low():
    features = WorkloadFeatures(gpu_utilization_percent=95.0, gpu_memory_bandwidth_utilization_percent=20.0)
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.COMPUTE_BOUND


def test_memory_bound_when_bandwidth_saturated_even_if_compute_also_high():
    # Calibrated from a real golden experiment (tests/benchmarks/test_golden_experiments.py):
    # a genuinely memory-bound elementwise kernel measured ~88-100% SM utilization
    # *and* ~87-100% memory-bandwidth utilization simultaneously on the reference GPU.
    # NVML's SM utilization is not a reliable signal for ruling out memory-bound --
    # this must still classify as MEMORY_BOUND, just at reduced confidence.
    features = WorkloadFeatures(gpu_utilization_percent=99.7, gpu_memory_bandwidth_utilization_percent=87.4)
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.MEMORY_BOUND
    assert 0.5 < diagnosis.confidence < 0.9  # lower than the compute-idle case above


def test_transfer_bound_threshold_calibrated_to_pageable_transfer_reality():
    # Calibrated from a real golden experiment: pageable H2D/D2H transfers on the
    # reference GPU (PCIe Gen3 x8) plateaued around 58-62% of the theoretical link
    # ceiling, never higher, because pageable transfers need an extra host-side
    # staging copy that pinned memory avoids. A threshold above this (the original
    # value was 70%) made a real, reproducible transfer-bound workload unreachable.
    features = WorkloadFeatures(gpu_utilization_percent=67.0, pcie_utilization_percent=58.0)
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.TRANSFER_BOUND


def test_cpu_bound_when_gpu_idle_and_bandwidth_low_and_cpu_saturated():
    features = WorkloadFeatures(
        gpu_utilization_percent=5.0,
        gpu_memory_bandwidth_utilization_percent=5.0,
        cpu_utilization_percent=95.0,
    )
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.CPU_BOUND


def test_latency_bound_when_everything_idle():
    features = WorkloadFeatures(
        gpu_utilization_percent=5.0,
        gpu_memory_bandwidth_utilization_percent=5.0,
        cpu_utilization_percent=10.0,
    )
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.LATENCY_BOUND


def test_transfer_bound_when_pcie_saturated_and_gpu_idle():
    features = WorkloadFeatures(pcie_utilization_percent=85.0, gpu_utilization_percent=10.0)
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.TRANSFER_BOUND
    assert diagnosis.confidence >= 0.8


def test_launch_overhead_takes_priority():
    features = WorkloadFeatures(
        kernel_launch_overhead_ratio=0.5,
        gpu_utilization_percent=90.0,
        gpu_memory_bandwidth_utilization_percent=10.0,
    )
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.LAUNCH_OVERHEAD


def test_capacity_bound_when_memory_nearly_full():
    features = WorkloadFeatures(memory_used_ratio=0.95)
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.CAPACITY_BOUND


def test_unknown_when_almost_nothing_measured():
    features = WorkloadFeatures()
    diagnosis = classify(features)
    assert diagnosis.bottleneck == BottleneckClass.UNKNOWN
    assert diagnosis.confidence == 0.0
    assert len(diagnosis.missing_features) >= 4


def test_missing_features_never_silently_dropped():
    features = WorkloadFeatures(gpu_utilization_percent=50.0)
    diagnosis = classify(features)
    assert "cpu_utilization_percent" in diagnosis.missing_features


class _FakeGpuSample:
    def __init__(self, utilization, memory_utilization, memory_used_mb=None, pcie_tx_kbps=None, pcie_rx_kbps=None):
        self.utilization_percent = utilization
        self.memory_utilization_percent = memory_utilization
        self.memory_used_mb = memory_used_mb
        self.pcie_tx_kbps = pcie_tx_kbps
        self.pcie_rx_kbps = pcie_rx_kbps


class _FakeCpuSample:
    def __init__(self, utilization):
        self.utilization_percent = utilization


class _FakeSample:
    def __init__(self, gpu_util, gpu_mem_util, cpu_util, memory_used_mb=None, pcie_tx_kbps=None, pcie_rx_kbps=None):
        self.gpu = (
            _FakeGpuSample(gpu_util, gpu_mem_util, memory_used_mb, pcie_tx_kbps, pcie_rx_kbps)
            if gpu_util is not None
            else None
        )
        self.cpu = _FakeCpuSample(cpu_util)


def test_features_from_telemetry_averages_and_skips_none():
    samples = [
        _FakeSample(gpu_util=80.0, gpu_mem_util=10.0, cpu_util=20.0),
        _FakeSample(gpu_util=90.0, gpu_mem_util=20.0, cpu_util=30.0),
        _FakeSample(gpu_util=None, gpu_mem_util=None, cpu_util=40.0),
    ]
    features = features_from_telemetry(samples)
    assert features.gpu_utilization_percent == 85.0
    assert features.cpu_utilization_percent == 30.0


def test_features_from_telemetry_empty_samples_all_none():
    features = features_from_telemetry([])
    assert features.gpu_utilization_percent is None
    assert features.cpu_utilization_percent is None


def test_features_from_telemetry_computes_memory_used_ratio_when_total_given():
    samples = [
        _FakeSample(gpu_util=90.0, gpu_mem_util=50.0, cpu_util=20.0, memory_used_mb=3072.0),
        _FakeSample(gpu_util=90.0, gpu_mem_util=50.0, cpu_util=20.0, memory_used_mb=3072.0),
    ]
    features = features_from_telemetry(samples, total_memory_mb=6144.0)
    assert features.memory_used_ratio == 0.5


def test_features_from_telemetry_memory_used_ratio_none_without_total():
    samples = [_FakeSample(gpu_util=90.0, gpu_mem_util=50.0, cpu_util=20.0, memory_used_mb=3072.0)]
    features = features_from_telemetry(samples)  # no total_memory_mb given
    assert features.memory_used_ratio is None


def test_features_from_telemetry_computes_pcie_utilization_when_ceiling_given():
    samples = [
        _FakeSample(gpu_util=60.0, gpu_mem_util=20.0, cpu_util=20.0, pcie_tx_kbps=2_000_000.0, pcie_rx_kbps=2_000_000.0),
    ]
    features = features_from_telemetry(samples, pcie_ceiling_kbps=4_000_000.0)
    assert features.pcie_utilization_percent == 50.0


def test_features_from_telemetry_pcie_utilization_none_without_ceiling():
    samples = [_FakeSample(gpu_util=60.0, gpu_mem_util=20.0, cpu_util=20.0, pcie_tx_kbps=2_000_000.0, pcie_rx_kbps=2_000_000.0)]
    features = features_from_telemetry(samples)  # no pcie_ceiling_kbps given
    assert features.pcie_utilization_percent is None
