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
    def __init__(self, utilization, memory_utilization):
        self.utilization_percent = utilization
        self.memory_utilization_percent = memory_utilization


class _FakeCpuSample:
    def __init__(self, utilization):
        self.utilization_percent = utilization


class _FakeSample:
    def __init__(self, gpu_util, gpu_mem_util, cpu_util):
        self.gpu = _FakeGpuSample(gpu_util, gpu_mem_util) if gpu_util is not None else None
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
