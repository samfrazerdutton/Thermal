"""Unit tests for thermal.workload plugin architecture."""

import pytest

from thermal.workload import (
    HardwareRequirement,
    UnsupportedHardwareError,
    Workload,
    WorkloadRegistry,
    WorkloadSpec,
    register_workload,
    run_workload,
)


class _DummyWorkload(Workload):
    spec = WorkloadSpec(
        name="__test_dummy__",
        version="0.0.1",
        description="test-only workload",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        warmup_iterations=2,
        measurement_iterations=3,
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.setup_calls = 0
        self.run_calls = 0
        self.teardown_calls = 0

    def setup(self):
        self.setup_calls += 1

    def run_once(self):
        self.run_calls += 1
        return {"device": "cpu", "value": self.run_calls}

    def teardown(self):
        self.teardown_calls += 1


class _UnsupportedWorkload(Workload):
    spec = WorkloadSpec(
        name="__test_unsupported__",
        version="0.0.1",
        description="always raises on setup",
        hardware_requirements=HardwareRequirement(gpu_required=True, cuda_required=True),
    )

    def setup(self):
        raise UnsupportedHardwareError("no CUDA-capable device on this machine")

    def run_once(self):  # pragma: no cover - never reached
        raise AssertionError("should not run without setup succeeding")


def test_registry_register_and_get_roundtrip():
    register_workload(_DummyWorkload)
    assert WorkloadRegistry.get("__test_dummy__") is _DummyWorkload


def test_registry_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        WorkloadRegistry.get("__does_not_exist__")


def test_run_workload_does_warmup_then_measurement_only():
    register_workload(_DummyWorkload)
    instance = _DummyWorkload()
    result = run_workload(instance)

    assert instance.setup_calls == 1
    assert instance.teardown_calls == 1
    assert instance.run_calls == 2 + 3  # warmup + measured
    assert len(result.per_iteration_metrics) == 3  # only measured iterations returned
    assert result.device == "cpu"


def test_unsupported_hardware_raises_not_silently_falls_back():
    register_workload(_UnsupportedWorkload)
    instance = _UnsupportedWorkload()
    with pytest.raises(UnsupportedHardwareError):
        run_workload(instance)


def test_workload_run_result_reports_mixed_devices():
    register_workload(_DummyWorkload)

    class _MixedDeviceWorkload(_DummyWorkload):
        spec = WorkloadSpec(
            name="__test_mixed__",
            version="0.0.1",
            description="reports different devices per iteration",
            hardware_requirements=HardwareRequirement(),
            warmup_iterations=0,
            measurement_iterations=2,
        )

        def run_once(self):
            self.run_calls += 1
            device = "cpu" if self.run_calls % 2 else "cuda:0"
            return {"device": device, "value": self.run_calls}

    register_workload(_MixedDeviceWorkload)
    result = run_workload(_MixedDeviceWorkload())
    assert result.device.startswith("mixed:")
