"""Unit tests for thermal.hardware detection.

These tests validate structure and honesty of the detection layer, not
specific hardware values (CI machines vary and may have no GPU at all).
"""

from thermal.hardware import (
    Capability,
    detect_cpu,
    detect_gpu,
    detect_software,
    detect_toolchain,
)


def test_capability_to_dict_roundtrip():
    cap = Capability(name="x", available=True, value=42, reason=None)
    assert cap.to_dict() == {"name": "x", "available": True, "value": 42, "reason": None}


def test_capability_unavailable_has_reason_not_fake_value():
    cap = Capability(name="nvlink", available=False, reason="not supported on this GPU")
    assert cap.value is None
    assert cap.reason is not None


def test_detect_cpu_returns_real_core_counts():
    cpu = detect_cpu()
    assert cpu.logical_cores is not None
    assert cpu.logical_cores >= 1
    assert cpu.name


def test_detect_gpu_never_fabricates_when_unavailable():
    gpu = detect_gpu()
    if not gpu.available:
        assert gpu.reason is not None
        assert gpu.name is None
        assert gpu.telemetry == {}
    else:
        assert gpu.name is not None
        assert gpu.memory_total_mb is not None and gpu.memory_total_mb > 0


def test_detect_toolchain_marks_missing_tools_with_reason():
    toolchain = detect_toolchain()
    for cap in (
        toolchain.cuda_toolkit,
        toolchain.msvc,
        toolchain.gcc,
        toolchain.cmake,
        toolchain.docker,
        toolchain.git,
        toolchain.nsight_systems,
        toolchain.nsight_compute,
    ):
        if not cap.available:
            assert cap.reason, f"{cap.name} marked unavailable with no reason"


def test_detect_software_reports_python_version():
    software = detect_software()
    assert software.python_version
    assert software.os_name
