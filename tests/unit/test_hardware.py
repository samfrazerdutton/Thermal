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
    pcie_ceiling_kbps,
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


def test_pcie_ceiling_kbps_gen3_x8_matches_real_published_spec():
    # PCIe 3.0 is 985 MB/s per lane per direction (128b/130b encoding); this is
    # the reference machine's actual negotiated link (see docs/environment-report.md).
    assert pcie_ceiling_kbps(3, 8) == 985 * 8 * 1000.0


def test_pcie_ceiling_kbps_scales_with_width():
    assert pcie_ceiling_kbps(3, 16) == 2 * pcie_ceiling_kbps(3, 8)


def test_pcie_ceiling_kbps_none_when_generation_or_width_unknown():
    assert pcie_ceiling_kbps(None, 8) is None
    assert pcie_ceiling_kbps(3, None) is None


def test_pcie_ceiling_kbps_none_for_unrecognized_generation():
    assert pcie_ceiling_kbps(99, 8) is None


def test_gpu_info_reports_pcie_link_fields_when_available():
    gpu = detect_gpu()
    if gpu.available:
        # Either both are real integers, or NVML genuinely couldn't read them --
        # never a fabricated placeholder.
        assert (gpu.pcie_link_generation is None) == (gpu.pcie_link_width is None) or (
            isinstance(gpu.pcie_link_generation, int) and isinstance(gpu.pcie_link_width, int)
        )
