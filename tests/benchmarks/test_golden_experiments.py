"""Golden experiments (docs/roadmap.md Phase 40 / original design spec section 40):
workloads deliberately constructed to sit in one specific bottleneck regime, used
to validate -- and, where they exposed real gaps, to calibrate -- the deterministic
classifier in thermal/diagnosis.py.

Each scenario runs a continuous GPU-saturating burst for a fixed real duration
(not a fixed iteration count) so telemetry sampling reliably lands mid-activity
regardless of any single kernel's short duration -- the same class of problem
documented in README.md's Limitations section for `thermal workload run` on a
fast, short workload.

These tests found two real, reproducible calibration bugs during development
(both now fixed in thermal/diagnosis.py, with unit tests in test_diagnosis.py
pinning the fixed behavior down):

1. The MEMORY_BOUND rule required gpu_utilization_percent < 40, but a genuinely
   memory-bound, low-arithmetic-intensity kernel measured ~88-100% SM utilization
   *and* ~87-100% memory-bandwidth utilization simultaneously -- NVML's SM
   utilization reads high whenever a kernel is resident, including while stalled
   on memory, so it is not evidence against memory-bound.
2. The TRANSFER_BOUND threshold (70% of the PCIe link ceiling) was never reached
   by a real pageable-memory H2D/D2H transfer, which plateaued around 58-62%
   because pageable transfers need an extra host-side staging copy pinned memory
   avoids. Lowered to 50%.

Marked slow (each burst runs for ~2 real seconds) and skipped outright when no
CUDA device is present -- these are GPU-specific by construction, not something
to fake on CPU-only hardware.
"""

from __future__ import annotations

import time

import pytest

from thermal.device import gpu_available
from thermal.diagnosis import BottleneckClass, classify, features_from_telemetry
from thermal.hardware import collect_hardware_report, pcie_ceiling_kbps
from thermal.telemetry import TelemetryCollector

pytestmark = [
    pytest.mark.skipif(not gpu_available(), reason="golden experiments for GPU bottleneck classes require a real CUDA device"),
]

BURST_SECONDS = 2.0


@pytest.fixture()
def hardware_context():
    report = collect_hardware_report()
    ceiling = pcie_ceiling_kbps(report.gpu.pcie_link_generation, report.gpu.pcie_link_width)
    yield report, ceiling
    import torch

    torch.cuda.empty_cache()  # avoid one golden experiment's allocations inflating the next's memory_used_ratio


def _classify_burst(samples, hardware_context) -> BottleneckClass:
    report, ceiling = hardware_context
    features = features_from_telemetry(samples, total_memory_mb=report.gpu.memory_total_mb, pcie_ceiling_kbps=ceiling)
    return classify(features).bottleneck


def test_compute_bound_golden_experiment(hardware_context):
    """Continuous large matmul: high arithmetic intensity, should read
    high SM utilization and low memory-bandwidth utilization."""
    import torch

    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    telemetry = TelemetryCollector(interval_seconds=0.03)
    telemetry.start()
    start = time.perf_counter()
    while time.perf_counter() - start < BURST_SECONDS:
        c = a @ b
    torch.cuda.synchronize()
    telemetry.stop()

    bottleneck = _classify_burst(telemetry.samples, hardware_context)
    assert bottleneck == BottleneckClass.COMPUTE_BOUND

    del a, b, c


def test_memory_bound_golden_experiment(hardware_context):
    """Continuous large elementwise op: low arithmetic intensity (2 FLOPs per
    12 bytes touched), should read high memory-bandwidth utilization even
    though SM utilization also reads high (see module docstring)."""
    import torch

    x = torch.randn(100_000_000, device="cuda")
    y = torch.randn(100_000_000, device="cuda")
    telemetry = TelemetryCollector(interval_seconds=0.03)
    telemetry.start()
    start = time.perf_counter()
    while time.perf_counter() - start < BURST_SECONDS:
        z = x * y + x
    torch.cuda.synchronize()
    telemetry.stop()

    bottleneck = _classify_burst(telemetry.samples, hardware_context)
    assert bottleneck == BottleneckClass.MEMORY_BOUND

    del x, y, z


def test_transfer_bound_golden_experiment(hardware_context):
    """Continuous H2D+D2H round trips: should saturate the PCIe link (relative
    to this GPU's actual negotiated link) while GPU compute stays moderate.

    This measurement sits closer to its classification threshold than the
    compute/memory golden experiments do (observed 58-62% against a 50%
    threshold in isolated runs), so it is more sensitive to real system noise
    -- e.g. thermal state left over from the compute/memory bursts that ran
    immediately before it in the same test session. Retried up to twice with
    a short cooldown for that reason; a real, reproducible transfer-bound
    reading either shows up within that budget or the calibration in
    thermal/diagnosis.py needs another look, which is exactly what this test
    existing is for.
    """
    import torch

    last_bottleneck = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(1.0)  # let the GPU cool/settle between attempts

        host = torch.randn(100_000_000)
        device_buf = torch.empty(100_000_000, device="cuda")
        telemetry = TelemetryCollector(interval_seconds=0.03)
        telemetry.start()
        start = time.perf_counter()
        while time.perf_counter() - start < BURST_SECONDS:
            device_buf.copy_(host)
            back = device_buf.cpu()
        torch.cuda.synchronize()
        telemetry.stop()

        last_bottleneck = _classify_burst(telemetry.samples, hardware_context)
        del host, device_buf, back
        torch.cuda.empty_cache()

        if last_bottleneck == BottleneckClass.TRANSFER_BOUND:
            return

    assert last_bottleneck == BottleneckClass.TRANSFER_BOUND, (
        f"got {last_bottleneck} across 3 attempts -- see thermal/diagnosis.py's _HIGH_PCIE_UTIL calibration"
    )


def test_latency_bound_golden_experiment(hardware_context):
    """A tiny, fast op run with realistic iteration counts (not a continuous
    burst): neither compute nor memory bandwidth should saturate."""
    import torch

    from thermal.telemetry import TelemetryCollector as _TC

    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    telemetry = _TC(interval_seconds=0.2)  # the CLI's real default interval
    telemetry.start()
    for _ in range(10):
        c = a @ b
        torch.cuda.synchronize()
        time.sleep(0.05)  # a small realistic gap, e.g. Python-side bookkeeping between iterations
    telemetry.stop()

    bottleneck = _classify_burst(telemetry.samples, hardware_context)
    assert bottleneck in (BottleneckClass.LATENCY_BOUND, BottleneckClass.UNKNOWN, BottleneckClass.MIXED)

    del a, b, c
