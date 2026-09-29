"""Deterministic bottleneck classifier.

Feature-based, rule-driven, no AI. Given measured hardware/workload features,
it assigns a bottleneck class with an explicit confidence and, critically, the
evidence (the actual feature values) behind the call -- so the verdict can be
audited, not just trusted. Thresholds are documented inline rather than
buried as magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BottleneckClass(str, Enum):
    COMPUTE_BOUND = "COMPUTE_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"
    TRANSFER_BOUND = "TRANSFER_BOUND"
    LATENCY_BOUND = "LATENCY_BOUND"
    SYNCHRONIZATION_BOUND = "SYNCHRONIZATION_BOUND"
    CPU_BOUND = "CPU_BOUND"
    GPU_BOUND = "GPU_BOUND"
    IO_BOUND = "IO_BOUND"
    CAPACITY_BOUND = "CAPACITY_BOUND"
    LAUNCH_OVERHEAD = "LAUNCH_OVERHEAD"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class WorkloadFeatures:
    """Measured features a classification decision is based on. Every field is
    Optional -- a feature THERMAL couldn't measure on this backend must be
    passed as None, never estimated, and the classifier accounts for that by
    lowering confidence rather than guessing."""

    gpu_utilization_percent: Optional[float] = None
    gpu_memory_bandwidth_utilization_percent: Optional[float] = None
    cpu_utilization_percent: Optional[float] = None
    pcie_utilization_percent: Optional[float] = None
    arithmetic_intensity_flops_per_byte: Optional[float] = None
    kernel_launch_overhead_ratio: Optional[float] = None  # launch overhead / total kernel time
    memory_used_ratio: Optional[float] = None  # used / total device memory
    io_wait_ratio: Optional[float] = None


@dataclass(frozen=True)
class Evidence:
    feature: str
    value: float
    threshold: float
    comparison: str  # e.g. ">=", "<"


@dataclass(frozen=True)
class Diagnosis:
    bottleneck: BottleneckClass
    confidence: float  # 0..1, heuristic strength of the rule that fired, not a probability
    evidence: list[Evidence] = field(default_factory=list)
    rationale: str = ""
    missing_features: list[str] = field(default_factory=list)


# Thresholds. These are first-generation heuristics (see docs/roadmap.md Phase 6) --
# tuned to be conservative rather than confidently wrong; refine against the golden
# experiments in Phase 40 (tests/benchmarks) rather than adjusting blindly.
_HIGH_UTIL = 80.0
_LOW_UTIL = 40.0
_HIGH_BANDWIDTH_UTIL = 80.0
_LOW_ARITHMETIC_INTENSITY = 10.0  # FLOPs/byte; below this, most hardware is bandwidth-limited
_HIGH_ARITHMETIC_INTENSITY = 50.0
_HIGH_LAUNCH_OVERHEAD_RATIO = 0.30
_HIGH_MEMORY_USED_RATIO = 0.90
_HIGH_IO_WAIT_RATIO = 0.30
_HIGH_PCIE_UTIL = 70.0


def features_from_telemetry(samples: list) -> WorkloadFeatures:
    """Average a list of thermal.telemetry.TelemetrySample into WorkloadFeatures.

    NVML's "memory utilization" is the fraction of time the memory controller
    was active -- a genuine bandwidth-utilization proxy, not a fabricated
    stand-in. A feature is left None (not zero) when no sample ever measured
    it, so the classifier can tell "measured as zero" apart from "unmeasurable".
    """
    def _avg(getter):
        values = [v for s in samples if (v := getter(s)) is not None]
        return sum(values) / len(values) if values else None

    return WorkloadFeatures(
        gpu_utilization_percent=_avg(lambda s: s.gpu.utilization_percent if s.gpu else None),
        gpu_memory_bandwidth_utilization_percent=_avg(lambda s: s.gpu.memory_utilization_percent if s.gpu else None),
        cpu_utilization_percent=_avg(lambda s: s.cpu.utilization_percent),
        memory_used_ratio=None,  # requires total device memory context; computed by callers that have it
        pcie_utilization_percent=None,  # NVML reports throughput (KB/s), not a %; needs a device-specific ceiling to normalize
    )


def classify(features: WorkloadFeatures) -> Diagnosis:
    missing = [name for name, value in vars(features).items() if value is None]

    # Rule order matters: more specific / higher-evidence rules first.
    if features.kernel_launch_overhead_ratio is not None and features.kernel_launch_overhead_ratio >= _HIGH_LAUNCH_OVERHEAD_RATIO:
        return Diagnosis(
            bottleneck=BottleneckClass.LAUNCH_OVERHEAD,
            confidence=0.85,
            evidence=[Evidence("kernel_launch_overhead_ratio", features.kernel_launch_overhead_ratio, _HIGH_LAUNCH_OVERHEAD_RATIO, ">=")],
            rationale="Launch overhead accounts for a large share of kernel time -- many small kernels dominate over actual compute.",
            missing_features=missing,
        )

    if features.memory_used_ratio is not None and features.memory_used_ratio >= _HIGH_MEMORY_USED_RATIO:
        return Diagnosis(
            bottleneck=BottleneckClass.CAPACITY_BOUND,
            confidence=0.8,
            evidence=[Evidence("memory_used_ratio", features.memory_used_ratio, _HIGH_MEMORY_USED_RATIO, ">=")],
            rationale="Device memory is nearly full -- allocation pressure and paging/eviction likely dominate.",
            missing_features=missing,
        )

    if features.io_wait_ratio is not None and features.io_wait_ratio >= _HIGH_IO_WAIT_RATIO:
        return Diagnosis(
            bottleneck=BottleneckClass.IO_BOUND,
            confidence=0.75,
            evidence=[Evidence("io_wait_ratio", features.io_wait_ratio, _HIGH_IO_WAIT_RATIO, ">=")],
            rationale="A large share of wall-clock time is spent waiting on I/O.",
            missing_features=missing,
        )

    if features.pcie_utilization_percent is not None and features.pcie_utilization_percent >= _HIGH_PCIE_UTIL:
        gpu_low = features.gpu_utilization_percent is not None and features.gpu_utilization_percent < _LOW_UTIL
        return Diagnosis(
            bottleneck=BottleneckClass.TRANSFER_BOUND,
            confidence=0.85 if gpu_low else 0.65,
            evidence=[Evidence("pcie_utilization_percent", features.pcie_utilization_percent, _HIGH_PCIE_UTIL, ">=")],
            rationale="Host<->device transfer (PCIe) is saturated" + (", while the GPU itself sits idle." if gpu_low else "."),
            missing_features=missing,
        )

    if (
        features.gpu_memory_bandwidth_utilization_percent is not None
        and features.gpu_utilization_percent is not None
    ):
        bw = features.gpu_memory_bandwidth_utilization_percent
        compute = features.gpu_utilization_percent
        if bw >= _HIGH_BANDWIDTH_UTIL and compute < _LOW_UTIL:
            ai_note = ""
            confidence = 0.9
            if features.arithmetic_intensity_flops_per_byte is not None:
                ai_note = f" Arithmetic intensity ({features.arithmetic_intensity_flops_per_byte:.2g} FLOPs/byte) is low, consistent with a bandwidth-limited kernel."
                if features.arithmetic_intensity_flops_per_byte < _LOW_ARITHMETIC_INTENSITY:
                    confidence = 0.95
            return Diagnosis(
                bottleneck=BottleneckClass.MEMORY_BOUND,
                confidence=confidence,
                evidence=[
                    Evidence("gpu_memory_bandwidth_utilization_percent", bw, _HIGH_BANDWIDTH_UTIL, ">="),
                    Evidence("gpu_utilization_percent", compute, _LOW_UTIL, "<"),
                ],
                rationale="HBM bandwidth is saturated while SM/compute utilization stays low." + ai_note,
                missing_features=missing,
            )
        if compute >= _HIGH_UTIL and bw < _LOW_UTIL:
            return Diagnosis(
                bottleneck=BottleneckClass.COMPUTE_BOUND,
                confidence=0.9,
                evidence=[
                    Evidence("gpu_utilization_percent", compute, _HIGH_UTIL, ">="),
                    Evidence("gpu_memory_bandwidth_utilization_percent", bw, _LOW_UTIL, "<"),
                ],
                rationale="SM/compute utilization is high while memory bandwidth is not the limiter.",
                missing_features=missing,
            )
        if compute < _LOW_UTIL and bw < _LOW_UTIL:
            if features.cpu_utilization_percent is not None and features.cpu_utilization_percent >= _HIGH_UTIL:
                return Diagnosis(
                    bottleneck=BottleneckClass.CPU_BOUND,
                    confidence=0.7,
                    evidence=[
                        Evidence("gpu_utilization_percent", compute, _LOW_UTIL, "<"),
                        Evidence("cpu_utilization_percent", features.cpu_utilization_percent, _HIGH_UTIL, ">="),
                    ],
                    rationale="GPU sits mostly idle while the CPU is saturated -- host-side code is the limiter.",
                    missing_features=missing,
                )
            return Diagnosis(
                bottleneck=BottleneckClass.LATENCY_BOUND,
                confidence=0.55,
                evidence=[
                    Evidence("gpu_utilization_percent", compute, _LOW_UTIL, "<"),
                    Evidence("gpu_memory_bandwidth_utilization_percent", bw, _LOW_UTIL, "<"),
                ],
                rationale="Neither compute nor memory bandwidth is saturated -- likely dominated by synchronization, dependency chains, or small-batch latency.",
                missing_features=missing,
            )

    if features.gpu_utilization_percent is not None and features.cpu_utilization_percent is not None:
        if features.gpu_utilization_percent < _LOW_UTIL and features.cpu_utilization_percent >= _HIGH_UTIL:
            return Diagnosis(
                bottleneck=BottleneckClass.CPU_BOUND,
                confidence=0.6,
                evidence=[
                    Evidence("gpu_utilization_percent", features.gpu_utilization_percent, _LOW_UTIL, "<"),
                    Evidence("cpu_utilization_percent", features.cpu_utilization_percent, _HIGH_UTIL, ">="),
                ],
                rationale="GPU idle, CPU saturated.",
                missing_features=missing,
            )
        if features.gpu_utilization_percent >= _HIGH_UTIL:
            return Diagnosis(
                bottleneck=BottleneckClass.GPU_BOUND,
                confidence=0.5,
                evidence=[Evidence("gpu_utilization_percent", features.gpu_utilization_percent, _HIGH_UTIL, ">=")],
                rationale="GPU utilization is high; insufficient additional evidence to distinguish compute- vs memory-bound.",
                missing_features=missing,
            )

    if len(missing) >= 4:
        return Diagnosis(
            bottleneck=BottleneckClass.UNKNOWN,
            confidence=0.0,
            rationale="Too few features were measurable on this backend to classify.",
            missing_features=missing,
        )

    return Diagnosis(
        bottleneck=BottleneckClass.MIXED,
        confidence=0.3,
        rationale="Measured features don't clearly match a single bottleneck pattern.",
        missing_features=missing,
    )
