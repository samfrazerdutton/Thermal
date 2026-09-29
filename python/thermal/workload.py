"""Workload plugin architecture.

A Workload declares its hardware requirements and measurement policy up front,
then exposes setup/run_once/teardown. The runner is responsible for warmup,
device-availability checks, and telemetry -- a workload only ever reports
metrics it actually measured, including which device it actually ran on
(never the device it merely wished it could use).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class HardwareRequirement:
    gpu_required: bool = False
    cuda_required: bool = False
    min_vram_mb: Optional[int] = None


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    version: str
    description: str
    hardware_requirements: HardwareRequirement
    default_parameters: dict[str, Any] = field(default_factory=dict)
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    output_metrics: tuple[str, ...] = ()


class UnsupportedHardwareError(RuntimeError):
    """Raised by setup() when the required hardware/backend genuinely isn't present."""


class Workload(ABC):
    spec: WorkloadSpec

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = {**self.spec.default_parameters, **(params or {})}

    @abstractmethod
    def setup(self) -> None:
        """Allocate inputs. Raise UnsupportedHardwareError if requirements aren't met."""

    @abstractmethod
    def run_once(self) -> dict[str, Any]:
        """Execute one measured iteration and return a metrics dict.

        Must include a "device" key stating what actually executed
        ("cpu" / "cuda:0" / ...) -- never claim GPU execution that
        didn't happen.
        """

    def teardown(self) -> None:  # pragma: no cover - default no-op
        pass


class WorkloadRegistry:
    _workloads: dict[str, type[Workload]] = {}

    @classmethod
    def register(cls, workload_cls: type[Workload]) -> type[Workload]:
        name = workload_cls.spec.name
        if name in cls._workloads and cls._workloads[name] is not workload_cls:
            raise ValueError(f"workload '{name}' already registered")
        cls._workloads[name] = workload_cls
        return workload_cls

    @classmethod
    def get(cls, name: str) -> type[Workload]:
        if name not in cls._workloads:
            raise KeyError(f"unknown workload '{name}'. Known: {sorted(cls._workloads)}")
        return cls._workloads[name]

    @classmethod
    def list(cls) -> list[WorkloadSpec]:
        return [w.spec for w in cls._workloads.values()]


def register_workload(workload_cls: type[Workload]) -> type[Workload]:
    return WorkloadRegistry.register(workload_cls)


@dataclass
class WorkloadRunResult:
    spec_name: str
    device: str
    warmup_iterations: int
    measurement_iterations: int
    per_iteration_metrics: list[dict[str, Any]]


def run_workload(
    workload: Workload,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> WorkloadRunResult:
    """Run warmup + measured iterations. Warmup results are discarded, not hidden --
    they simply aren't part of the returned measurement set (see docs/roadmap.md
    Phase 4 for the statistical baseline engine built on top of this).

    on_event(name, payload), if given, is called synchronously at each stage
    (warmup_started, warmup_iteration, measurement_started,
    iteration_completed) -- Phase 14's live streaming forwards these over a
    WebSocket, but this function has no idea that's happening; it just emits.
    """
    def emit(name: str, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(name, payload)

    workload.setup()
    try:
        emit("warmup_started", {"count": workload.spec.warmup_iterations})
        for i in range(workload.spec.warmup_iterations):
            workload.run_once()
            emit("warmup_iteration", {"index": i})

        emit("measurement_started", {"count": workload.spec.measurement_iterations})
        results = []
        for i in range(workload.spec.measurement_iterations):
            metrics = workload.run_once()
            results.append(metrics)
            emit("iteration_completed", {"index": i, "metrics": metrics})
    finally:
        workload.teardown()

    devices = {r.get("device", "unknown") for r in results}
    device = devices.pop() if len(devices) == 1 else "mixed:" + ",".join(sorted(devices))

    return WorkloadRunResult(
        spec_name=workload.spec.name,
        device=device,
        warmup_iterations=workload.spec.warmup_iterations,
        measurement_iterations=workload.spec.measurement_iterations,
        per_iteration_metrics=results,
    )
