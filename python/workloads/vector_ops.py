"""Elementwise vector operations workload: measures throughput of a fused
multiply-add over a large vector."""

from __future__ import annotations

import time
from typing import Any

from thermal.device import select_device
from thermal.workload import (
    HardwareRequirement,
    Workload,
    WorkloadSpec,
    register_workload,
)


@register_workload
class VectorOpsWorkload(Workload):
    spec = WorkloadSpec(
        name="vector_ops",
        version="1.0.0",
        description="Elementwise c = a * b + a over a large vector, reports GFLOP/s.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={"size_millions": 16, "dtype": "float32", "device": "auto"},
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "gflops"),
    )

    def setup(self) -> None:
        import torch

        self.device = select_device(prefer_gpu=True, override=self.params.get("device", "auto"))
        dtype = getattr(torch, self.params.get("dtype", "float32"))
        self.n = int(self.params["size_millions"] * 1_000_000)
        self.a = torch.randn(self.n, dtype=dtype, device=self.device)
        self.b = torch.randn(self.n, dtype=dtype, device=self.device)
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        c = self.a * self.b + self.a
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            _ = c.sum().item()
        elapsed = time.perf_counter() - start

        flops = 2 * self.n  # one multiply + one add per element
        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "gflops": (flops / elapsed) / 1e9 if elapsed > 0 else None,
            "n": self.n,
        }

    def teardown(self) -> None:
        self.a = None
        self.b = None
