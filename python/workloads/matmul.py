"""Matrix multiplication workload: measures achieved FLOPs on whatever device
is actually available (GPU via torch.cuda if present, else CPU via torch/numpy)."""

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
class MatMulWorkload(Workload):
    spec = WorkloadSpec(
        name="matmul",
        version="1.0.0",
        description="Dense matrix multiplication (N x N), reports achieved GFLOP/s.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={"size": 1024, "dtype": "float32"},
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "gflops"),
    )

    def setup(self) -> None:
        import torch

        self.device = select_device(prefer_gpu=True)
        self.size = int(self.params["size"])
        dtype = getattr(torch, self.params.get("dtype", "float32"))
        self.a = torch.randn(self.size, self.size, dtype=dtype, device=self.device)
        self.b = torch.randn(self.size, self.size, dtype=dtype, device=self.device)
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        c = self.a @ self.b
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            # force materialization so lazy/backend scheduling can't hide the work
            _ = c.sum().item()
        elapsed = time.perf_counter() - start

        flops = 2 * (self.size ** 3)  # multiply-add per output element
        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "gflops": (flops / elapsed) / 1e9 if elapsed > 0 else None,
            "size": self.size,
        }

    def teardown(self) -> None:
        self.a = None
        self.b = None
