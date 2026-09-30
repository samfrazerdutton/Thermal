"""Batch-size scaling workload: a batched linear-layer forward pass
(input @ weight), the standard operation whose per-sample throughput
typically improves with batch size up to a hardware-dependent saturation
point. Compare batch sizes with `thermal experiment run` or
`thermal optimize --param batch_size --values ...` -- this workload doesn't
sweep batch size itself, it exposes it as a parameter like every other
workload."""

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
class BatchSizeScalingWorkload(Workload):
    spec = WorkloadSpec(
        name="batch_size_scaling",
        version="1.0.0",
        description="Batched linear layer forward pass (batch_size x in_features) @ (in_features x out_features); reports samples/sec.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={
            "batch_size": 32,
            "in_features": 4096,
            "out_features": 4096,
            "dtype": "float32",
        },
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "samples_per_sec", "gflops"),
    )

    def setup(self) -> None:
        import torch

        self.device = select_device(prefer_gpu=True)
        dtype = getattr(torch, self.params.get("dtype", "float32"))

        self.batch_size = int(self.params["batch_size"])
        self.in_features = int(self.params["in_features"])
        self.out_features = int(self.params["out_features"])

        self.input = torch.randn(self.batch_size, self.in_features, dtype=dtype, device=self.device)
        self.weight = torch.randn(self.in_features, self.out_features, dtype=dtype, device=self.device)
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        output = self.input @ self.weight
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            _ = output.sum().item()
        elapsed = time.perf_counter() - start

        flops = 2 * self.batch_size * self.in_features * self.out_features
        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "samples_per_sec": self.batch_size / elapsed if elapsed > 0 else None,
            "gflops": (flops / elapsed) / 1e9 if elapsed > 0 else None,
            "batch_size": self.batch_size,
        }

    def teardown(self) -> None:
        self.input = None
        self.weight = None
