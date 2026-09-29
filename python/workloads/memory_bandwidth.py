"""Memory bandwidth workload: measures achieved copy bandwidth for a large
buffer on whatever device is actually available."""

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
class MemoryBandwidthWorkload(Workload):
    spec = WorkloadSpec(
        name="memory_bandwidth",
        version="1.0.0",
        description="Large-buffer copy (a[:] = b), reports achieved bandwidth in GB/s.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={"size_mb": 256, "dtype": "float32"},
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "bandwidth_gbps"),
    )

    def setup(self) -> None:
        import torch

        self.device = select_device(prefer_gpu=True)
        dtype = getattr(torch, self.params.get("dtype", "float32"))
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        self.num_elements = int(self.params["size_mb"] * 1024 * 1024 / bytes_per_elem)
        self.src = torch.randn(self.num_elements, dtype=dtype, device=self.device)
        self.dst = torch.empty_like(self.src)
        self._torch = torch
        self._bytes_per_elem = bytes_per_elem

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        self.dst.copy_(self.src)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_bytes = self.num_elements * self._bytes_per_elem * 2  # read src + write dst
        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "bandwidth_gbps": (total_bytes / elapsed) / 1e9 if elapsed > 0 else None,
            "size_mb": self.params["size_mb"],
        }

    def teardown(self) -> None:
        self.src = None
        self.dst = None
