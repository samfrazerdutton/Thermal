"""CPU/GPU transfer workload: measures real host<->device copy bandwidth in
both directions. Genuinely requires a CUDA device -- there is no meaningful
"transfer" to measure on a CPU-only machine, so setup() raises
UnsupportedHardwareError there rather than faking a number."""

from __future__ import annotations

import time
from typing import Any

from thermal.workload import (
    HardwareRequirement,
    UnsupportedHardwareError,
    Workload,
    WorkloadSpec,
    register_workload,
)


@register_workload
class CPUGPUTransferWorkload(Workload):
    spec = WorkloadSpec(
        name="cpu_gpu_transfer",
        version="1.0.0",
        description="Host<->device copy of a pinned or pageable buffer; reports H2D and D2H bandwidth in GB/s.",
        hardware_requirements=HardwareRequirement(gpu_required=True, cuda_required=True),
        default_parameters={"size_mb": 256, "pinned": False, "dtype": "float32"},
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("h2d_bandwidth_gbps", "d2h_bandwidth_gbps", "duration_seconds"),
    )

    def setup(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise UnsupportedHardwareError(
                "cpu_gpu_transfer requires a CUDA-capable GPU -- none is available on this machine "
                "(torch.cuda.is_available() is False). There is no host<->device transfer to measure without one."
            )

        dtype = getattr(torch, self.params.get("dtype", "float32"))
        bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
        self.num_elements = int(self.params["size_mb"] * 1024 * 1024 / bytes_per_elem)
        self.total_bytes = self.num_elements * bytes_per_elem
        self.pinned = bool(self.params.get("pinned", False))

        self.host_buf = torch.randn(self.num_elements, dtype=dtype, pin_memory=self.pinned)
        self.device_buf = torch.empty(self.num_elements, dtype=dtype, device="cuda")
        self.host_dst = torch.empty(self.num_elements, dtype=dtype, pin_memory=self.pinned)
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch

        torch.cuda.synchronize()
        start = time.perf_counter()
        self.device_buf.copy_(self.host_buf, non_blocking=self.pinned)
        torch.cuda.synchronize()
        h2d_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        self.host_dst.copy_(self.device_buf, non_blocking=self.pinned)
        torch.cuda.synchronize()
        d2h_elapsed = time.perf_counter() - start

        return {
            "device": "cuda:0",
            "duration_seconds": h2d_elapsed + d2h_elapsed,
            "h2d_bandwidth_gbps": (self.total_bytes / h2d_elapsed) / 1e9 if h2d_elapsed > 0 else None,
            "d2h_bandwidth_gbps": (self.total_bytes / d2h_elapsed) / 1e9 if d2h_elapsed > 0 else None,
            "pinned": self.pinned,
        }

    def teardown(self) -> None:
        self.host_buf = None
        self.device_buf = None
        self.host_dst = None
