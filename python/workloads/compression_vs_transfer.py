"""Compression-vs-transfer workload: measures whether compressing a buffer
before a host<->device round trip is actually faster than transferring it
raw -- per the spec's explicit instruction not to assume compression helps,
but to measure it. `mode=raw` times a plain H2D+D2H round trip; `mode=compressed`
times zlib-compress (CPU) -> H2D -> D2H -> zlib-decompress (CPU), the full
pipeline cost, not just the transfer. Compare with:

    thermal experiment run compression_vs_transfer \\
        --baseline-param mode=raw --treatment-param mode=compressed \\
        --metric duration_seconds --lower-is-better

Requires a CUDA device -- there is no host<->device transfer to measure
without one, so setup() raises UnsupportedHardwareError rather than faking
a number on CPU-only machines.
"""

from __future__ import annotations

import time
import zlib
from typing import Any

import numpy as np

from thermal.workload import (
    HardwareRequirement,
    UnsupportedHardwareError,
    Workload,
    WorkloadSpec,
    register_workload,
)


@register_workload
class CompressionVsTransferWorkload(Workload):
    spec = WorkloadSpec(
        name="compression_vs_transfer",
        version="1.0.0",
        description="Raw H2D+D2H round trip vs compress->transfer->decompress; reports total pipeline duration_seconds.",
        hardware_requirements=HardwareRequirement(gpu_required=True, cuda_required=True),
        default_parameters={"mode": "raw", "size_mb": 64},
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds",),
    )

    def setup(self) -> None:
        import torch

        self.mode = self.params.get("mode", "raw")
        if self.mode not in ("raw", "compressed"):
            raise ValueError(f"mode must be 'raw' or 'compressed', got {self.mode!r}")

        if not torch.cuda.is_available():
            raise UnsupportedHardwareError(
                "compression_vs_transfer requires a CUDA-capable GPU -- none is available on this "
                "machine (torch.cuda.is_available() is False). There is no host<->device transfer to "
                "measure without one."
            )

        size_bytes = int(self.params["size_mb"] * 1024 * 1024)
        # Random float32 data compresses poorly (realistic and honest -- real
        # telemetry/tensor data often doesn't compress well either; this
        # workload should not be rigged to make compression look good).
        rng = np.random.default_rng(0)
        self.raw_array = rng.standard_normal(size_bytes // 4).astype(np.float32)
        self.raw_bytes = size_bytes
        self._torch = torch

        if self.mode == "compressed":
            self.compressed = zlib.compress(self.raw_array.tobytes(), level=6)

    def run_once(self) -> dict[str, Any]:
        torch = self._torch

        if self.mode == "raw":
            torch.cuda.synchronize()
            start = time.perf_counter()
            device_buf = torch.from_numpy(self.raw_array).to("cuda", non_blocking=False)
            host_back = device_buf.cpu()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            _ = host_back

            return {
                "device": "cuda:0",
                "duration_seconds": elapsed,
                "mode": "raw",
                "bytes_transferred": self.raw_bytes,
            }

        start = time.perf_counter()
        compressed = zlib.compress(self.raw_array.tobytes(), level=6)
        compressed_np = np.frombuffer(compressed, dtype=np.uint8)
        device_buf = torch.from_numpy(compressed_np.copy()).to("cuda", non_blocking=False)
        host_back = device_buf.cpu().numpy().tobytes()
        decompressed = zlib.decompress(host_back)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        _ = decompressed

        return {
            "device": "cuda:0",
            "duration_seconds": elapsed,
            "mode": "compressed",
            "bytes_transferred": len(compressed),
            "compression_ratio": self.raw_bytes / len(compressed) if compressed else None,
        }

    def teardown(self) -> None:
        self.raw_array = None
