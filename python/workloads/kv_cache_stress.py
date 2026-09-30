"""KV-cache stress workload: simulates one autoregressive decode step's
attention computation against a KV-cache of a given length -- the operation
whose cost grows with context length in real transformer decoding. Measures
how single-token decode latency scales with kv_cache_length, the same
relationship the spec's memory-bound demonstration scenario is built on."""

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
class KVCacheStressWorkload(Workload):
    spec = WorkloadSpec(
        name="kv_cache_stress",
        version="1.0.0",
        description="One decode-step attention op (1 query token vs a kv_cache_length-token cache); reports duration and tokens/sec.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={
            "kv_cache_length": 4096,
            "num_heads": 8,
            "head_dim": 64,
            "batch_size": 1,
            "dtype": "float32",
            "device": "auto",
        },
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "tokens_per_sec"),
    )

    def setup(self) -> None:
        import torch

        self.device = select_device(prefer_gpu=True, override=self.params.get("device", "auto"))
        dtype = getattr(torch, self.params.get("dtype", "float32"))

        batch = int(self.params["batch_size"])
        heads = int(self.params["num_heads"])
        cache_len = int(self.params["kv_cache_length"])
        head_dim = int(self.params["head_dim"])

        # one new query token attending over the full existing cache
        self.q = torch.randn(batch, heads, 1, head_dim, dtype=dtype, device=self.device)
        self.k_cache = torch.randn(batch, heads, cache_len, head_dim, dtype=dtype, device=self.device)
        self.v_cache = torch.randn(batch, heads, cache_len, head_dim, dtype=dtype, device=self.device)
        self.scale = head_dim ** -0.5
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        scores = (self.q @ self.k_cache.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        out = weights @ self.v_cache
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            _ = out.sum().item()
        elapsed = time.perf_counter() - start

        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "tokens_per_sec": 1.0 / elapsed if elapsed > 0 else None,
            "kv_cache_length": self.params["kv_cache_length"],
        }

    def teardown(self) -> None:
        self.q = None
        self.k_cache = None
        self.v_cache = None
