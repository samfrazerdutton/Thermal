"""Transformer inference workload: a real (synthetic-weights) transformer
encoder stack forward pass, reporting tokens/sec. No pretrained weights are
downloaded -- the model is randomly initialized, which is sufficient to
measure compute/memory characteristics honestly (the numbers this produces
are throughput, not model quality)."""

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
class TransformerInferenceWorkload(Workload):
    spec = WorkloadSpec(
        name="transformer_inference",
        version="1.0.0",
        description="Forward pass through a randomly-initialized transformer encoder stack; reports tokens/sec.",
        hardware_requirements=HardwareRequirement(gpu_required=False),
        default_parameters={
            "batch_size": 8,
            "seq_length": 256,
            "hidden_dim": 512,
            "num_layers": 6,
            "num_heads": 8,
            "dtype": "float32",
        },
        warmup_iterations=3,
        measurement_iterations=10,
        output_metrics=("duration_seconds", "tokens_per_sec"),
    )

    def setup(self) -> None:
        import torch
        import torch.nn as nn

        self.device = select_device(prefer_gpu=True)
        dtype = getattr(torch, self.params.get("dtype", "float32"))

        self.batch_size = int(self.params["batch_size"])
        self.seq_length = int(self.params["seq_length"])
        hidden_dim = int(self.params["hidden_dim"])
        num_layers = int(self.params["num_layers"])
        num_heads = int(self.params["num_heads"])

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dtype=dtype,
        )
        self.model = nn.TransformerEncoder(layer, num_layers=num_layers).to(self.device)
        self.model.eval()

        self.inputs = torch.randn(self.batch_size, self.seq_length, hidden_dim, dtype=dtype, device=self.device)
        self._torch = torch

    def run_once(self) -> dict[str, Any]:
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model(self.inputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            _ = output.sum().item()
        elapsed = time.perf_counter() - start

        num_tokens = self.batch_size * self.seq_length
        return {
            "device": self.device,
            "duration_seconds": elapsed,
            "tokens_per_sec": num_tokens / elapsed if elapsed > 0 else None,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
        }

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
