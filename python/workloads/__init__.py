"""Built-in THERMAL workloads. Importing this package registers them all."""

from . import (  # noqa: F401
    batch_size_scaling,
    compression_vs_transfer,
    cpu_gpu_transfer,
    kv_cache_stress,
    matmul,
    memory_bandwidth,
    transformer_inference,
    vector_ops,
)

__all__ = [
    "batch_size_scaling",
    "compression_vs_transfer",
    "cpu_gpu_transfer",
    "kv_cache_stress",
    "matmul",
    "memory_bandwidth",
    "transformer_inference",
    "vector_ops",
]
