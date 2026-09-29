"""Built-in THERMAL workloads. Importing this package registers them all."""

from . import matmul, memory_bandwidth, vector_ops  # noqa: F401

__all__ = ["matmul", "memory_bandwidth", "vector_ops"]
