"""Real, honest device selection shared by workloads.

Never claims GPU execution unless a CUDA-capable framework is actually
present and working on this machine.
"""

from __future__ import annotations


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def select_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and gpu_available():
        return "cuda:0"
    return "cpu"
