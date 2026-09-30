"""Real, honest device selection shared by workloads.

Never claims GPU execution unless a CUDA-capable framework is actually
present and working on this machine.
"""

from __future__ import annotations

from thermal.workload import UnsupportedHardwareError


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def select_device(prefer_gpu: bool = True, override: str = "auto") -> str:
    """override: "auto" (default behavior), "cpu" (force CPU even if a GPU is
    present -- useful for a direct CPU-vs-GPU comparison experiment), or
    "cuda:0" (force GPU; raises UnsupportedHardwareError if none is actually
    available rather than silently falling back to CPU, which would make an
    experiment's "device" column lie about what the user asked to compare).
    """
    if override == "cpu":
        return "cpu"
    if override.startswith("cuda"):
        if not gpu_available():
            raise UnsupportedHardwareError(
                f"device={override!r} was requested but no CUDA-capable GPU is available on this machine "
                "(torch.cuda.is_available() is False)."
            )
        return override
    if override != "auto":
        raise ValueError(f"device must be 'auto', 'cpu', or a 'cuda:N' device string, got {override!r}")

    if prefer_gpu and gpu_available():
        return "cuda:0"
    return "cpu"
