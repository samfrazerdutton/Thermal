"""Bridge to the native CUDA kernel lab (native/cuda), built separately via
scripts/build-native.ps1 -- see docs/environment-report.md for why this can't
just be `cmake --build .` with the default Windows generator.

This module never fabricates a result when the binary hasn't been built: it
reports that plainly so the caller can tell "not built yet" apart from
"ran and failed".
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

KERNEL_NAMES = (
    "vector_add_naive",
    "vector_add_grid_stride",
    "reduce_naive_atomic",
    "reduce_shared_memory",
    "matmul_naive",
    "matmul_tiled",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BINARY = _REPO_ROOT / "build" / "native" / "native" / "cuda" / "thermal_kernel_bench.exe"


class KernelBenchNotBuiltError(RuntimeError):
    pass


class KernelBenchFailedError(RuntimeError):
    pass


@dataclass(frozen=True)
class KernelBenchResult:
    kernel: str
    device: str
    n: int
    block_size: int
    correct: bool
    bytes_moved: float
    flops: float
    iteration_times_ms: list[float]


def binary_path() -> Path:
    return _DEFAULT_BINARY


def is_built() -> bool:
    return binary_path().exists()


def run_kernel_bench(
    kernel: str,
    n: int,
    iters: int = 20,
    warmup: int = 5,
    block_size: int = 256,
    timeout: float = 60.0,
) -> KernelBenchResult:
    if kernel not in KERNEL_NAMES:
        raise ValueError(f"unknown kernel '{kernel}'. Known: {KERNEL_NAMES}")

    exe = binary_path()
    if not exe.exists():
        raise KernelBenchNotBuiltError(
            f"{exe} does not exist. Build it first: powershell -File scripts/build-native.ps1"
        )

    try:
        result = subprocess.run(
            [
                str(exe),
                "--kernel", kernel,
                "--n", str(n),
                "--iters", str(iters),
                "--warmup", str(warmup),
                "--block-size", str(block_size),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise KernelBenchFailedError(f"kernel bench timed out after {timeout}s: {exc}") from exc

    if result.returncode != 0:
        raise KernelBenchFailedError(
            f"kernel bench exited {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise KernelBenchFailedError(f"could not parse kernel bench output as JSON: {result.stdout}") from exc

    return KernelBenchResult(
        kernel=data["kernel"],
        device=data["device"],
        n=data["n"],
        block_size=data["block_size"],
        correct=data["correct"],
        bytes_moved=data["bytes_moved"],
        flops=data["flops"],
        iteration_times_ms=data["iteration_times_ms"],
    )


def achieved_gflops(result: KernelBenchResult, time_ms: float) -> Optional[float]:
    if time_ms <= 0:
        return None
    return (result.flops / (time_ms / 1000.0)) / 1e9


def achieved_bandwidth_gbps(result: KernelBenchResult, time_ms: float) -> Optional[float]:
    if time_ms <= 0:
        return None
    return (result.bytes_moved / (time_ms / 1000.0)) / 1e9
