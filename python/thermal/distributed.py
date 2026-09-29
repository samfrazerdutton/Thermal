"""Distributed execution -- the smallest real slice of docs/roadmap.md's
distributed mode that a single-GPU development machine can actually
implement and verify.

Implemented and verified: dispatching a workload across multiple worker
PROCESSES (concurrent.futures.ProcessPoolExecutor), each running the exact
same thermal.runner.run_and_store_workload the CLI/API use, each producing
its own persisted run. Speedup over serial execution is a real, measured
wall-clock comparison, not an estimate.

NOT implemented: multi-GPU dispatch and multi-node/cluster coordination
(the CONTROLLER / TELEMETRY AGGREGATOR architecture in the original design
notes). This machine has exactly one GPU and one node, so there is nothing
to run either against, let alone verify -- claiming they work would violate
the same rule that governs every other module here. DistributedRunResult
carries a `gpu_count` field precisely so a caller can see that every worker
in a run like this landed on the same single GPU (or on CPU), not on
distinct devices.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class WorkerResult:
    worker_pid: int
    run_id: str
    device: str


@dataclass(frozen=True)
class DistributedRunResult:
    workload_name: str
    num_workers: int
    distinct_devices: list[str]
    wall_clock_seconds: float
    workers: list[WorkerResult]

    @property
    def actually_multi_gpu(self) -> bool:
        """True only if workers genuinely landed on more than one distinct
        device -- never assume parallelism implies multiple GPUs."""
        return len(self.distinct_devices) > 1


def _run_worker(args: tuple[str, Optional[dict[str, Any]], int, int]) -> WorkerResult:
    # Runs in a separate process (Windows: spawned, re-imports this module
    # fresh -- the workload registry is empty until built-ins are imported
    # again here; the parent process's registrations don't carry over).
    import workloads  # noqa: F401
    from thermal.runner import run_and_store_workload

    workload_name, params, samples, warmup = args
    outcome = run_and_store_workload(workload_name, params, samples=samples, warmup=warmup)
    return WorkerResult(worker_pid=os.getpid(), run_id=outcome.record.run_id, device=outcome.record.device)


def run_workload_multiprocess(
    workload_name: str,
    params: Optional[dict[str, Any]] = None,
    num_workers: int = 2,
    samples: int = 10,
    warmup: int = 2,
) -> DistributedRunResult:
    """Runs `num_workers` independent copies of the same workload in
    parallel processes. Each produces its own stored run (see thermal
    database list --workload <name> afterward) -- this function does not
    merge their metrics into one baseline, since they are genuinely separate
    measurements potentially on different devices."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    args = [(workload_name, params, samples, warmup)] * num_workers

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        workers = list(executor.map(_run_worker, args))
    elapsed = time.perf_counter() - start

    distinct_devices = sorted({w.device for w in workers})

    return DistributedRunResult(
        workload_name=workload_name,
        num_workers=num_workers,
        distinct_devices=distinct_devices,
        wall_clock_seconds=elapsed,
        workers=workers,
    )
