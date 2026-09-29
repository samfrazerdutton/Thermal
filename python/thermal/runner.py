"""Shared execution logic for running a workload or experiment and persisting
the result -- used by both the CLI (thermal/cli.py) and the FastAPI service
(services/api) so the two front ends can never drift into computing a
baseline, diagnosis, or comparison differently. Neither front end should
reimplement this logic; both call these functions.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from analysis.baseline import BaselineStats, InsufficientSamplesError, compute_baseline
from thermal.diagnosis import Diagnosis, classify, features_from_telemetry
from thermal.experiment import ExperimentRunResult, ExperimentSpec, run_experiment
from thermal.hardware import collect_hardware_report
from thermal.storage import (
    ExperimentRecord,
    ExperimentRepository,
    RunRecord,
    SQLiteRunRepository,
    default_db_path,
)
from thermal.telemetry import TelemetryCollector
from thermal.workload import WorkloadRegistry, run_workload


def git_commit() -> Optional[str]:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class WorkloadRunOutcome:
    record: RunRecord
    baselines: dict[str, BaselineStats]
    diagnosis: Diagnosis
    per_iteration_metrics: list[dict[str, Any]]


def run_and_store_workload(
    workload_name: str,
    params: Optional[dict[str, Any]] = None,
    samples: Optional[int] = None,
    warmup: Optional[int] = None,
) -> WorkloadRunOutcome:
    workload_cls = WorkloadRegistry.get(workload_name)
    instance = workload_cls(params)
    if samples is not None or warmup is not None:
        instance.spec = instance.spec.__class__(
            **{
                **instance.spec.__dict__,
                "warmup_iterations": warmup if warmup is not None else instance.spec.warmup_iterations,
                "measurement_iterations": samples if samples is not None else instance.spec.measurement_iterations,
            }
        )

    telemetry = TelemetryCollector(interval_seconds=0.2)
    telemetry.start()
    try:
        result = run_workload(instance)
    finally:
        telemetry.stop()

    numeric_keys = [
        k
        for k in result.per_iteration_metrics[0]
        if k != "device" and isinstance(result.per_iteration_metrics[0].get(k), (int, float))
    ]
    baselines: dict[str, BaselineStats] = {}
    for key in numeric_keys:
        values = [m[key] for m in result.per_iteration_metrics if isinstance(m.get(key), (int, float))]
        try:
            baselines[key] = compute_baseline(values, metric_name=key)
        except InsufficientSamplesError:
            continue

    features = features_from_telemetry(telemetry.samples)
    diagnosis = classify(features)

    telemetry_dir = default_db_path().parent / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    report = collect_hardware_report()
    record = RunRecord.new(
        workload_name=instance.spec.name,
        workload_version=instance.spec.version,
        device=result.device,
        warmup_iterations=result.warmup_iterations,
        measurement_iterations=result.measurement_iterations,
        configuration=instance.params,
        hardware_fingerprint={
            "cpu_name": report.cpu.name,
            "gpu_name": report.gpu.name,
            "gpu_driver_version": report.gpu.driver_version,
            "cuda_driver_version": report.gpu.cuda_driver_version,
            "os": report.software.os_name,
        },
        metrics={k: asdict(v) for k, v in baselines.items()},
        git_commit=git_commit(),
        diagnosis={
            "bottleneck": diagnosis.bottleneck.value,
            "confidence": diagnosis.confidence,
            "rationale": diagnosis.rationale,
            "evidence": [asdict(e) for e in diagnosis.evidence],
            "missing_features": diagnosis.missing_features,
        },
    )
    telemetry_path = telemetry_dir / f"{record.run_id}.jsonl"
    telemetry.write_jsonl(telemetry_path)
    record.telemetry_path = str(telemetry_path)

    SQLiteRunRepository(default_db_path()).save(record)

    return WorkloadRunOutcome(
        record=record,
        baselines=baselines,
        diagnosis=diagnosis,
        per_iteration_metrics=result.per_iteration_metrics,
    )


@dataclass
class ExperimentRunOutcome:
    record: ExperimentRecord
    result: ExperimentRunResult


def run_and_store_experiment(
    workload_name: str,
    baseline_params: dict[str, Any],
    treatment_params: dict[str, Any],
    metric_name: str,
    higher_is_better: bool = True,
    repetitions: int = 15,
    warmup_iterations: int = 3,
    hypothesis: str = "",
) -> ExperimentRunOutcome:
    spec = ExperimentSpec(
        workload_name=workload_name,
        baseline_params=baseline_params,
        treatment_params=treatment_params,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        repetitions=repetitions,
        warmup_iterations=warmup_iterations,
        hypothesis=hypothesis,
    )
    result = run_experiment(spec)
    comparison = result.comparison

    record = ExperimentRecord.new(
        workload_name=workload_name,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        repetitions=repetitions,
        baseline_config=baseline_params,
        treatment_config=treatment_params,
        baseline_values=result.baseline_values,
        treatment_values=result.treatment_values,
        baseline_device=result.baseline_device,
        treatment_device=result.treatment_device,
        comparison=asdict(comparison),
        verdict=comparison.verdict.value,
        hypothesis=hypothesis,
        git_commit=git_commit(),
    )
    ExperimentRepository(default_db_path()).save(record)

    return ExperimentRunOutcome(record=record, result=result)
