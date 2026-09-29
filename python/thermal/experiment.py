"""Controlled experiment engine: baseline vs. treatment, run alternating
(ABAB) to reduce temporal bias (thermal drift, background load creeping in
over a long run), then handed to analysis.comparison for verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.comparison import ComparisonResult, compare_samples
from thermal.workload import Workload, WorkloadRegistry


@dataclass(frozen=True)
class ExperimentSpec:
    workload_name: str
    baseline_params: dict[str, Any]
    treatment_params: dict[str, Any]
    metric_name: str
    higher_is_better: bool = True
    repetitions: int = 15
    warmup_iterations: int = 3
    hypothesis: str = ""


@dataclass
class ExperimentRunResult:
    spec: ExperimentSpec
    baseline_values: list[float] = field(default_factory=list)
    treatment_values: list[float] = field(default_factory=list)
    baseline_device: str = "unknown"
    treatment_device: str = "unknown"
    comparison: ComparisonResult | None = None


def _instantiate(workload_name: str, params: dict[str, Any]) -> Workload:
    workload_cls = WorkloadRegistry.get(workload_name)
    return workload_cls(params)


def run_experiment(spec: ExperimentSpec) -> ExperimentRunResult:
    baseline = _instantiate(spec.workload_name, spec.baseline_params)
    treatment = _instantiate(spec.workload_name, spec.treatment_params)

    baseline.setup()
    treatment.setup()
    try:
        # Randomization policy: strict alternation (ABAB...), not two separate
        # blocks -- this is what actually protects against a slow thermal
        # ramp or a background process appearing partway through the run
        # from silently favoring whichever arm happened to run first/second.
        for _ in range(spec.warmup_iterations):
            baseline.run_once()
            treatment.run_once()

        result = ExperimentRunResult(spec=spec)
        for _ in range(spec.repetitions):
            b = baseline.run_once()
            t = treatment.run_once()
            if spec.metric_name not in b or spec.metric_name not in t:
                raise KeyError(
                    f"metric '{spec.metric_name}' not produced by workload '{spec.workload_name}' "
                    f"(available: {sorted(b.keys())})"
                )
            result.baseline_values.append(b[spec.metric_name])
            result.treatment_values.append(t[spec.metric_name])
            result.baseline_device = b.get("device", "unknown")
            result.treatment_device = t.get("device", "unknown")
    finally:
        baseline.teardown()
        treatment.teardown()

    result.comparison = compare_samples(
        result.baseline_values,
        result.treatment_values,
        metric_name=spec.metric_name,
        higher_is_better=spec.higher_is_better,
        min_samples=min(5, spec.repetitions),
    )
    return result
