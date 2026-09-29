"""Counterfactual engine: turns a bottleneck classification into concrete,
testable hypotheses -- not just prose, but a proposed treatment configuration
that can be handed straight to thermal/experiment.py. A hypothesis without a
runnable experiment attached is not a hypothesis THERMAL considers actionable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from thermal.diagnosis import BottleneckClass


@dataclass(frozen=True)
class Hypothesis:
    intervention_type: str
    description: str
    independent_variable: str
    dependent_variable: str
    expected_direction: str  # "increase" or "decrease" in the dependent variable
    controls: list[str] = field(default_factory=list)
    proposed_treatment_params: Optional[dict[str, Any]] = None
    actionable: bool = False
    reason_not_actionable: Optional[str] = None


# bottleneck -> ordered list of (intervention_type, description template, param strategy).
# Each strategy is a function(current_params, known_param_names) -> Optional[dict] proposing
# a concrete treatment, or None if this workload doesn't expose a relevant knob.

def _propose_precision_change(current_params: dict, known: set[str]) -> Optional[dict]:
    if "dtype" not in known:
        return None
    current = current_params.get("dtype", "float32")
    order = ["float64", "float32", "float16", "bfloat16"]
    if current not in order:
        return None
    idx = order.index(current)
    if idx + 1 >= len(order):
        return None
    return {"dtype": order[idx + 1]}


def _propose_larger_batch(current_params: dict, known: set[str]) -> Optional[dict]:
    for key in ("size", "size_mb", "size_millions", "batch_size"):
        if key in known and key in current_params:
            try:
                current = current_params[key]
                return {key: current * 2}
            except TypeError:
                continue
    return None


def _propose_smaller_batch(current_params: dict, known: set[str]) -> Optional[dict]:
    for key in ("size", "size_mb", "size_millions", "batch_size"):
        if key in known and key in current_params:
            try:
                current = current_params[key]
                candidate = current / 2
                return {key: type(current)(candidate) if isinstance(current, int) else candidate}
            except TypeError:
                continue
    return None


_ProposeFn = Callable[[dict, set], Optional[dict]]
_STRATEGIES: dict[BottleneckClass, list[tuple[str, str, str, str, _ProposeFn]]] = {
    BottleneckClass.MEMORY_BOUND: [
        (
            "reduce_precision",
            "Reducing numeric precision cuts bytes moved per element, which should relieve HBM bandwidth pressure.",
            "dtype", "throughput", _propose_precision_change,
        ),
        (
            "reduce_problem_size",
            "A smaller working set may fit better in cache, reducing traffic to HBM.",
            "size", "throughput", _propose_smaller_batch,
        ),
    ],
    BottleneckClass.COMPUTE_BOUND: [
        (
            "reduce_precision",
            "Lower-precision arithmetic (e.g. fp16 tensor cores) can increase achieved FLOP/s when compute, not memory, is the ceiling.",
            "dtype", "throughput", _propose_precision_change,
        ),
    ],
    BottleneckClass.CAPACITY_BOUND: [
        (
            "reduce_problem_size",
            "Device memory is nearly full; a smaller working set relieves allocation pressure.",
            "size", "throughput", _propose_smaller_batch,
        ),
    ],
    BottleneckClass.LAUNCH_OVERHEAD: [
        (
            "increase_batch_size",
            "Fewer, larger kernel launches amortize per-launch overhead.",
            "batch_size", "throughput", _propose_larger_batch,
        ),
    ],
    BottleneckClass.LATENCY_BOUND: [
        (
            "increase_batch_size",
            "A larger batch/problem size amortizes fixed per-call overhead over more work.",
            "size", "throughput", _propose_larger_batch,
        ),
    ],
    BottleneckClass.CPU_BOUND: [
        (
            "increase_batch_size",
            "If host-side per-call overhead dominates, doing more work per call reduces its relative share.",
            "size", "throughput", _propose_larger_batch,
        ),
    ],
    BottleneckClass.TRANSFER_BOUND: [
        (
            "reduce_precision",
            "Smaller transferred payloads (lower precision) reduce time spent on host<->device copies.",
            "dtype", "throughput", _propose_precision_change,
        ),
    ],
}


def generate_hypotheses(
    bottleneck: BottleneckClass,
    current_params: dict[str, Any],
    known_param_names: set[str],
) -> list[Hypothesis]:
    strategies = _STRATEGIES.get(bottleneck, [])
    hypotheses: list[Hypothesis] = []

    for intervention_type, description, independent_var, dependent_var, propose in strategies:
        proposed = propose(current_params, known_param_names)
        if proposed is not None:
            hypotheses.append(
                Hypothesis(
                    intervention_type=intervention_type,
                    description=description,
                    independent_variable=independent_var,
                    dependent_variable=dependent_var,
                    expected_direction="increase",
                    controls=[k for k in current_params if k not in proposed],
                    proposed_treatment_params=proposed,
                    actionable=True,
                )
            )
        else:
            hypotheses.append(
                Hypothesis(
                    intervention_type=intervention_type,
                    description=description,
                    independent_variable=independent_var,
                    dependent_variable=dependent_var,
                    expected_direction="increase",
                    actionable=False,
                    reason_not_actionable=(
                        f"this workload has no '{independent_var}'-like parameter to vary"
                    ),
                )
            )

    if not hypotheses:
        hypotheses.append(
            Hypothesis(
                intervention_type="none",
                description=f"No first-generation intervention strategy exists yet for {bottleneck.value}.",
                independent_variable="",
                dependent_variable="",
                expected_direction="",
                actionable=False,
                reason_not_actionable="no strategy registered for this bottleneck class (see docs/roadmap.md Phase 9)",
            )
        )

    return hypotheses


def experiment_command_for(
    hypothesis: Hypothesis,
    workload_name: str,
    baseline_params: dict[str, Any],
    metric_name: str,
    higher_is_better: bool = True,
) -> Optional[str]:
    """Render the exact `thermal experiment run` invocation that would test this
    hypothesis, or None if it isn't actionable."""
    if not hypothesis.actionable or hypothesis.proposed_treatment_params is None:
        return None

    baseline_flags = " ".join(f"--baseline-param {k}={v}" for k, v in baseline_params.items())
    treatment_params = {**baseline_params, **hypothesis.proposed_treatment_params}
    treatment_flags = " ".join(f"--treatment-param {k}={v}" for k, v in treatment_params.items())
    direction_flag = "" if higher_is_better else " --lower-is-better"

    return (
        f"thermal experiment run {workload_name} {baseline_flags} {treatment_flags} "
        f"--metric {metric_name}{direction_flag} --hypothesis \"{hypothesis.description}\""
    )
