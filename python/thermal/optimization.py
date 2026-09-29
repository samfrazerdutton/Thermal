"""Optimization search: a safe framework for exploring a workload's parameter
space, where every candidate is a real controlled experiment against the
current baseline (Phase 8) -- never a single noisy sample compared against
another single noisy sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.comparison import Verdict
from thermal.experiment import ExperimentSpec, run_experiment
from thermal.workload import WorkloadRegistry


def _resolve_params(workload_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Merge params onto the workload's declared defaults so comparisons
    against 'the baseline' use its actual resolved configuration, not
    whatever subset of keys the caller happened to pass explicitly."""
    defaults = WorkloadRegistry.get(workload_name).spec.default_parameters
    return {**defaults, **params}


@dataclass(frozen=True)
class OptimizationCandidate:
    params: dict[str, Any]
    metric_value: float
    verdict: Verdict
    percent_change_from_baseline: float


@dataclass
class OptimizationResult:
    baseline_params: dict[str, Any]
    baseline_metric_value: float
    candidates: list[OptimizationCandidate] = field(default_factory=list)
    best: OptimizationCandidate | None = None

    def summary_text(self) -> str:
        lines = [f"Baseline: {self.baseline_metric_value:.4g} at {self.baseline_params}"]
        for c in self.candidates:
            lines.append(f"  {c.params}: {c.metric_value:.4g} ({c.percent_change_from_baseline:+.1f}%, {c.verdict.value})")
        if self.best is not None:
            lines.append(
                f"\nBest verified improvement: {self.best.params} "
                f"({self.best.percent_change_from_baseline:+.1f}%, {self.best.metric_value:.4g})"
            )
        else:
            lines.append("\nNo candidate produced a statistically verified improvement over baseline.")
        return "\n".join(lines)


def _evaluate_candidate(
    workload_name: str,
    baseline_params: dict[str, Any],
    candidate_params: dict[str, Any],
    metric_name: str,
    higher_is_better: bool,
    repetitions: int,
    warmup_iterations: int,
) -> OptimizationCandidate:
    spec = ExperimentSpec(
        workload_name=workload_name,
        baseline_params=baseline_params,
        treatment_params=candidate_params,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        repetitions=repetitions,
        warmup_iterations=warmup_iterations,
    )
    result = run_experiment(spec)
    comparison = result.comparison
    return OptimizationCandidate(
        params=candidate_params,
        metric_value=comparison.treatment_mean,
        verdict=comparison.verdict,
        percent_change_from_baseline=comparison.percent_change,
    )


def grid_search(
    workload_name: str,
    baseline_params: dict[str, Any],
    param_name: str,
    candidate_values: list[Any],
    metric_name: str,
    higher_is_better: bool = True,
    repetitions: int = 10,
    warmup_iterations: int = 2,
) -> OptimizationResult:
    """Try every value in candidate_values for one parameter, each as its own
    controlled experiment against the shared baseline. Picks the best verified
    (IMPROVED) candidate by measured effect, or reports that none beat baseline
    with statistical support -- never picks a candidate whose verdict was
    INCONCLUSIVE or REGRESSED just because it has the highest raw mean."""
    resolved_baseline = _resolve_params(workload_name, baseline_params)
    result = OptimizationResult(baseline_params=resolved_baseline, baseline_metric_value=0.0)

    baseline_only = _evaluate_candidate(
        workload_name, resolved_baseline, resolved_baseline, metric_name, higher_is_better, repetitions, warmup_iterations
    )
    result.baseline_metric_value = baseline_only.metric_value

    for value in candidate_values:
        candidate_params = {**resolved_baseline, param_name: value}
        if candidate_params == resolved_baseline:
            continue
        candidate = _evaluate_candidate(
            workload_name, resolved_baseline, candidate_params, metric_name, higher_is_better, repetitions, warmup_iterations
        )
        result.candidates.append(candidate)

    improved = [c for c in result.candidates if c.verdict == Verdict.IMPROVED]
    if improved:
        result.best = max(improved, key=lambda c: c.percent_change_from_baseline if higher_is_better else -c.percent_change_from_baseline)

    return result


def coordinate_descent(
    workload_name: str,
    baseline_params: dict[str, Any],
    param_candidates: dict[str, list[Any]],
    metric_name: str,
    higher_is_better: bool = True,
    repetitions: int = 10,
    warmup_iterations: int = 2,
) -> OptimizationResult:
    """Optimize one parameter at a time, keeping the best verified value found
    so far fixed while moving to the next parameter. Cheaper than a full grid
    search over multiple parameters, at the cost of not exploring interactions
    between them -- documented here rather than silently assumed away."""
    resolved_baseline = _resolve_params(workload_name, baseline_params)
    current_params = dict(resolved_baseline)
    result = OptimizationResult(baseline_params=dict(resolved_baseline), baseline_metric_value=0.0)

    baseline_eval = _evaluate_candidate(
        workload_name, resolved_baseline, resolved_baseline, metric_name, higher_is_better, repetitions, warmup_iterations
    )
    result.baseline_metric_value = baseline_eval.metric_value

    for param_name, candidate_values in param_candidates.items():
        best_this_round: OptimizationCandidate | None = None
        for value in candidate_values:
            candidate_params = {**current_params, param_name: value}
            if candidate_params == current_params:
                continue
            candidate = _evaluate_candidate(
                workload_name, current_params, candidate_params, metric_name, higher_is_better, repetitions, warmup_iterations
            )
            result.candidates.append(candidate)
            if candidate.verdict == Verdict.IMPROVED:
                if best_this_round is None or (
                    candidate.percent_change_from_baseline > best_this_round.percent_change_from_baseline
                    if higher_is_better
                    else candidate.percent_change_from_baseline < best_this_round.percent_change_from_baseline
                ):
                    best_this_round = candidate
        if best_this_round is not None:
            current_params = dict(best_this_round.params)

    if current_params != resolved_baseline:
        final = _evaluate_candidate(
            workload_name, resolved_baseline, current_params, metric_name, higher_is_better, repetitions, warmup_iterations
        )
        if final.verdict == Verdict.IMPROVED:
            result.best = final

    return result
