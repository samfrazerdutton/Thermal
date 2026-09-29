"""Unit tests for thermal.optimization (grid search / coordinate descent)."""

import random

from analysis.comparison import Verdict
from thermal.optimization import coordinate_descent, grid_search
from thermal.workload import HardwareRequirement, Workload, WorkloadSpec, register_workload


class _TunableWorkload(Workload):
    """metric = base_value(x) * base_value(y), so the optimum for (x, y) is
    known ahead of time and we can check the search actually finds it."""

    spec = WorkloadSpec(
        name="__test_tunable__",
        version="0.0.1",
        description="synthetic two-parameter workload with a known optimum",
        hardware_requirements=HardwareRequirement(),
        default_parameters={"x": 1, "y": 1},
        warmup_iterations=1,
        measurement_iterations=1,
    )

    _X_SCORES = {1: 10.0, 2: 20.0, 4: 5.0}
    _Y_SCORES = {1: 1.0, 2: 3.0, 3: 0.5}

    def setup(self):
        self._rng = random.Random(7)

    def run_once(self):
        noise = 1.0 + self._rng.uniform(-0.01, 0.01)
        value = self._X_SCORES[self.params["x"]] * self._Y_SCORES[self.params["y"]] * noise
        return {"device": "cpu", "score": value}


register_workload(_TunableWorkload)


def test_grid_search_finds_best_verified_candidate():
    result = grid_search(
        workload_name="__test_tunable__",
        baseline_params={"x": 1, "y": 1},
        param_name="x",
        candidate_values=[2, 4],
        metric_name="score",
        higher_is_better=True,
        repetitions=15,
        warmup_iterations=2,
    )
    assert result.best is not None
    assert result.best.params["x"] == 2  # x=2 gives the highest score (20 vs 10 vs 5)


def test_grid_search_reports_no_best_when_all_candidates_worse():
    result = grid_search(
        workload_name="__test_tunable__",
        baseline_params={"x": 2, "y": 1},
        param_name="x",
        candidate_values=[4],
        metric_name="score",
        higher_is_better=True,
        repetitions=15,
        warmup_iterations=2,
    )
    assert result.best is None
    assert all(c.verdict != Verdict.IMPROVED for c in result.candidates)


def test_grid_search_skips_candidate_equal_to_resolved_baseline():
    result = grid_search(
        workload_name="__test_tunable__",
        baseline_params={},  # resolves to x=1, y=1 via workload defaults
        param_name="x",
        candidate_values=[1, 2],  # x=1 equals the resolved baseline -> must be skipped
        metric_name="score",
        higher_is_better=True,
        repetitions=15,
        warmup_iterations=2,
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].params["x"] == 2
    assert result.baseline_params == {"x": 1, "y": 1}


def test_coordinate_descent_finds_joint_optimum():
    result = coordinate_descent(
        workload_name="__test_tunable__",
        baseline_params={"x": 1, "y": 1},
        param_candidates={"x": [2, 4], "y": [2, 3]},
        metric_name="score",
        higher_is_better=True,
        repetitions=15,
        warmup_iterations=2,
    )
    assert result.best is not None
    assert result.best.params["x"] == 2
    assert result.best.params["y"] == 2  # x=2,y=2 gives 60, the joint optimum


def test_optimization_result_summary_text_mentions_baseline():
    result = grid_search(
        workload_name="__test_tunable__",
        baseline_params={"x": 1, "y": 1},
        param_name="x",
        candidate_values=[2],
        metric_name="score",
        repetitions=15,
        warmup_iterations=2,
    )
    text = result.summary_text()
    assert "Baseline" in text
