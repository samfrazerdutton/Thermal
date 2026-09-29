"""Unit tests for thermal.experiment (controlled A/B experiment engine)."""

import pytest

from analysis.comparison import Verdict
from thermal.experiment import ExperimentSpec, run_experiment
from thermal.workload import (
    HardwareRequirement,
    Workload,
    WorkloadSpec,
    register_workload,
)


class _ScaledWorkload(Workload):
    """A deterministic-ish test workload whose metric scales with a param,
    so experiments against it have a known, verifiable direction."""

    spec = WorkloadSpec(
        name="__test_scaled__",
        version="0.0.1",
        description="metric = scale * (1 + tiny noise)",
        hardware_requirements=HardwareRequirement(),
        default_parameters={"scale": 1.0},
        warmup_iterations=1,
        measurement_iterations=1,
    )

    def setup(self):
        import random

        self._rng = random.Random(42)
        self.scale = float(self.params["scale"])

    def run_once(self):
        noise = 1.0 + self._rng.uniform(-0.02, 0.02)
        return {"device": "cpu", "value": self.scale * noise}


register_workload(_ScaledWorkload)


def test_treatment_with_higher_scale_is_improved():
    spec = ExperimentSpec(
        workload_name="__test_scaled__",
        baseline_params={"scale": 100.0},
        treatment_params={"scale": 150.0},
        metric_name="value",
        higher_is_better=True,
        repetitions=20,
        warmup_iterations=2,
    )
    result = run_experiment(spec)
    assert result.comparison.verdict == Verdict.IMPROVED
    assert len(result.baseline_values) == 20
    assert len(result.treatment_values) == 20


def test_lower_is_better_flips_verdict_direction():
    spec = ExperimentSpec(
        workload_name="__test_scaled__",
        baseline_params={"scale": 100.0},
        treatment_params={"scale": 60.0},
        metric_name="value",
        higher_is_better=False,  # e.g. latency: lower treatment value is an improvement
        repetitions=20,
        warmup_iterations=2,
    )
    result = run_experiment(spec)
    assert result.comparison.verdict == Verdict.IMPROVED


def test_identical_configs_no_meaningful_change():
    spec = ExperimentSpec(
        workload_name="__test_scaled__",
        baseline_params={"scale": 100.0},
        treatment_params={"scale": 100.0},
        metric_name="value",
        repetitions=20,
        warmup_iterations=2,
    )
    result = run_experiment(spec)
    assert result.comparison.verdict in (Verdict.NO_CHANGE, Verdict.INCONCLUSIVE)


def test_missing_metric_raises_keyerror():
    spec = ExperimentSpec(
        workload_name="__test_scaled__",
        baseline_params={"scale": 100.0},
        treatment_params={"scale": 100.0},
        metric_name="does_not_exist",
        repetitions=6,
        warmup_iterations=1,
    )
    with pytest.raises(KeyError):
        run_experiment(spec)


def test_unknown_workload_raises_keyerror():
    spec = ExperimentSpec(
        workload_name="__not_registered__",
        baseline_params={},
        treatment_params={},
        metric_name="value",
        repetitions=6,
    )
    with pytest.raises(KeyError):
        run_experiment(spec)
