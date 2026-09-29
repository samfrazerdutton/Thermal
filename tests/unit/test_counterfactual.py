"""Unit tests for thermal.counterfactual."""

from thermal.counterfactual import (
    experiment_command_for,
    generate_hypotheses,
)
from thermal.diagnosis import BottleneckClass


def test_memory_bound_proposes_precision_reduction_when_dtype_known():
    hypotheses = generate_hypotheses(
        BottleneckClass.MEMORY_BOUND,
        current_params={"size": 1024, "dtype": "float32"},
        known_param_names={"size", "dtype"},
    )
    actionable = [h for h in hypotheses if h.actionable]
    assert any(h.intervention_type == "reduce_precision" for h in actionable)
    precision_hyp = next(h for h in actionable if h.intervention_type == "reduce_precision")
    assert precision_hyp.proposed_treatment_params == {"dtype": "float16"}


def test_memory_bound_precision_not_actionable_without_dtype_param():
    hypotheses = generate_hypotheses(
        BottleneckClass.MEMORY_BOUND,
        current_params={"size_mb": 64},
        known_param_names={"size_mb"},
    )
    precision_hyp = next(h for h in hypotheses if h.intervention_type == "reduce_precision")
    assert precision_hyp.actionable is False
    assert "dtype" in precision_hyp.reason_not_actionable or "parameter" in precision_hyp.reason_not_actionable


def test_launch_overhead_proposes_larger_batch():
    hypotheses = generate_hypotheses(
        BottleneckClass.LAUNCH_OVERHEAD,
        current_params={"batch_size": 8},
        known_param_names={"batch_size"},
    )
    actionable = [h for h in hypotheses if h.actionable]
    assert len(actionable) == 1
    assert actionable[0].proposed_treatment_params == {"batch_size": 16}


def test_unknown_bottleneck_class_returns_non_actionable_placeholder():
    hypotheses = generate_hypotheses(
        BottleneckClass.UNKNOWN,
        current_params={},
        known_param_names=set(),
    )
    assert len(hypotheses) == 1
    assert hypotheses[0].actionable is False


def test_precision_change_stops_at_lowest_supported_dtype():
    hypotheses = generate_hypotheses(
        BottleneckClass.COMPUTE_BOUND,
        current_params={"dtype": "bfloat16"},
        known_param_names={"dtype"},
    )
    hyp = hypotheses[0]
    assert hyp.actionable is False


def test_experiment_command_for_non_actionable_returns_none():
    hypotheses = generate_hypotheses(
        BottleneckClass.MEMORY_BOUND,
        current_params={"size_mb": 64},
        known_param_names={"size_mb"},
    )
    precision_hyp = next(h for h in hypotheses if h.intervention_type == "reduce_precision")
    assert experiment_command_for(precision_hyp, "memory_bandwidth", {"size_mb": 64}, "bandwidth_gbps") is None


def test_experiment_command_for_actionable_includes_both_arms():
    hypotheses = generate_hypotheses(
        BottleneckClass.MEMORY_BOUND,
        current_params={"size": 1024, "dtype": "float32"},
        known_param_names={"size", "dtype"},
    )
    hyp = next(h for h in hypotheses if h.intervention_type == "reduce_precision")
    cmd = experiment_command_for(hyp, "matmul", {"size": 1024, "dtype": "float32"}, "gflops")
    assert "thermal experiment run matmul" in cmd
    assert "--baseline-param dtype=float32" in cmd
    assert "--treatment-param dtype=float16" in cmd
    assert "--metric gflops" in cmd


def test_experiment_command_for_lower_is_better_adds_flag():
    hypotheses = generate_hypotheses(
        BottleneckClass.LATENCY_BOUND,
        current_params={"size": 100},
        known_param_names={"size"},
    )
    hyp = hypotheses[0]
    cmd = experiment_command_for(hyp, "matmul", {"size": 100}, "duration_seconds", higher_is_better=False)
    assert "--lower-is-better" in cmd
