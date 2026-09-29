"""Unit tests for thermal.causal."""

import pytest

from thermal.causal import (
    CausalEdge,
    CausalGraph,
    EvidenceKind,
    Relationship,
    build_graph_from_experiments,
    edge_from_experiment,
)
from thermal.storage import ExperimentRecord


def _make_experiment(**overrides):
    defaults = dict(
        workload_name="matmul",
        metric_name="gflops",
        higher_is_better=True,
        repetitions=15,
        baseline_config={"size": 256, "dtype": "float32"},
        treatment_config={"size": 512, "dtype": "float32"},
        baseline_values=[100.0] * 15,
        treatment_values=[160.0] * 15,
        baseline_device="cpu",
        treatment_device="cpu",
        comparison={
            "percent_change": 60.0,
            "percent_change_ci_low": 40.0,
            "percent_change_ci_high": 80.0,
            "confidence_level": 0.95,
        },
        verdict="IMPROVED",
    )
    defaults.update(overrides)
    return ExperimentRecord.new(**defaults)


def test_edge_from_experiment_refuses_when_ci_does_not_exclude_zero():
    with pytest.raises(ValueError):
        edge_from_experiment("size", "gflops", "exp-1", 10.0, 0.95, ci_excludes_zero=False)


def test_edge_from_experiment_positive_relationship():
    edge = edge_from_experiment("size", "gflops", "exp-1", 60.0, 0.95, ci_excludes_zero=True)
    assert edge.relationship == Relationship.POSITIVE
    assert edge.evidence_kind == EvidenceKind.EXPERIMENTAL_EVIDENCE


def test_edge_from_experiment_negative_relationship():
    edge = edge_from_experiment("size", "duration_seconds", "exp-1", -20.0, 0.95, ci_excludes_zero=True)
    assert edge.relationship == Relationship.NEGATIVE


def test_build_graph_skips_inconclusive_experiments():
    experiments = [_make_experiment(verdict="INCONCLUSIVE")]
    graph = build_graph_from_experiments(experiments)
    assert graph.edges() == []


def test_build_graph_skips_experiments_without_ci_excluding_zero():
    experiments = [
        _make_experiment(
            verdict="IMPROVED",
            comparison={"percent_change": 5.0, "percent_change_ci_low": -1.0, "percent_change_ci_high": 10.0, "confidence_level": 0.95},
        )
    ]
    graph = build_graph_from_experiments(experiments)
    assert graph.edges() == []


def test_build_graph_creates_edge_for_changed_param():
    experiments = [_make_experiment()]
    graph = build_graph_from_experiments(experiments)
    edges = graph.edges()
    assert len(edges) == 1
    assert edges[0].source == "size"
    assert edges[0].target == "gflops"


def test_build_graph_ignores_unchanged_params():
    experiments = [_make_experiment()]  # dtype unchanged between baseline/treatment
    graph = build_graph_from_experiments(experiments)
    sources = {e.source for e in graph.edges()}
    assert "dtype" not in sources


def test_graph_path_finds_multi_hop_chain():
    graph = CausalGraph()
    graph.add_edge(CausalEdge("batch_size", "kv_cache_size", Relationship.POSITIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))
    graph.add_edge(CausalEdge("kv_cache_size", "hbm_bandwidth", Relationship.POSITIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))
    graph.add_edge(CausalEdge("hbm_bandwidth", "throughput", Relationship.NEGATIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))

    path = graph.path("batch_size", "throughput")
    assert path is not None
    assert [e.target for e in path] == ["kv_cache_size", "hbm_bandwidth", "throughput"]


def test_graph_path_returns_none_when_unreachable():
    graph = CausalGraph()
    graph.add_edge(CausalEdge("a", "b", Relationship.POSITIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))
    assert graph.path("a", "z") is None


def test_graph_path_same_node_returns_empty_list():
    graph = CausalGraph()
    assert graph.path("a", "a") == []


def test_graph_nodes_collects_all_sources_and_targets():
    graph = CausalGraph()
    graph.add_edge(CausalEdge("a", "b", Relationship.POSITIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))
    graph.add_edge(CausalEdge("b", "c", Relationship.POSITIVE, EvidenceKind.EXPERIMENTAL_EVIDENCE))
    assert graph.nodes() == {"a", "b", "c"}
