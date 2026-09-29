"""Causal performance graph.

A graph of parameter/metric relationships, where every edge is tagged with
the kind of evidence backing it -- correlation alone is never labeled a
causal relationship. Edges typically come from experiment results (Phase 8):
an experiment that varied one parameter and measured a statistically
supported change in a metric is EXPERIMENTAL_EVIDENCE; a relationship
inferred from domain knowledge without a matching experiment is
INFERRED_RELATIONSHIP; a relationship read off telemetry without any
controlled variation is OBSERVED_CORRELATION -- the weakest of the three and
never rendered as "X causes Y" without that qualifier attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvidenceKind(str, Enum):
    OBSERVED_CORRELATION = "OBSERVED_CORRELATION"
    EXPERIMENTAL_EVIDENCE = "EXPERIMENTAL_EVIDENCE"
    INFERRED_RELATIONSHIP = "INFERRED_RELATIONSHIP"


class Relationship(str, Enum):
    POSITIVE = "positive"  # increasing source tends to increase target
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    relationship: Relationship
    evidence_kind: EvidenceKind
    experiment_id: Optional[str] = None
    percent_change: Optional[float] = None
    confidence_level: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship.value,
            "evidence": {
                "kind": self.evidence_kind.value,
                "experiment_id": self.experiment_id,
                "percent_change": self.percent_change,
                "confidence_level": self.confidence_level,
                "note": self.note,
            },
        }


class CausalGraph:
    """An append-only, evidence-tagged graph. Nodes are implicit (any string
    used as a source/target); duplicate edges between the same pair are kept
    as separate entries so the evidence history isn't lost on re-assertion."""

    def __init__(self) -> None:
        self._edges: list[CausalEdge] = []

    def add_edge(self, edge: CausalEdge) -> None:
        self._edges.append(edge)

    def edges(self) -> list[CausalEdge]:
        return list(self._edges)

    def edges_from(self, source: str) -> list[CausalEdge]:
        return [e for e in self._edges if e.source == source]

    def edges_to(self, target: str) -> list[CausalEdge]:
        return [e for e in self._edges if e.target == target]

    def nodes(self) -> set[str]:
        return {e.source for e in self._edges} | {e.target for e in self._edges}

    def path(self, source: str, target: str) -> Optional[list[CausalEdge]]:
        """Breadth-first search for a chain of edges from source to target,
        following only forward (source -> target) edges."""
        if source == target:
            return []
        visited = {source}
        queue: list[tuple[str, list[CausalEdge]]] = [(source, [])]
        while queue:
            node, path_so_far = queue.pop(0)
            for edge in self.edges_from(node):
                if edge.target in visited:
                    continue
                new_path = path_so_far + [edge]
                if edge.target == target:
                    return new_path
                visited.add(edge.target)
                queue.append((edge.target, new_path))
        return None

    def to_dict(self) -> dict:
        return {"edges": [e.to_dict() for e in self._edges]}


def _ci_excludes_zero(comparison: dict) -> bool:
    low, high = comparison.get("percent_change_ci_low"), comparison.get("percent_change_ci_high")
    if low is None or high is None:
        return False
    return (low > 0) == (high > 0) and low != 0 and high != 0


def build_graph_from_experiments(experiments: list) -> CausalGraph:
    """Derive a causal graph from stored experiments (thermal.storage.ExperimentRecord).
    Only experiments with a verdict of IMPROVED or REGRESSED contribute edges --
    an INCONCLUSIVE or NO_CHANGE experiment asserted no relationship, so it adds none."""
    graph = CausalGraph()
    for exp in experiments:
        if exp.verdict not in ("IMPROVED", "REGRESSED"):
            continue
        if not _ci_excludes_zero(exp.comparison):
            continue
        changed_params = {
            k for k in {**exp.baseline_config, **exp.treatment_config}
            if exp.baseline_config.get(k) != exp.treatment_config.get(k)
        }
        for param in changed_params:
            edge = edge_from_experiment(
                independent_variable=param,
                dependent_variable=exp.metric_name,
                experiment_id=exp.experiment_id,
                percent_change=exp.comparison["percent_change"],
                confidence_level=exp.comparison["confidence_level"],
                ci_excludes_zero=True,
            )
            graph.add_edge(edge)
    return graph


def edge_from_experiment(
    independent_variable: str,
    dependent_variable: str,
    experiment_id: str,
    percent_change: float,
    confidence_level: float,
    ci_excludes_zero: bool,
) -> CausalEdge:
    """Build an edge from a completed experiment. Only labeled
    EXPERIMENTAL_EVIDENCE when the CI actually excludes zero -- an experiment
    with an inconclusive result asserts no causal edge at all (the caller
    should not call this for an INCONCLUSIVE verdict)."""
    if not ci_excludes_zero:
        raise ValueError(
            "refusing to record a causal edge from an experiment whose confidence "
            "interval does not exclude zero -- that is not evidence of a relationship"
        )
    relationship = Relationship.POSITIVE if percent_change > 0 else Relationship.NEGATIVE
    return CausalEdge(
        source=independent_variable,
        target=dependent_variable,
        relationship=relationship,
        evidence_kind=EvidenceKind.EXPERIMENTAL_EVIDENCE,
        experiment_id=experiment_id,
        percent_change=percent_change,
        confidence_level=confidence_level,
        note=f"experiment {experiment_id}: {percent_change:+.1f}% change, CI excludes zero",
    )
