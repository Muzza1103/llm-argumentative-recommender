from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph_builder import ArgumentGraph, ArgumentNode


@dataclass
class DFQuADResult:
    """
    Result of DF-QuAD evaluation on the root node.
    """
    root_id: str
    root_text: str
    root_base_score: float
    support_scores: list[float]
    attack_scores: list[float]
    aggregated_support: float
    aggregated_attack: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "root_text": self.root_text,
            "root_base_score": self.root_base_score,
            "support_scores": self.support_scores,
            "attack_scores": self.attack_scores,
            "aggregated_support": self.aggregated_support,
            "aggregated_attack": self.aggregated_attack,
            "final_score": self.final_score,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def aggregate_strength(strengths: list[float]) -> float:
    """
    Aggregate multiple strengths into one value in [0, 1].
    Use the probabilistic sum:
        1 - Π(1 - s_i)
    """
    if not strengths:
        return 0.0

    product = 1.0
    for s in strengths:
        product *= (1.0 - _clamp(s))

    return 1.0 - product


def dfquad_combine(
    base_score: float,
    attack_strength: float,
    support_strength: float,
) -> float:
    """
    Simple DF-QuAD-style combination

    Formula used :
        if support_strength >= attack_strength:
            result = base + (1 - base) * (support - attack)
        else:
            result = base - base * (attack - support)

    stronger supports -> move upward toward 1
    stronger attacks -> move downward toward 0
    """
    base_score = _clamp(base_score)
    attack_strength = _clamp(attack_strength)
    support_strength = _clamp(support_strength)

    if support_strength >= attack_strength:
        delta = support_strength - attack_strength
        return _clamp(base_score + (1.0 - base_score) * delta)

    delta = attack_strength - support_strength
    return _clamp(base_score - base_score * delta)


def evaluate_root_dfquad(graph: ArgumentGraph) -> DFQuADResult:
    """
    Evaluate only the root claim in the graph for the first version.
    """
    root = graph.get_root()

    supporters = graph.get_supporters_of(root.node_id)
    attackers = graph.get_attackers_of(root.node_id)

    support_scores = [node.base_score for node in supporters]
    attack_scores = [node.base_score for node in attackers]

    aggregated_support = aggregate_strength(support_scores)
    aggregated_attack = aggregate_strength(attack_scores)

    final_score = dfquad_combine(
        base_score=root.base_score,
        attack_strength=aggregated_attack,
        support_strength=aggregated_support,
    )

    return DFQuADResult(
        root_id=root.node_id,
        root_text=root.text,
        root_base_score=root.base_score,
        support_scores=support_scores,
        attack_scores=attack_scores,
        aggregated_support=aggregated_support,
        aggregated_attack=aggregated_attack,
        final_score=final_score,
    )