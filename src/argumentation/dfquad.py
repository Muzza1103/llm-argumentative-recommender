from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .graph_builder import ArgumentGraph


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
    aggregation_method: str
    calibration_method: str
    calibration_beta: float

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
            "aggregation_method": self.aggregation_method,
            "calibration_method": self.calibration_method,
            "calibration_beta": self.calibration_beta,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def aggregate_strength(
    strengths: list[float],
    method: str = "dfquad",
) -> float:
    if not strengths:
        return 0.0

    strengths = [_clamp(s) for s in strengths]

    if method == "dfquad":
        product = 1.0
        for s in strengths:
            product *= 1.0 - s
        return 1.0 - product

    if method == "mean":
        return sum(strengths) / len(strengths)

    if method == "max":
        return max(strengths)

    raise ValueError(f"Unknown aggregation method: {method}")


def dfquad_combine(
    base_score: float,
    attack_strength: float,
    support_strength: float,
) -> float:
    base_score = _clamp(base_score)
    attack_strength = _clamp(attack_strength)
    support_strength = _clamp(support_strength)

    if support_strength >= attack_strength:
        delta = support_strength - attack_strength
        return _clamp(base_score + (1.0 - base_score) * delta)

    delta = attack_strength - support_strength
    return _clamp(base_score - base_score * delta)


def calibrate_score(
    score: float,
    method: str = "none",
    beta: float = 12.0,
) -> float:
    score = _clamp(score)

    if method == "none":
        return score

    if method == "centered_sigmoid":
        return _clamp(1.0 / (1.0 + math.exp(-beta * (score - 0.5))))

    raise ValueError(f"Unknown calibration method: {method}")


def evaluate_root_dfquad(
    graph: ArgumentGraph,
    aggregation_method: str = "dfquad",
    calibration_method: str = "none",
    calibration_beta: float = 12.0,
) -> DFQuADResult:
    root = graph.get_root()

    supporters = graph.get_supporters_of(root.node_id)
    attackers = graph.get_attackers_of(root.node_id)

    support_scores = [node.base_score for node in supporters]
    attack_scores = [node.base_score for node in attackers]

    aggregated_support = aggregate_strength(
        support_scores,
        method=aggregation_method,
    )
    aggregated_attack = aggregate_strength(
        attack_scores,
        method=aggregation_method,
    )

    final_score = dfquad_combine(
        base_score=root.base_score,
        attack_strength=aggregated_attack,
        support_strength=aggregated_support,
    )

    final_score = calibrate_score(
        final_score,
        method=calibration_method,
        beta=calibration_beta,
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
        aggregation_method=aggregation_method,
        calibration_method=calibration_method,
        calibration_beta=calibration_beta,
    )