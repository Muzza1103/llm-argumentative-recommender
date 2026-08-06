from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import Argument


@dataclass
class ArgumentNode:
    """
    One node in the argumentative graph, can be a generated argument or the recommandation claim
    """
    node_id: str
    node_type: str  # "argument" or "root"
    text: str
    base_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArgumentEdge:
    """
    Directed relation between two nodes, can be "attack' or "support"
    """
    source_id: str
    target_id: str
    relation_type: str


@dataclass
class ArgumentGraph:
    root_id: str
    nodes: dict[str, ArgumentNode]
    edges: list[ArgumentEdge]

    def get_root(self) -> ArgumentNode:
        return self.nodes[self.root_id]

    def get_argument_nodes(self) -> list[ArgumentNode]:
        return [
            node
            for node in self.nodes.values()
            if node.node_type == "argument"
        ]

    def get_supporters_of(self, target_id: str) -> list[ArgumentNode]:
        supporter_ids = [
            edge.source_id
            for edge in self.edges
            if edge.target_id == target_id and edge.relation_type == "support"
        ]
        return [self.nodes[node_id] for node_id in supporter_ids]

    def get_attackers_of(self, target_id: str) -> list[ArgumentNode]:
        attacker_ids = [
            edge.source_id
            for edge in self.edges
            if edge.target_id == target_id and edge.relation_type == "attack"
        ]
        return [self.nodes[node_id] for node_id in attacker_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "text": node.text,
                    "base_score": node.base_score,
                    "metadata": node.metadata,
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                }
                for edge in self.edges
            ],
        }


def get_argument_base_score(argument: Argument) -> float:
    """
    Choose the best available intrinsic score for one argument.
    """
    if argument.intrinsic_strength is not None:
        return float(argument.intrinsic_strength)

    if argument.combined_score is not None:
        return float(argument.combined_score)

    if argument.llm_score is not None:
        return float(argument.llm_score)

    if argument.mf_score is not None:
        return float(argument.mf_score)

    return 0.5


def build_argument_graph(
    arguments: list[Argument],
    root_base_score: float = 0.5,
    *,
    root_text: str | None = None,
    target_item_name: str | None = None,
    allow_empty: bool = False,
) -> ArgumentGraph:
    """
    Build a minimal argument graph.

    Graph shape:
    - one root node = recommendation claim
    - each argument node points to the root
      with either a support or attack relation
    """
    if not arguments and not allow_empty:
        raise ValueError("Cannot build an argument graph from an empty argument list.")

    resolved_target_name = target_item_name
    if resolved_target_name is None and arguments:
        resolved_target_name = arguments[0].target_item_name
    resolved_target_name = resolved_target_name or "target_item"
    root_id = "ROOT"

    root_node = ArgumentNode(
        node_id=root_id,
        node_type="root",
        text=root_text or f"Recommend item: {resolved_target_name}",
        base_score=root_base_score,
        metadata={
            "target_item_name": resolved_target_name,
        },
    )

    nodes: dict[str, ArgumentNode] = {
        root_id: root_node,
    }
    edges: list[ArgumentEdge] = []

    for argument in arguments:
        node_metadata = {
            "arg_type": argument.arg_type,
            "evidence": argument.evidence,
            "used_aspects": argument.used_aspects,
            "aspect_effect": argument.aspect_effect,
            "llm_score": argument.llm_score,
            "llm_score_reason": argument.llm_score_reason,
            "mf_score": argument.mf_score,
            "combined_score": argument.combined_score,
        }
        if argument.argument_family is not None:
            node_metadata.update(
                {
                    "argument_family": argument.argument_family,
                    "aspect": argument.aspect,
                    "intrinsic_strength": argument.intrinsic_strength,
                    "importance_raw": argument.importance_raw,
                    "normalized_weight": argument.normalized_weight,
                    "evidence_score": argument.evidence_score,
                    "n_support": argument.n_support,
                    "n_attack": argument.n_attack,
                    "n_neutral": argument.n_neutral,
                    "review_sources": argument.review_sources,
                }
            )
            if (
                argument.source_refs
                or argument.preference_refs
                or argument.scoring_unit_id is not None
                or argument.explanatory_only
            ):
                node_metadata.update(
                    {
                        "source_refs": argument.source_refs,
                        "preference_refs": argument.preference_refs,
                        "scoring_unit_id": argument.scoring_unit_id,
                        "explanatory_only": argument.explanatory_only,
                    }
                )

        node = ArgumentNode(
            node_id=argument.id,
            node_type="argument",
            text=argument.text,
            base_score=get_argument_base_score(argument),
            metadata=node_metadata,
        )
        nodes[node.node_id] = node

        relation_type = "support" if argument.is_support() else "attack"

        edges.append(
            ArgumentEdge(
                source_id=argument.id,
                target_id=root_id,
                relation_type=relation_type,
            )
        )

    return ArgumentGraph(
        root_id=root_id,
        nodes=nodes,
        edges=edges,
    )
