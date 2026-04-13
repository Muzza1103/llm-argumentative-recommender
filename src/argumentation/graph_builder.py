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
) -> ArgumentGraph:
    """
    Build a minimal argument graph.

    Graph shape:
    - one root node = recommendation claim
    - each argument node points to the root
      with either a support or attack relation
    """
    if not arguments:
        raise ValueError("Cannot build an argument graph from an empty argument list.")

    target_item_name = arguments[0].target_item_name or "target_item"
    root_id = "ROOT"

    root_node = ArgumentNode(
        node_id=root_id,
        node_type="root",
        text=f"Recommend item: {target_item_name}",
        base_score=root_base_score,
        metadata={
            "target_item_name": target_item_name,
        },
    )

    nodes: dict[str, ArgumentNode] = {
        root_id: root_node,
    }
    edges: list[ArgumentEdge] = []

    for argument in arguments:
        node = ArgumentNode(
            node_id=argument.id,
            node_type="argument",
            text=argument.text,
            base_score=get_argument_base_score(argument),
            metadata={
                "arg_type": argument.arg_type,
                "evidence": argument.evidence,
                "llm_score": argument.llm_score,
                "llm_score_reason": argument.llm_score_reason,
                "mf_score": argument.mf_score,
                "combined_score": argument.combined_score,
            },
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