from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.argumentation.dfquad import evaluate_root_dfquad
from src.argumentation.graph_builder import (
    ArgumentEdge,
    ArgumentGraph,
    ArgumentNode,
    build_argument_graph,
)
from src.argumentation.schema import build_arguments_from_scored_json


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_record(path: Path, index: int | None = None) -> dict[str, Any]:
    """Load one evaluation record from JSON or JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}."
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} of {path} is not a JSON object."
                    )
                records.append(record)

        if index is not None:
            for record in records:
                if record.get("index") == index:
                    return record
            raise ValueError(f"No record with index={index} found in {path}.")

        if len(records) != 1:
            raise ValueError(
                "A JSONL file containing several records requires --index."
            )
        return records[0]

    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)

    if not isinstance(record, dict):
        raise ValueError("The input JSON must contain one evaluation object.")
    if index is not None and record.get("index") != index:
        raise ValueError(
            f"The JSON record has index={record.get('index')}, not index={index}."
        )
    return record


def graph_from_dict(payload: dict[str, Any]) -> ArgumentGraph:
    """Restore an ArgumentGraph saved by test_dfquad.py or dfquad_batch.py."""
    root_id = payload.get("root_id")
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")

    if not isinstance(root_id, str) or not root_id:
        raise ValueError("The saved argument graph has no valid root_id.")
    if not isinstance(raw_nodes, dict) or not isinstance(raw_edges, list):
        raise ValueError("The saved argument graph is incomplete.")

    nodes: dict[str, ArgumentNode] = {}
    for node_id, raw_node in raw_nodes.items():
        if not isinstance(node_id, str) or not isinstance(raw_node, dict):
            raise ValueError("The saved argument graph contains an invalid node.")

        base_score = raw_node.get("base_score")
        if not isinstance(base_score, (int, float)):
            raise ValueError(f"Node {node_id!r} has no numeric base_score.")

        nodes[node_id] = ArgumentNode(
            node_id=node_id,
            node_type=str(raw_node.get("node_type", "argument")),
            text=str(raw_node.get("text", "")),
            base_score=float(base_score),
            metadata=copy.deepcopy(raw_node.get("metadata", {})),
        )

    if root_id not in nodes:
        raise ValueError(f"Root node {root_id!r} is missing from the graph.")

    edges = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            raise ValueError("The saved argument graph contains an invalid edge.")

        source_id = raw_edge.get("source_id")
        target_id = raw_edge.get("target_id")
        relation_type = raw_edge.get("relation_type")

        if source_id not in nodes or target_id not in nodes:
            raise ValueError("A saved graph edge references an unknown node.")
        if relation_type not in {"support", "attack"}:
            raise ValueError(f"Invalid relation type: {relation_type!r}.")

        edges.append(
            ArgumentEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
            )
        )

    return ArgumentGraph(root_id=root_id, nodes=nodes, edges=edges)


def build_graph_from_record(record: dict[str, Any]) -> ArgumentGraph:
    saved_graph = record.get("argument_graph")
    if isinstance(saved_graph, dict):
        return graph_from_dict(saved_graph)

    scored_arguments = record.get("scored_arguments")
    if not isinstance(scored_arguments, list) or not scored_arguments:
        raise ValueError(
            "The input must contain argument_graph or non-empty scored_arguments."
        )

    dfquad = record.get("dfquad", {})
    root_base_score = dfquad.get(
        "effective_root_base_score",
        dfquad.get("root_base_score", 0.5),
    )
    if not isinstance(root_base_score, (int, float)):
        root_base_score = 0.5

    arguments = build_arguments_from_scored_json(scored_arguments)
    return build_argument_graph(arguments, root_base_score=float(root_base_score))


def evaluate_with_saved_configuration(
    graph: ArgumentGraph,
    saved_dfquad: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recompute DF-QuAD and preserve an optional MF-aware final combination."""
    config = saved_dfquad if isinstance(saved_dfquad, dict) else {}

    result = evaluate_root_dfquad(
        graph,
        aggregation_method=config.get("aggregation_method", "dfquad"),
        combination_method=config.get("combination_method", "dfquad"),
        contrastive_gamma=float(config.get("contrastive_gamma", 5.0)),
        calibration_method=config.get("calibration_method", "none"),
        calibration_beta=float(config.get("calibration_beta", 12.0)),
    )
    output = result.to_dict()
    output["dfquad_score_before_mf_mix"] = result.final_score

    mf_mode = config.get("mf_combination_mode", "none")
    mf_score = config.get("mf_item_score")
    mf_lambda = config.get("mf_lambda", 0.0)
    no_clamp = bool(config.get("no_clamp_final_score", False))

    if mf_mode == "linear_mix" and isinstance(mf_score, (int, float)):
        mixed_score = (
            float(mf_lambda) * float(mf_score)
            + (1.0 - float(mf_lambda)) * result.final_score
        )
        output["final_score"] = mixed_score if no_clamp else clamp_score(mixed_score)

    elif mf_mode == "mf_correction" and isinstance(mf_score, (int, float)):
        correction = result.aggregated_support - result.aggregated_attack
        corrected_score = float(mf_score) + float(mf_lambda) * correction
        output["argumentative_correction"] = correction
        output["final_score"] = (
            corrected_score if no_clamp else clamp_score(corrected_score)
        )

    for key in (
        "mf_item_score",
        "mf_lambda",
        "mf_combination_mode",
        "no_clamp_final_score",
        "root_base_source",
        "effective_root_base_score",
    ):
        if key in config:
            output[key] = config[key]

    return output


def argument_rows(graph: ArgumentGraph) -> list[dict[str, Any]]:
    relation_by_source = {
        edge.source_id: edge.relation_type
        for edge in graph.edges
        if edge.target_id == graph.root_id
    }
    return [
        {
            "id": node.node_id,
            "type": relation_by_source.get(node.node_id),
            "strength": node.base_score,
            "text": node.text,
        }
        for node in graph.get_argument_nodes()
    ]


def contest_record(
    record: dict[str, Any],
    argument_id: str,
    *,
    delta: float | None = None,
    new_strength: float | None = None,
) -> dict[str, Any]:
    if (delta is None) == (new_strength is None):
        raise ValueError("Provide exactly one of delta or new_strength.")

    graph_before = build_graph_from_record(record)
    if argument_id == graph_before.root_id:
        raise ValueError("This first version only contests argument nodes, not the root.")
    if argument_id not in graph_before.nodes:
        available = ", ".join(row["id"] for row in argument_rows(graph_before))
        raise ValueError(
            f"Unknown argument id {argument_id!r}. Available ids: {available or 'none'}."
        )

    selected_before = graph_before.nodes[argument_id]
    if selected_before.node_type != "argument":
        raise ValueError(f"Node {argument_id!r} is not an argument.")

    original_strength = float(selected_before.base_score)
    requested_strength = (
        float(new_strength)
        if new_strength is not None
        else original_strength + float(delta)
    )
    effective_strength = clamp_score(requested_strength)

    graph_after = copy.deepcopy(graph_before)
    selected_after = graph_after.nodes[argument_id]
    selected_after.base_score = effective_strength
    selected_after.metadata["contested"] = True
    selected_after.metadata["strength_before_contestation"] = original_strength
    selected_after.metadata["strength_after_contestation"] = effective_strength

    saved_dfquad = record.get("dfquad")
    before_result = evaluate_with_saved_configuration(graph_before, saved_dfquad)
    after_result = evaluate_with_saved_configuration(graph_after, saved_dfquad)

    output = copy.deepcopy(record)
    output["contestation"] = {
        "type": "argument_strength_update",
        "argument_id": argument_id,
        "argument_type": selected_before.metadata.get("arg_type"),
        "argument_text": selected_before.text,
        "delta": delta,
        "requested_strength": requested_strength,
        "strength_before": original_strength,
        "strength_after": effective_strength,
        "clamped_to_unit_interval": requested_strength != effective_strength,
        "llm_called": False,
    }
    output["dfquad_before_contestation"] = before_result
    output["argument_graph_before_contestation"] = graph_before.to_dict()
    output["dfquad"] = after_result
    output["argument_graph"] = graph_after.to_dict()
    return output


def print_arguments(graph: ArgumentGraph) -> None:
    print("Available arguments:")
    for row in argument_rows(graph):
        print(
            f"- {row['id']} | {row['type']} | strength={row['strength']:.6f}"
        )
        print(f"  {row['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Contest the strength of one Yelp argument and recompute the score "
            "without calling the LLM."
        )
    )
    parser.add_argument("--input", required=True, help="Input evaluation JSON or JSONL.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Record index when --input is a multi-record JSONL file.",
    )
    parser.add_argument("--argument-id", default=None, help="Argument node to contest.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument(
        "--list-arguments",
        action="store_true",
        help="List the arguments and their current strengths, then exit.",
    )
    strength_group = parser.add_mutually_exclusive_group()
    strength_group.add_argument(
        "--delta",
        type=float,
        help="Signed change applied to the current strength (result is clamped to [0, 1]).",
    )
    strength_group.add_argument(
        "--new-strength",
        type=float,
        help="Replacement strength in [0, 1].",
    )
    args = parser.parse_args()

    record = load_record(Path(args.input), args.index)
    graph = build_graph_from_record(record)

    if args.list_arguments:
        print_arguments(graph)
        return

    if args.argument_id is None:
        parser.error("--argument-id is required unless --list-arguments is used.")
    if args.output is None:
        parser.error("--output is required when applying a contestation.")
    if args.delta is None and args.new_strength is None:
        parser.error("Provide either --delta or --new-strength.")
    if args.new_strength is not None and not 0.0 <= args.new_strength <= 1.0:
        parser.error("--new-strength must be in [0, 1].")

    contested = contest_record(
        record,
        args.argument_id,
        delta=args.delta,
        new_strength=args.new_strength,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(contested, stream, indent=2, ensure_ascii=False)

    before = contested["dfquad_before_contestation"]
    after = contested["dfquad"]
    change = contested["contestation"]

    print(f"Argument:           {change['argument_id']}")
    print(f"Strength:           {change['strength_before']:.6f} -> {change['strength_after']:.6f}")
    print(f"Aggregated support: {before['aggregated_support']:.6f} -> {after['aggregated_support']:.6f}")
    print(f"Aggregated attack:  {before['aggregated_attack']:.6f} -> {after['aggregated_attack']:.6f}")
    print(f"Final score:        {before['final_score']:.6f} -> {after['final_score']:.6f}")
    print(f"LLM called:         {change['llm_called']}")
    print(f"Saved to:           {output_path}")


if __name__ == "__main__":
    main()
