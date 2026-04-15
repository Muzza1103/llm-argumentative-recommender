import argparse
import json
from pathlib import Path

from src.argumentation.schema import build_arguments_from_scored_json
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.dfquad import evaluate_root_dfquad
from src.prompting.formatters import get_filtered_attributes


def load_record_by_index(jsonl_path: Path, index: int) -> dict:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("index") == index:
                return record

    raise ValueError(f"No record found for dataset index {index}.")


def load_example_by_index(dataset_path: Path, index: int) -> dict:
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)

    raise ValueError(f"No dataset example found for index {index}.")


def normalize_rating(rating: float | None) -> float | None:
    if rating is None:
        return None
    return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))


def build_context_summary(example: dict) -> dict:
    target_item = example.get("target_item", {})
    history = example.get("history", [])

    user_target_stars = target_item.get("user_target_stars")
    normalized_target_score = normalize_rating(user_target_stars)

    history_summary = []
    for item in history:
        history_summary.append(
            {
                "name": item.get("name"),
                "user_stars": item.get("user_stars"),
                "categories": item.get("categories", []),
                "attributes": get_filtered_attributes(item.get("attributes", {})),
            }
        )

    return {
        "user_id": example.get("user_id"),
        "target_item": {
            "name": target_item.get("name"),
            "categories": target_item.get("categories", []),
            "attributes": get_filtered_attributes(target_item.get("attributes", {})),
            "global_stars": target_item.get("global_stars"),
            "user_target_stars": user_target_stars,
            "normalized_user_target_score": normalized_target_score,
        },
        "history": history_summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test DF-QuAD aggregation on one scored example."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the scored arguments JSONL file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to inspect.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to the source dataset JSONL file for adding context to the saved output.",
    )
    parser.add_argument(
        "--root-base-score",
        type=float,
        default=0.5,
        help="Base score assigned to the root recommendation claim.",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the graph structure as JSON.",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Display scored arguments.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the DF-QuAD result as JSON.",
    )
    args = parser.parse_args()

    record = load_record_by_index(Path(args.input), args.index)

    scored_arguments_json = record.get("scored_arguments")
    if not scored_arguments_json:
        raise ValueError(
            f"Record at dataset index {args.index} has no scored_arguments."
        )

    arguments = build_arguments_from_scored_json(scored_arguments_json)

    graph = build_argument_graph(
        arguments=arguments,
        root_base_score=args.root_base_score,
    )

    dfquad_result = evaluate_root_dfquad(graph)

    print("=" * 100)
    print(f"DATASET INDEX:   {record.get('index')}")
    print(f"USER ID:         {record.get('user_id')}")
    print(f"TARGET ITEM:     {record.get('target_name')}")
    print(f"ROOT BASE SCORE: {args.root_base_score}")
    print("=" * 100)
    print()

    if args.show_scores:
        print("SCORED ARGUMENTS")
        print("=" * 100)
        for argument in arguments:
            print("-" * 100)
            print(f"ID:             {argument.id}")
            print(f"TYPE:           {argument.arg_type}")
            print(f"TEXT:           {argument.text}")
            print(f"EVIDENCE:       {argument.evidence}")
            print(f"LLM SCORE:      {argument.llm_score}")
            print(f"LLM REASON:     {argument.llm_score_reason}")
            print(f"MF SCORE:       {argument.mf_score}")
            print(f"COMBINED SCORE: {argument.combined_score}")
            print()

    if args.show_graph:
        print("=" * 100)
        print("ARGUMENT GRAPH")
        print("=" * 100)
        print(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False))
        print()

    print("=" * 100)
    print("DF-QUAD RESULT")
    print("=" * 100)
    print(f"ROOT ID:              {dfquad_result.root_id}")
    print(f"ROOT TEXT:            {dfquad_result.root_text}")
    print(f"ROOT BASE SCORE:      {dfquad_result.root_base_score}")
    print(f"SUPPORT SCORES:       {dfquad_result.support_scores}")
    print(f"ATTACK SCORES:        {dfquad_result.attack_scores}")
    print(f"AGGREGATED SUPPORT:   {dfquad_result.aggregated_support}")
    print(f"AGGREGATED ATTACK:    {dfquad_result.aggregated_attack}")
    print(f"FINAL SCORE:          {dfquad_result.final_score}")
    print()

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_record = dict(record)
        output_record["argument_graph"] = graph.to_dict()
        output_record["dfquad"] = dfquad_result.to_dict()

        if args.dataset is not None:
            example = load_example_by_index(Path(args.dataset), args.index)
            output_record["context"] = build_context_summary(example)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_record, f, indent=2, ensure_ascii=False)

        print(f"Saved DF-QuAD output to: {output_path}")

if __name__ == "__main__":
    main()