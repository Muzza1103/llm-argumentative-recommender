import argparse
import json
from pathlib import Path

from src.argumentation.schema import build_arguments_from_scored_json
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.dfquad import evaluate_root_dfquad
from scripts.test_dfquad import build_context_summary


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_summary(records: list[dict], skipped: int) -> dict:
    final_scores = []
    support_scores = []
    attack_scores = []

    for record in records:
        dfquad = record.get("dfquad", {})

        final_score = dfquad.get("final_score")
        aggregated_support = dfquad.get("aggregated_support")
        aggregated_attack = dfquad.get("aggregated_attack")

        if isinstance(final_score, (int, float)):
            final_scores.append(float(final_score))

        if isinstance(aggregated_support, (int, float)):
            support_scores.append(float(aggregated_support))

        if isinstance(aggregated_attack, (int, float)):
            attack_scores.append(float(aggregated_attack))

    def mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "num_records_processed": len(records),
        "num_records_skipped": skipped,
        "mean_final_score": mean(final_scores),
        "mean_aggregated_support": mean(support_scores),
        "mean_aggregated_attack": mean(attack_scores),
        "min_final_score": min(final_scores) if final_scores else None,
        "max_final_score": max(final_scores) if final_scores else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply argumentative aggregation to all scored argument records."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the scored arguments JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the DF-QuAD output JSONL file.",
    )
    parser.add_argument(
        "--root-base-score",
        type=float,
        default=0.5,
        help="Base score assigned to the root recommendation claim.",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save the full argument graph in each output record.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to the source dataset JSONL used to add context to each record.",
    )
    parser.add_argument(
        "--aggregation-method",
        choices=["dfquad", "mean", "max"],
        default="dfquad",
        help="Method used to aggregate support and attack strengths.",
    )
    parser.add_argument(
        "--combination-method",
        choices=["dfquad", "contrastive_power"],
        default="dfquad",
        help="Method used to combine aggregated support and attack strengths.",
    )
    parser.add_argument(
        "--contrastive-gamma",
        type=float,
        default=5.0,
        help="Gamma value for contrastive_power combination.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "centered_sigmoid"],
        default="none",
        help="Optional post-aggregation calibration method.",
    )
    parser.add_argument(
        "--calibration-beta",
        type=float,
        default=12.0,
        help="Beta value for centered sigmoid calibration.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    records = load_jsonl(input_path)

    dataset_by_index = None

    if args.dataset is not None:
        dataset_records = load_jsonl(Path(args.dataset))
        dataset_by_index = {
            index: example
            for index, example in enumerate(dataset_records)
        }

    output_records = []
    skipped = 0

    print(f"Loaded {len(records)} scored records from {input_path}")
    print(f"Aggregation method: {args.aggregation_method}")
    print(f"Combination method: {args.combination_method}")
    print(f"Contrastive gamma:  {args.contrastive_gamma}")
    print(f"Calibration method: {args.calibration_method}")
    print(f"Calibration beta:   {args.calibration_beta}")

    for i, record in enumerate(records, start=1):
        scored_arguments_json = record.get("scored_arguments")

        if not scored_arguments_json:
            skipped += 1
            continue

        arguments = build_arguments_from_scored_json(scored_arguments_json)

        graph = build_argument_graph(
            arguments=arguments,
            root_base_score=args.root_base_score,
        )

        dfquad_result = evaluate_root_dfquad(
            graph,
            aggregation_method=args.aggregation_method,
            combination_method=args.combination_method,
            contrastive_gamma=args.contrastive_gamma,
            calibration_method=args.calibration_method,
            calibration_beta=args.calibration_beta,
        )

        output_record = dict(record)
        output_record["dfquad"] = dfquad_result.to_dict()

        if dataset_by_index is not None:
            dataset_index = record.get("index")
            example = dataset_by_index.get(dataset_index)

            if example is not None:
                output_record["context"] = build_context_summary(example)

        if args.save_graph:
            output_record["argument_graph"] = graph.to_dict()

        output_records.append(output_record)

        print(
            f"[{i}/{len(records)}] "
            f"index={record.get('index')} "
            f"target={record.get('target_name')} "
            f"final_score={dfquad_result.final_score}"
        )

    save_jsonl(output_records, output_path)

    summary = build_summary(output_records, skipped)
    summary["input_file"] = str(input_path)
    summary["output_file"] = str(output_path)
    summary["root_base_score"] = args.root_base_score
    summary["aggregation_method"] = args.aggregation_method
    summary["combination_method"] = args.combination_method
    summary["contrastive_gamma"] = args.contrastive_gamma
    summary["calibration_method"] = args.calibration_method
    summary["calibration_beta"] = args.calibration_beta

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDF-QuAD output: {output_path}")
    print(f"Summary:        {summary_path}")
    print(f"Processed:      {len(output_records)}")
    print(f"Skipped:        {skipped}")


if __name__ == "__main__":
    main()