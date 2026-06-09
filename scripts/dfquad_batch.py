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

def load_mf_item_predictions(path):
    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    lookup = {}

    for pred in predictions:
        user_id = pred.get("user_id")
        business_id = pred.get("business_id")
        score = pred.get("score")

        if isinstance(user_id, str) and isinstance(business_id, str):
            if isinstance(score, (int, float)):
                lookup[(user_id, business_id)] = float(score)

    return lookup


def get_mf_item_score(example, mf_lookup):
    user_id = example.get("user_id")
    target_item = example.get("target_item", {})
    business_id = target_item.get("business_id")

    return mf_lookup.get((user_id, business_id))

def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))

def maybe_clamp_score(value: float, no_clamp: bool) -> float:
    if no_clamp:
        return value
    return clamp_score(value)

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
    parser.add_argument(
        "--root-base-source",
        choices=["constant", "mf_item"],
        default="constant",
        help="Source of the root/base recommendation score.",
    )
    parser.add_argument(
        "--mf-item-predictions",
        type=str,
        default=None,
        help="Path to MF-only item predictions JSON file.",
    )
    parser.add_argument(
        "--mf-lambda",
        type=float,
        default=0.0,
        help="Weight of MF item score in final score mixing.",
    )
    parser.add_argument(
        "--mf-combination-mode",
        choices=["none", "linear_mix", "mf_correction"],
        default="none",
        help=(
            "How to combine MF item score with argumentative aggregation. "
            "'none' keeps DF-QuAD output, "
            "'linear_mix' computes lambda * MF + (1-lambda) * DF-QuAD, "
            "'mf_correction' computes MF + lambda * (support - attack)."
        ),
    )
    parser.add_argument(
        "--argument-score-source",
        choices=["combined", "llm", "mf"],
        default="combined",
        help="Score used as argument strength during graph construction.",
    )
    parser.add_argument(
        "--no-clamp-final-score",
        action="store_true",
        help="Do not clamp the final MF-aware score to [0, 1]. Useful for ranking.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    records = load_jsonl(input_path)
    mf_item_lookup = load_mf_item_predictions(args.mf_item_predictions)

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

        example = None

        if dataset_by_index is not None:
            dataset_index = record.get("index")
            example = dataset_by_index.get(dataset_index)

        mf_item_score = None

        if example is not None:
            mf_item_score = get_mf_item_score(example, mf_item_lookup)

        root_base_score = args.root_base_score

        if args.root_base_source == "mf_item" and mf_item_score is not None:
            root_base_score = mf_item_score

        score_key_by_source = {
            "combined": "combined_score",
            "llm": "llm_score",
            "mf": "mf_score",
        }

        selected_score_key = score_key_by_source[args.argument_score_source]

        for argument_json in scored_arguments_json:
            selected_score = argument_json.get(selected_score_key)

            if isinstance(selected_score, (int, float)):
                argument_json["combined_score"] = float(selected_score)

        arguments = build_arguments_from_scored_json(scored_arguments_json)

        graph = build_argument_graph(
            arguments=arguments,
            root_base_score=root_base_score,
        )

        dfquad_result = evaluate_root_dfquad(
            graph,
            aggregation_method=args.aggregation_method,
            combination_method=args.combination_method,
            contrastive_gamma=args.contrastive_gamma,
            calibration_method=args.calibration_method,
            calibration_beta=args.calibration_beta,
        )

        dfquad_dict = dfquad_result.to_dict()
        dfquad_score_before_mf_mix = dfquad_dict.get("final_score")

        dfquad_dict["dfquad_score_before_mf_mix"] = dfquad_score_before_mf_mix

        if args.mf_combination_mode != "none":
            if mf_item_score is None:
                print(
                    f"Warning: MF item score missing for index={record.get('index')}. "
                    "Keeping original DF-QuAD score."
                )

            elif not isinstance(dfquad_score_before_mf_mix, (int, float)):
                print(
                    f"Warning: DF-QuAD score missing for index={record.get('index')}. "
                    "Keeping original DF-QuAD score."
                )

            elif args.mf_combination_mode == "linear_mix":
                mixed_score = (
                    args.mf_lambda * mf_item_score
                    + (1.0 - args.mf_lambda) * float(dfquad_score_before_mf_mix)
                )
                dfquad_dict["final_score"] = maybe_clamp_score(
                    mixed_score,
                    args.no_clamp_final_score,
                )

            elif args.mf_combination_mode == "mf_correction":
                aggregated_support = dfquad_dict.get("aggregated_support")
                aggregated_attack = dfquad_dict.get("aggregated_attack")

                if isinstance(aggregated_support, (int, float)) and isinstance(
                    aggregated_attack,
                    (int, float),
                ):
                    correction = float(aggregated_support) - float(aggregated_attack)
                    corrected_score = mf_item_score + args.mf_lambda * correction
                    dfquad_dict["argumentative_correction"] = correction
                    dfquad_dict["final_score"] = maybe_clamp_score(
                        corrected_score,
                        args.no_clamp_final_score,
                    )
                else:
                    print(
                        f"Warning: support/attack aggregation missing for "
                        f"index={record.get('index')}. Keeping original DF-QuAD score."
                    )

        dfquad_dict["mf_item_score"] = mf_item_score
        dfquad_dict["mf_lambda"] = args.mf_lambda
        dfquad_dict["mf_combination_mode"] = args.mf_combination_mode
        dfquad_dict["no_clamp_final_score"] = args.no_clamp_final_score
        dfquad_dict["root_base_source"] = args.root_base_source
        dfquad_dict["effective_root_base_score"] = root_base_score

        output_record = dict(record)
        output_record["dfquad"] = dfquad_dict

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
            f"final_score={dfquad_dict.get('final_score')}"
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
    summary["root_base_source"] = args.root_base_source
    summary["mf_item_predictions"] = args.mf_item_predictions
    summary["mf_lambda"] = args.mf_lambda
    summary["mf_combination_mode"] = args.mf_combination_mode
    summary["argument_score_source"] = args.argument_score_source
    summary["no_clamp_final_score"] = args.no_clamp_final_score

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDF-QuAD output: {output_path}")
    print(f"Summary:        {summary_path}")
    print(f"Processed:      {len(output_records)}")
    print(f"Skipped:        {skipped}")


if __name__ == "__main__":
    main()