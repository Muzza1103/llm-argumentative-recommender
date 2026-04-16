import argparse
import json
from pathlib import Path

from src.argumentation.schema import build_arguments_from_parsed_json
from src.argumentation.scoring import (
    ScoreConfig,
    score_arguments,
)
from src.argumentation.mf_scorer import (
    GlobalRatingFallbackMFScorer,
    PrecomputedMFScorer,
)
from src.argumentation.llm_scorer import LocalLLMScorer, LLMScorerConfig
from src.llm.config import LLMConfig
from src.llm.loader import load_model_and_tokenizer
from src.llm.generator import LocalLLMGenerator


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset_index(dataset_path: Path) -> dict[int, dict]:
    dataset_by_index = {}

    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            dataset_by_index[i] = json.loads(line)

    return dataset_by_index


def build_summary(
    scored_records: list[dict],
    input_file: str,
    output_file: str,
    dataset_file: str,
    llm_model: str,
    llm_weight: float,
    mf_weight: float,
    mf_source: str,
    skipped_records: int,
    only_valid: bool,
) -> dict:
    total_records = len(scored_records)

    total_arguments = 0
    llm_scores = []
    mf_scores = []
    combined_scores = []
    support_combined_scores = []
    attack_combined_scores = []

    for record in scored_records:
        for argument in record.get("scored_arguments", []):
            total_arguments += 1

            llm_score = argument.get("llm_score")
            mf_score = argument.get("mf_score")
            combined_score = argument.get("combined_score")
            arg_type = argument.get("arg_type")

            if isinstance(llm_score, (int, float)):
                llm_scores.append(float(llm_score))
            if isinstance(mf_score, (int, float)):
                mf_scores.append(float(mf_score))
            if isinstance(combined_score, (int, float)):
                combined_scores.append(float(combined_score))

                if arg_type == "support":
                    support_combined_scores.append(float(combined_score))
                elif arg_type == "attack":
                    attack_combined_scores.append(float(combined_score))

    def safe_mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "dataset_file": dataset_file,
        "input_file": input_file,
        "output_file": output_file,
        "llm_model": llm_model,
        "llm_weight": llm_weight,
        "mf_weight": mf_weight,
        "mf_source": mf_source,
        "only_valid": only_valid,
        "num_records_scored": total_records,
        "num_records_skipped": skipped_records,
        "num_arguments_scored": total_arguments,
        "mean_llm_score": safe_mean(llm_scores),
        "mean_mf_score": safe_mean(mf_scores),
        "mean_combined_score": safe_mean(combined_scores),
        "mean_support_combined_score": safe_mean(support_combined_scores),
        "mean_attack_combined_score": safe_mean(attack_combined_scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score generated arguments in batch using LLM + MF scoring."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the source dataset JSONL file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the generated results JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the scored output JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name for semantic scoring.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum number of generated tokens for scoring.",
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.5,
        help="Weight for the LLM semantic score.",
    )
    parser.add_argument(
        "--mf-weight",
        type=float,
        default=0.5,
        help="Weight for the MF empirical score.",
    )
    parser.add_argument(
        "--mf-predictions",
        type=str,
        default=None,
        help="Optional path to a JSON file containing precomputed MF predictions.",
    )
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="Score only records whose validation status is valid.",
    )
    parser.add_argument(
        "--save-llm-prompt",
        action="store_true",
        help="Keep the full LLM scoring prompt in the scored output.",
    )
    parser.add_argument(
        "--save-llm-raw",
        action="store_true",
        help="Keep the raw LLM scoring output in the scored output.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    dataset_by_index = load_dataset_index(dataset_path)
    records = load_jsonl(input_path)

    llm_config = LLMConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        do_sample=False,
    )

    tokenizer, model = load_model_and_tokenizer(llm_config)
    generator = LocalLLMGenerator(
        model=model,
        tokenizer=tokenizer,
        config=llm_config,
    )

    llm_scorer = LocalLLMScorer(
        generator=generator,
        config=LLMScorerConfig(default_score=0.5),
    )

    if args.mf_predictions is not None:
        mf_scorer = PrecomputedMFScorer.from_json(args.mf_predictions)
        mf_source = args.mf_predictions
    else:
        mf_scorer = GlobalRatingFallbackMFScorer()
        mf_source = "global_rating_fallback"

    score_config = ScoreConfig(
        llm_weight=args.llm_weight,
        mf_weight=args.mf_weight,
    )

    scored_records = []
    skipped_records = 0

    print(f"Loaded {len(records)} generated records from {input_path}")
    print(f"MF source: {mf_source}")

    for i, record in enumerate(records, start=1):
        validation = record.get("validation", {})
        is_valid = validation.get("is_valid", False)

        if args.only_valid and not is_valid:
            skipped_records += 1
            continue

        parsed_json = record.get("parsed_json")
        if parsed_json is None:
            skipped_records += 1
            continue

        dataset_index = record.get("index")
        if dataset_index not in dataset_by_index:
            skipped_records += 1
            continue

        example = dataset_by_index[dataset_index]

        arguments = build_arguments_from_parsed_json(parsed_json, example)

        scored_arguments = score_arguments(
            arguments=arguments,
            llm_scorer=llm_scorer,
            mf_scorer=mf_scorer,
            config=score_config,
        )

        scored_arguments_dicts = []
        for argument in scored_arguments:
            argument_dict = argument.to_dict()

            if not args.save_llm_prompt:
                argument_dict.pop("llm_scoring_prompt", None)

            if not args.save_llm_raw:
                argument_dict.pop("llm_scoring_raw_output", None)

            scored_arguments_dicts.append(argument_dict)

        enriched_record = dict(record)
        enriched_record["scoring"] = {
            "llm_model": args.model,
            "llm_weight": args.llm_weight,
            "mf_weight": args.mf_weight,
            "mf_source": mf_source,
        }
        enriched_record["scored_arguments"] = scored_arguments_dicts

        scored_records.append(enriched_record)

        print(
            f"[{i}/{len(records)}] "
            f"index={record.get('index')} "
            f"target={record.get('target_name')} "
            f"scored_arguments={len(scored_arguments_dicts)}"
        )

    save_jsonl(scored_records, output_path)

    summary = build_summary(
        scored_records=scored_records,
        input_file=str(input_path),
        output_file=str(output_path),
        dataset_file=str(dataset_path),
        llm_model=args.model,
        llm_weight=args.llm_weight,
        mf_weight=args.mf_weight,
        mf_source=mf_source,
        skipped_records=skipped_records,
        only_valid=args.only_valid,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Scored output:  {output_path}")
    print(f"Summary:        {summary_path}")
    print(f"Scored records: {len(scored_records)}")
    print(f"Skipped:        {skipped_records}")


if __name__ == "__main__":
    main()