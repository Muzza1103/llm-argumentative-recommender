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


def load_example(jsonl_path: Path, index: int) -> dict:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)

    raise IndexError(f"Index {index} out of range.")


def load_generated_record(results_jsonl_path: Path, index: int) -> dict:
    with results_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("index") == index:
                return record

    raise ValueError(f"No generated record found for dataset index {index}.")


def main():
    parser = argparse.ArgumentParser(
        description="Test argument structuring and scoring on one generated example."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset JSONL file.",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the generated results JSONL file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset index to inspect.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name.",
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
        "--show-prompt",
        action="store_true",
        help="Display the full scoring prompt for each argument.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Display the raw LLM scoring output for each argument.",
    )
    args = parser.parse_args()

    example = load_example(Path(args.input), args.index)
    generated_record = load_generated_record(Path(args.results), args.index)

    parsed_json = generated_record.get("parsed_json")
    if parsed_json is None:
        raise ValueError(
            f"Record at dataset index {args.index} has no parsed_json. "
            "Choose a valid generated example."
        )

    arguments = build_arguments_from_parsed_json(parsed_json, example)

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

    scored_arguments = score_arguments(
        arguments=arguments,
        llm_scorer=llm_scorer,
        mf_scorer=mf_scorer,
        config=ScoreConfig(
            llm_weight=args.llm_weight,
            mf_weight=args.mf_weight,
        ),
    )

    print("=" * 100)
    print(f"DATASET INDEX: {args.index}")
    print(f"USER ID:       {example.get('user_id')}")
    print(f"TARGET ITEM:   {example.get('target_item', {}).get('name')}")
    print(f"MF SOURCE:     {mf_source}")
    print("=" * 100)
    print()

    for argument in scored_arguments:
        print("-" * 100)
        print(f"ID:             {argument.id}")
        print(f"TYPE:           {argument.arg_type}")
        print(f"TEXT:           {argument.text}")
        print(f"EVIDENCE:       {argument.evidence}")
        print(f"LLM SCORE:      {argument.llm_score}")
        print(f"LLM REASON:     {argument.llm_score_reason}")
        print(f"MF SCORE:       {argument.mf_score}")
        print(f"COMBINED SCORE: {argument.combined_score}")

        if args.show_prompt:
            print("\nSCORING PROMPT:")
            print(argument.llm_scoring_prompt)

        if args.show_raw:
            print("\nSCORING RAW OUTPUT:")
            print(argument.llm_scoring_raw_output)

        print()


if __name__ == "__main__":
    main()