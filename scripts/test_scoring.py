import argparse
import json
from pathlib import Path

from src.argumentation.schema import build_arguments_from_parsed_json
from src.argumentation.scoring import (
    DummyLLMScorer,
    DummyMFScorer,
    ScoreConfig,
    score_arguments,
)


def load_example(jsonl_path: Path, index: int) -> dict:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)

    raise IndexError(f"Index {index} out of range.")


def load_generated_record(results_jsonl_path: Path, index: int) -> dict:
    """
    Load one generated batch record by dataset index.
    """
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

    scored_arguments = score_arguments(
        arguments=arguments,
        llm_scorer=DummyLLMScorer(),
        mf_scorer=DummyMFScorer(),
        config=ScoreConfig(
            llm_weight=args.llm_weight,
            mf_weight=args.mf_weight,
        ),
    )

    print("=" * 100)
    print(f"DATASET INDEX: {args.index}")
    print(f"USER ID:       {example.get('user_id')}")
    print(f"TARGET ITEM:   {example.get('target_item', {}).get('name')}")
    print("=" * 100)
    print()

    for argument in scored_arguments:
        print("-" * 100)
        print(f"ID:             {argument.id}")
        print(f"TYPE:           {argument.arg_type}")
        print(f"TEXT:           {argument.text}")
        print(f"EVIDENCE:       {argument.evidence}")
        print(f"LLM SCORE:      {argument.llm_score}")
        print(f"MF SCORE:       {argument.mf_score}")
        print(f"COMBINED SCORE: {argument.combined_score}")
        print()

if __name__ == "__main__":
    main()