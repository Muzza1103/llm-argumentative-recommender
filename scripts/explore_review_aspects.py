import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.llm.config import LLMConfig
from src.llm.generator import LocalLLMGenerator
from src.llm.loader import load_model_and_tokenizer
from src.llm.utils import extract_first_json_object
from src.prompting.review_aspect_prompt import (
    build_review_aspect_exploration_prompt,
)


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def collect_reviews(
    examples: list[dict],
    include_target: bool = True,
) -> list[dict]:
    reviews = []

    for example in examples:
        user_id = example.get("user_id")

        for item in example.get("history", []):
            review_text = item.get("review_text", "").strip()
            if not review_text:
                continue

            reviews.append(
                {
                    "user_id": user_id,
                    "business_id": item.get("business_id"),
                    "item_name": item.get("name"),
                    "rating": item.get("user_stars"),
                    "review_text": review_text,
                    "source": "history",
                }
            )

        if include_target:
            target = example.get("target_item", {})
            review_text = target.get("target_review_text", "").strip()
            if review_text:
                reviews.append(
                    {
                        "user_id": user_id,
                        "business_id": target.get("business_id"),
                        "item_name": target.get("name"),
                        "rating": target.get("user_target_stars"),
                        "review_text": review_text,
                        "source": "target",
                    }
                )

    return reviews


def sample_reviews(
    reviews: list[dict],
    sample_size: int,
    seed: int,
) -> list[dict]:
    if sample_size >= len(reviews):
        return reviews

    rng = random.Random(seed)
    return rng.sample(reviews, sample_size)


def extract_aspects_from_response(raw_output: str) -> list[str]:
    parsed = extract_first_json_object(raw_output)
    if not isinstance(parsed, dict):
        return []

    aspects = parsed.get("aspects", [])
    if not isinstance(aspects, list):
        return []

    cleaned = []
    for aspect in aspects:
        if not isinstance(aspect, str):
            continue

        aspect = aspect.strip().lower()
        if not aspect:
            continue

        cleaned.append(aspect)

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Explore candidate review aspects from a sample of Yelp reviews."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of reviews to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Also include target reviews in the exploration sample.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save the full prompts for inspection.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model outputs for inspection.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    examples = load_jsonl(input_path)
    reviews = collect_reviews(examples, include_target=args.include_target)
    sampled_reviews = sample_reviews(reviews, args.sample_size, args.seed)

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

    explored_records = []
    aspect_counter = Counter()

    print(f"Loaded {len(examples)} examples.")
    print(f"Collected {len(reviews)} reviews.")
    print(f"Sampled {len(sampled_reviews)} reviews.")

    for i, review_record in enumerate(sampled_reviews, start=1):
        prompt = build_review_aspect_exploration_prompt(
            item_name=review_record.get("item_name", ""),
            rating=review_record.get("rating"),
            review_text=review_record.get("review_text", ""),
            source=review_record.get("source", ""),
        )

        raw_output = generator.generate(prompt)
        aspects = extract_aspects_from_response(raw_output)

        for aspect in aspects:
            aspect_counter[aspect] += 1

        enriched = dict(review_record)
        enriched["candidate_aspects"] = aspects

        if args.save_prompts:
            enriched["prompt"] = prompt

        if args.save_raw:
            enriched["raw_output"] = raw_output

        explored_records.append(enriched)

        print(
            f"[{i}/{len(sampled_reviews)}] "
            f"item={review_record.get('item_name')} "
            f"aspects={aspects}"
        )

    top_aspects = [
        {"aspect": aspect, "count": count}
        for aspect, count in aspect_counter.most_common()
    ]

    output = {
        "input_file": str(input_path),
        "model": args.model,
        "sample_size_requested": args.sample_size,
        "sample_size_used": len(sampled_reviews),
        "records": explored_records,
        "top_aspects": top_aspects,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "model": args.model,
        "num_examples": len(examples),
        "num_reviews_collected": len(reviews),
        "num_reviews_sampled": len(sampled_reviews),
        "num_unique_candidate_aspects": len(aspect_counter),
        "top_20_aspects": top_aspects[:20],
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Exploration output: {output_path}")
    print(f"Summary:            {summary_path}")
    print(f"Unique aspects:     {len(aspect_counter)}")


if __name__ == "__main__":
    main()