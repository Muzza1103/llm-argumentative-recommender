import argparse
import json
from pathlib import Path

from src.llm.config import LLMConfig
from src.llm.generator import LocalLLMGenerator
from src.llm.loader import load_model_and_tokenizer
from src.llm.utils import extract_first_json_object
from src.prompting.review_aspect_prompt import (
    build_review_aspect_extraction_prompt,
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


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_aspect_vocabulary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_aspect(
    raw_aspect: str,
    allowed_aspects: list[str],
    normalization_map: dict[str, str],
) -> str | None:
    if not isinstance(raw_aspect, str):
        return None

    aspect = raw_aspect.strip().lower()
    if not aspect:
        return None

    aspect = normalization_map.get(aspect, aspect)

    if aspect not in allowed_aspects:
        return None

    return aspect


def parse_extracted_aspects(
    raw_output: str,
    allowed_aspects: list[str],
    normalization_map: dict[str, str],
) -> list[dict]:
    parsed = extract_first_json_object(raw_output)
    if not isinstance(parsed, dict):
        return []

    aspects = parsed.get("aspects", [])
    if not isinstance(aspects, list):
        return []

    normalized_results = []
    seen = set()

    for item in aspects:
        if not isinstance(item, dict):
            continue

        raw_name = item.get("name")
        polarity = item.get("polarity", "neutral")

        normalized_name = normalize_aspect(
            raw_aspect=raw_name,
            allowed_aspects=allowed_aspects,
            normalization_map=normalization_map,
        )
        if normalized_name is None:
            continue

        if polarity not in {"positive", "negative", "neutral"}:
            polarity = "neutral"

        key = (normalized_name, polarity)
        if key in seen:
            continue
        seen.add(key)

        normalized_results.append(
            {
                "name": normalized_name,
                "polarity": polarity,
            }
        )

    return normalized_results


def extract_review_aspects_for_item(
    item_name: str,
    rating: float | int | None,
    review_text: str,
    source: str,
    allowed_aspects: list[str],
    generator: LocalLLMGenerator,
    normalization_map: dict[str, str],
) -> tuple[list[dict], str]:
    prompt = build_review_aspect_extraction_prompt(
        item_name=item_name,
        rating=rating,
        review_text=review_text,
        source=source,
        allowed_aspects=allowed_aspects,
    )

    raw_output = generator.generate(prompt)

    aspects = parse_extracted_aspects(
        raw_output=raw_output,
        allowed_aspects=allowed_aspects,
        normalization_map=normalization_map,
    )

    return aspects, raw_output


def enrich_history_item(
    item: dict,
    user_id: str,
    allowed_aspects: list[str],
    normalization_map: dict[str, str],
    generator: LocalLLMGenerator,
    save_prompt: bool,
    save_raw: bool,
) -> dict:
    enriched = dict(item)

    review_text = item.get("review_text", "").strip()
    if not review_text:
        enriched["review_aspects"] = []
        return enriched

    prompt = build_review_aspect_extraction_prompt(
        item_name=item.get("name", ""),
        rating=item.get("user_stars"),
        review_text=review_text,
        source="history",
        allowed_aspects=allowed_aspects,
    )
    raw_output = generator.generate(prompt)

    aspects = parse_extracted_aspects(
        raw_output=raw_output,
        allowed_aspects=allowed_aspects,
        normalization_map=normalization_map,
    )

    enriched["review_aspects"] = aspects

    if save_prompt:
        enriched["review_aspect_prompt"] = prompt

    if save_raw:
        enriched["review_aspect_raw_output"] = raw_output

    return enriched


def enrich_target_item(
    target_item: dict,
    allowed_aspects: list[str],
    normalization_map: dict[str, str],
    generator: LocalLLMGenerator,
    save_prompt: bool,
    save_raw: bool,
) -> dict:
    enriched = dict(target_item)

    review_text = target_item.get("target_review_text", "").strip()
    if not review_text:
        enriched["review_aspects"] = []
        return enriched

    prompt = build_review_aspect_extraction_prompt(
        item_name=target_item.get("name", ""),
        rating=target_item.get("user_target_stars"),
        review_text=review_text,
        source="target",
        allowed_aspects=allowed_aspects,
    )
    raw_output = generator.generate(prompt)

    aspects = parse_extracted_aspects(
        raw_output=raw_output,
        allowed_aspects=allowed_aspects,
        normalization_map=normalization_map,
    )

    enriched["review_aspects"] = aspects

    if save_prompt:
        enriched["review_aspect_prompt"] = prompt

    if save_raw:
        enriched["review_aspect_raw_output"] = raw_output

    return enriched


def build_summary(
    input_file: str,
    output_file: str,
    vocab_file: str,
    model_name: str,
    num_examples: int,
    total_history_items: int,
    total_target_items: int,
    total_history_aspects: int,
    total_target_aspects: int,
) -> dict:
    return {
        "input_file": input_file,
        "output_file": output_file,
        "aspect_vocabulary_file": vocab_file,
        "model_name": model_name,
        "num_examples": num_examples,
        "total_history_items": total_history_items,
        "total_target_items": total_target_items,
        "total_history_aspects": total_history_aspects,
        "total_target_aspects": total_target_aspects,
        "total_aspects": total_history_aspects + total_target_aspects,
        "mean_history_aspects_per_item": (
            total_history_aspects / total_history_items if total_history_items else 0.0
        ),
        "mean_target_aspects_per_item": (
            total_target_aspects / total_target_items if total_target_items else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract normalized review aspects from Yelp reviews."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the source dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the enriched output JSONL file.",
    )
    parser.add_argument(
        "--aspect-vocab",
        type=str,
        default="configs/aspect_vocabulary.json",
        help="Path to the aspect vocabulary JSON file.",
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
        default=200,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Save review aspect extraction prompts in the output.",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw model outputs in the output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    vocab_path = Path(args.aspect_vocab)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    examples = load_jsonl(input_path)
    vocab = load_aspect_vocabulary(vocab_path)

    allowed_aspects = vocab.get("allowed_aspects", [])
    normalization_map = vocab.get("normalization_map", {})

    if not isinstance(allowed_aspects, list) or not all(
        isinstance(x, str) for x in allowed_aspects
    ):
        raise ValueError("Invalid aspect vocabulary: 'allowed_aspects' must be a list[str].")

    if not isinstance(normalization_map, dict):
        raise ValueError("Invalid aspect vocabulary: 'normalization_map' must be a dict.")

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

    enriched_examples = []

    total_history_items = 0
    total_target_items = 0
    total_history_aspects = 0
    total_target_aspects = 0

    print(f"Loaded {len(examples)} examples from {input_path}")
    print(f"Using aspect vocabulary from {vocab_path}")
    print(f"Allowed aspects: {len(allowed_aspects)}")

    for example_idx, example in enumerate(examples, start=1):
        enriched_example = dict(example)

        enriched_history = []
        for item in example.get("history", []):
            enriched_item = enrich_history_item(
                item=item,
                user_id=example.get("user_id", ""),
                allowed_aspects=allowed_aspects,
                normalization_map=normalization_map,
                generator=generator,
                save_prompt=args.save_prompts,
                save_raw=args.save_raw,
            )
            enriched_history.append(enriched_item)

            total_history_items += 1
            total_history_aspects += len(enriched_item.get("review_aspects", []))

        enriched_target = enrich_target_item(
            target_item=example.get("target_item", {}),
            allowed_aspects=allowed_aspects,
            normalization_map=normalization_map,
            generator=generator,
            save_prompt=args.save_prompts,
            save_raw=args.save_raw,
        )

        total_target_items += 1
        total_target_aspects += len(enriched_target.get("review_aspects", []))

        enriched_example["history"] = enriched_history
        enriched_example["target_item"] = enriched_target

        enriched_examples.append(enriched_example)

        print(
            f"[{example_idx}/{len(examples)}] "
            f"user_id={example.get('user_id')} "
            f"history_items={len(enriched_history)} "
            f"target_aspects={len(enriched_target.get('review_aspects', []))}"
        )

    save_jsonl(enriched_examples, output_path)

    summary = build_summary(
        input_file=str(input_path),
        output_file=str(output_path),
        vocab_file=str(vocab_path),
        model_name=args.model,
        num_examples=len(enriched_examples),
        total_history_items=total_history_items,
        total_target_items=total_target_items,
        total_history_aspects=total_history_aspects,
        total_target_aspects=total_target_aspects,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Enriched dataset: {output_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()