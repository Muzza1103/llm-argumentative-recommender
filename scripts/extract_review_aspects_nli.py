import argparse
import json
from pathlib import Path

from transformers import pipeline


DEFAULT_ASPECTS = [
    "food",
    "service",
    "ambience",
    "price",
    "portions",
    "drinks",
    "speed",
    "takeout",
    "delivery",
    "reservations",
    "good_for_groups",
    "good_for_kids",
    "noise",
    "freshness",
    "quality",
    "location",
    "spice_level",
    "crowdedness",
    "selection",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_aspect_vocabulary(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_ASPECTS

    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    aspects = vocab.get("allowed_aspects", DEFAULT_ASPECTS)

    if not isinstance(aspects, list):
        return DEFAULT_ASPECTS

    return [aspect for aspect in aspects if isinstance(aspect, str)]


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def detect_aspects(
    review_text: str,
    zero_shot_classifier,
    allowed_aspects: list[str],
    threshold: float,
    max_aspects: int,
) -> list[str]:
    if not review_text.strip():
        return []

    candidate_labels = [
        f"this review discusses {aspect.replace('_', ' ')}"
        for aspect in allowed_aspects
    ]

    result = zero_shot_classifier(
        review_text,
        candidate_labels=candidate_labels,
        multi_label=True,
    )

    detected = []

    for label, score in zip(result["labels"], result["scores"]):
        if score < threshold:
            continue

        label_text = label.replace("this review discusses ", "")
        aspect = label_text.replace(" ", "_")

        if aspect in allowed_aspects:
            detected.append((aspect, float(score)))

    detected.sort(key=lambda x: x[1], reverse=True)

    return [aspect for aspect, _ in detected[:max_aspects]]


def detect_polarity(
    review_text: str,
    sentiment_classifier,
) -> str:
    if not review_text.strip():
        return "neutral"

    result = sentiment_classifier(review_text[:1500])[0]
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))

    if score < 0.55:
        return "neutral"

    if "positive" in label or label in {"pos", "label_2", "5 stars", "4 stars"}:
        return "positive"

    if "negative" in label or label in {"neg", "label_0", "1 star", "2 stars"}:
        return "negative"

    return "neutral"


def extract_review_aspects(
    review_text: str,
    zero_shot_classifier,
    sentiment_classifier,
    allowed_aspects: list[str],
    aspect_threshold: float,
    max_aspects: int,
    max_chars: int,
) -> list[dict]:
    review_text = truncate_text(review_text, max_chars=max_chars)

    aspects = detect_aspects(
        review_text=review_text,
        zero_shot_classifier=zero_shot_classifier,
        allowed_aspects=allowed_aspects,
        threshold=aspect_threshold,
        max_aspects=max_aspects,
    )

    if not aspects:
        return []

    polarity = detect_polarity(
        review_text=review_text,
        sentiment_classifier=sentiment_classifier,
    )

    return [
        {
            "name": aspect,
            "polarity": polarity,
        }
        for aspect in aspects
    ]


def build_summary(
    records: list[dict],
    input_file: str,
    output_file: str,
    aspect_model: str,
    sentiment_model: str,
    aspect_threshold: float,
) -> dict:
    total_history_items = 0
    total_target_items = 0
    total_history_aspects = 0
    total_target_aspects = 0

    for record in records:
        for item in record.get("history", []):
            total_history_items += 1
            total_history_aspects += len(item.get("review_aspects", []))

        target_item = record.get("target_item", {})
        total_target_items += 1
        total_target_aspects += len(target_item.get("review_aspects", []))

    return {
        "input_file": input_file,
        "output_file": output_file,
        "aspect_model": aspect_model,
        "sentiment_model": sentiment_model,
        "aspect_threshold": aspect_threshold,
        "num_examples": len(records),
        "total_history_items": total_history_items,
        "total_target_items": total_target_items,
        "total_history_aspects": total_history_aspects,
        "total_target_aspects": total_target_aspects,
        "mean_history_aspects_per_item": (
            total_history_aspects / total_history_items if total_history_items else 0.0
        ),
        "mean_target_aspects_per_item": (
            total_target_aspects / total_target_items if total_target_items else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract review aspects using NLI zero-shot classification."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--aspect-vocab",
        type=str,
        default="configs/aspect_vocabulary.json",
    )
    parser.add_argument(
        "--aspect-model",
        type=str,
        default="facebook/bart-large-mnli",
    )
    parser.add_argument(
        "--sentiment-model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--max-aspects",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2500,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    vocab_path = Path(args.aspect_vocab) if args.aspect_vocab else None
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    allowed_aspects = load_aspect_vocabulary(vocab_path)
    records = load_jsonl(input_path)

    print(f"Loaded examples: {len(records)}")
    print(f"Allowed aspects: {len(allowed_aspects)}")
    print(f"Aspect model:    {args.aspect_model}")
    print(f"Sentiment model: {args.sentiment_model}")

    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=args.aspect_model,
        device=0,
    )

    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model=args.sentiment_model,
        device=0,
    )

    enriched_records = []

    total_jobs = sum(len(record.get("history", [])) + 1 for record in records)
    current_job = 0

    for record_index, record in enumerate(records, start=1):
        enriched_record = dict(record)
        enriched_record["history"] = [
            dict(item) for item in record.get("history", [])
        ]
        enriched_record["target_item"] = dict(record.get("target_item", {}))

        for item in enriched_record["history"]:
            current_job += 1
            print(f"[{current_job}/{total_jobs}] history review")

            review_text = item.get("review_text", "")
            item["review_aspects"] = extract_review_aspects(
                review_text=review_text,
                zero_shot_classifier=zero_shot_classifier,
                sentiment_classifier=sentiment_classifier,
                allowed_aspects=allowed_aspects,
                aspect_threshold=args.aspect_threshold,
                max_aspects=args.max_aspects,
                max_chars=args.max_chars,
            )

        target_item = enriched_record["target_item"]
        current_job += 1
        print(f"[{current_job}/{total_jobs}] target review")

        target_review_text = target_item.get("target_review_text", "")
        target_item["review_aspects"] = extract_review_aspects(
            review_text=target_review_text,
            zero_shot_classifier=zero_shot_classifier,
            sentiment_classifier=sentiment_classifier,
            allowed_aspects=allowed_aspects,
            aspect_threshold=args.aspect_threshold,
            max_aspects=args.max_aspects,
            max_chars=args.max_chars,
        )

        enriched_records.append(enriched_record)

        print(
            f"Example {record_index}/{len(records)} | "
            f"user_id={record.get('user_id')} | "
            f"target_aspects={len(target_item.get('review_aspects', []))}"
        )

    save_jsonl(enriched_records, output_path)

    summary = build_summary(
        records=enriched_records,
        input_file=str(input_path),
        output_file=str(output_path),
        aspect_model=args.aspect_model,
        sentiment_model=args.sentiment_model,
        aspect_threshold=args.aspect_threshold,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOutput:  {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()