import json
import argparse
from pathlib import Path


def load_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def count_history_sentiment(history):
    positive = sum(1 for item in history if item.get("user_stars", 0) >= 4)
    negative = sum(1 for item in history if item.get("user_stars", 0) <= 2)
    neutral = sum(1 for item in history if item.get("user_stars", 0) == 3)
    return positive, negative, neutral


def summarize_examples(examples):
    if not examples:
        print("No examples to summarize.")
        return

    history_sizes = []
    target_text_lengths = []
    positive_targets = 0
    negative_targets = 0
    neutral_targets = 0

    total_history_positive = 0
    total_history_negative = 0
    total_history_neutral = 0

    for ex in examples:
        history = ex.get("history", [])
        history_sizes.append(len(history))

        hist_pos, hist_neg, hist_neu = count_history_sentiment(history)
        total_history_positive += hist_pos
        total_history_negative += hist_neg
        total_history_neutral += hist_neu

        target_item = ex.get("target_item", {})
        target_text = target_item.get("target_review_text", "")
        target_text_lengths.append(len(target_text))

        target_rating = target_item.get("user_target_stars")
        if target_rating is not None:
            if target_rating >= 4:
                positive_targets += 1
            elif target_rating <= 2:
                negative_targets += 1
            else:
                neutral_targets += 1

    avg_history = sum(history_sizes) / len(history_sizes)
    avg_target_text_length = sum(target_text_lengths) / len(target_text_lengths)

    avg_hist_pos = total_history_positive / len(examples)
    avg_hist_neg = total_history_negative / len(examples)
    avg_hist_neu = total_history_neutral / len(examples)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of displayed examples: {len(examples)}")
    print(f"Average history size: {avg_history:.2f}")
    print(f"Average target review length: {avg_target_text_length:.2f} characters")
    print(f"Positive targets (rating >= 4): {positive_targets}")
    print(f"Negative targets (rating <= 2): {negative_targets}")
    print(f"Neutral targets (rating = 3): {neutral_targets}")
    print(f"Average positive items in history: {avg_hist_pos:.2f}")
    print(f"Average negative items in history: {avg_hist_neg:.2f}")
    print(f"Average neutral items in history: {avg_hist_neu:.2f}")
    print()


def inspect_jsonl(file_path: Path, num_examples: int = 3) -> None:
    if not file_path.exists():
        print(f"Error: file not found -> {file_path}")
        return

    print(f"\nInspecting file: {file_path}")

    all_examples = load_jsonl(file_path)
    total_lines = len(all_examples)

    print(f"Total number of examples in file: {total_lines}")
    print(f"Showing first {num_examples} example(s)\n")

    displayed_examples = all_examples[:num_examples]

    for i, obj in enumerate(displayed_examples):
        history = obj.get("history", [])
        hist_pos, hist_neg, hist_neu = count_history_sentiment(history)

        target_item = obj.get("target_item", {})
        target_name = target_item.get("name", "N/A")
        target_rating = target_item.get("user_target_stars", "N/A")
        target_text = target_item.get("target_review_text", "")
        target_attributes = target_item.get("attributes", {})

        print("=" * 80)
        print(f"Example {i + 1}")
        print("=" * 80)
        print(f"user_id: {obj.get('user_id', 'N/A')}")
        print(f"history size: {len(history)}")
        print(f"history positives: {hist_pos}")
        print(f"history negatives: {hist_neg}")
        print(f"history neutrals: {hist_neu}")
        print(f"target item: {target_name}")
        print(f"target rating: {target_rating}")
        print(f"target review length: {len(target_text)} characters")
        print(f"target attributes count: {len(target_attributes)}")
        print("-" * 80)
        print(json.dumps(obj, indent=2, ensure_ascii=False))
        print()

    summarize_examples(displayed_examples)


def main():
    parser = argparse.ArgumentParser(description="Inspect a JSONL file and display examples.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the JSONL file to inspect"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of examples to display"
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    inspect_jsonl(file_path, args.n)


if __name__ == "__main__":
    main()