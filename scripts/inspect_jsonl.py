import json
import argparse
from pathlib import Path


def load_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarize_examples(examples):
    if not examples:
        print("No examples to summarize.")
        return

    history_sizes = []
    target_text_lengths = []
    positive = 0
    negative = 0
    neutral = 0

    for ex in examples:
        history_sizes.append(len(ex.get("history", [])))

        target_item = ex.get("target_item", {})
        target_text = target_item.get("target_review_text", "")
        target_text_lengths.append(len(target_text))

        target_rating = target_item.get("user_target_stars")
        if target_rating is not None:
            if target_rating >= 4:
                positive += 1
            elif target_rating <= 2:
                negative += 1
            else:
                neutral += 1

    avg_history = sum(history_sizes) / len(history_sizes)
    avg_target_text_length = sum(target_text_lengths) / len(target_text_lengths)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Number of displayed examples: {len(examples)}")
    print(f"Average history size: {avg_history:.2f}")
    print(f"Average target review length: {avg_target_text_length:.2f} characters")
    print(f"Positive targets (rating >= 4): {positive}")
    print(f"Negative targets (rating <= 2): {negative}")
    print(f"Neutral targets (rating = 3): {neutral}")
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
        target_item = obj.get("target_item", {})
        target_name = target_item.get("name", "N/A")
        target_rating = target_item.get("user_target_stars", "N/A")
        target_text = target_item.get("target_review_text", "")

        print("=" * 80)
        print(f"Example {i + 1}")
        print("=" * 80)
        print(f"user_id: {obj.get('user_id', 'N/A')}")
        print(f"history size: {len(obj.get('history', []))}")
        print(f"target item: {target_name}")
        print(f"target rating: {target_rating}")
        print(f"target review length: {len(target_text)} characters")
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