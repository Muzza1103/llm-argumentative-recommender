import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.prompting.argument_prompt import build_prompt
from src.prompting.formatters import format_history, format_target_item


def load_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description="Build and display a prompt from a JSONL example.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the JSONL file"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the example to use"
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    examples = load_jsonl(file_path)

    if not examples:
        print("No examples found in file.")
        return

    if args.index < 0 or args.index >= len(examples):
        print(f"Invalid index {args.index}. File contains {len(examples)} example(s).")
        return

    example = examples[args.index]

    history_str = format_history(example["history"])
    target_str = format_target_item(example["target_item"])
    prompt = build_prompt(history_str, target_str)

    print("=" * 80)
    print(f"Prompt built from example index {args.index}")
    print("=" * 80)
    print(prompt)
    print()


if __name__ == "__main__":
    main()