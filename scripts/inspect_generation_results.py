import argparse
import json
from pathlib import Path


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def filter_records(records: list[dict], only: str) -> list[dict]:
    if only == "all":
        return records

    filtered = []
    expected_validity = only == "valid"

    for record in records:
        is_valid = record.get("validation", {}).get("is_valid")
        if is_valid == expected_validity:
            filtered.append(record)

    return filtered


def print_record(record: dict, show_prompt: bool, show_raw: bool):
    print("=" * 100)
    print(f"INDEX:      {record.get('index')}")
    print(f"USER_ID:    {record.get('user_id')}")
    print(f"TARGET:     {record.get('target_name')}")
    print(f"IS_VALID:   {record.get('validation', {}).get('is_valid')}")

    errors = record.get("validation", {}).get("errors", [])
    print("ERRORS:")
    if not errors:
        print("- none")
    else:
        for error in errors:
            if isinstance(error, dict):
                print(f"- {error.get('code')}: {error.get('message')}")
            else:
                print(f"- {error}")

    if show_prompt and "prompt" in record:
        print("\nPROMPT:")
        print(record["prompt"])

    print("\nPARSED JSON:")
    parsed_json = record.get("parsed_json")
    if parsed_json is None:
        print("None")
    else:
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

    if show_raw:
        print("\nRAW OUTPUT:")
        print(record.get("raw_output"))

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect JSONL results produced by batch argument generation."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the generation results JSONL file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of records to display.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "valid", "invalid"],
        default="all",
        help="Filter displayed records by validation status.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the full prompt when available.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Display the raw model output.",
    )
    args = parser.parse_args()

    records = load_jsonl(Path(args.file))
    filtered_records = filter_records(records, args.only)

    print("=" * 100)
    print(f"FILE:            {args.file}")
    print(f"TOTAL RECORDS:   {len(records)}")
    print(f"FILTER:          {args.only}")
    print(f"MATCHED RECORDS: {len(filtered_records)}")
    print(f"DISPLAYING:      {min(args.n, len(filtered_records))}")
    print("=" * 100)
    print()

    for record in filtered_records[: args.n]:
        print_record(
            record,
            show_prompt=args.show_prompt,
            show_raw=args.show_raw,
        )


if __name__ == "__main__":
    main()