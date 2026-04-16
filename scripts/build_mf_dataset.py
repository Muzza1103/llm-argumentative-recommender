import argparse
import csv
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


def build_mf_rows(
    examples: list[dict],
    include_target: bool = False,
    deduplicate: bool = True,
) -> list[dict]:
    """
    Build user-item-rating rows for MF training.
    """
    rows = []

    for example in examples:
        user_id = example.get("user_id")
        if not user_id:
            continue

        history = example.get("history", [])
        for item in history:
            business_id = item.get("business_id")
            rating = item.get("user_stars")

            if not business_id:
                continue
            if not isinstance(rating, (int, float)):
                continue

            rows.append(
                {
                    "user_id": user_id,
                    "business_id": business_id,
                    "rating": float(rating),
                }
            )

        if include_target:
            target_item = example.get("target_item", {})
            business_id = target_item.get("business_id")
            rating = target_item.get("user_target_stars")

            if business_id and isinstance(rating, (int, float)):
                rows.append(
                    {
                        "user_id": user_id,
                        "business_id": business_id,
                        "rating": float(rating),
                    }
                )

    if deduplicate:
        unique_rows = []
        seen = set()

        for row in rows:
            key = (row["user_id"], row["business_id"], row["rating"])
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)

        rows = unique_rows

    return rows


def save_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "business_id", "rating"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    rows: list[dict],
    input_file: str,
    output_file: str,
    include_target: bool,
    deduplicate: bool,
) -> dict:
    users = {row["user_id"] for row in rows}
    items = {row["business_id"] for row in rows}

    return {
        "input_file": input_file,
        "output_file": output_file,
        "include_target": include_target,
        "deduplicate": deduplicate,
        "num_rows": len(rows),
        "num_users": len(users),
        "num_items": len(items),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a user-item-rating dataset for matrix factorization."
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
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Also include the target item rating when available.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Keep duplicate user-item-rating rows.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    examples = load_jsonl(input_path)

    rows = build_mf_rows(
        examples=examples,
        include_target=args.include_target,
        deduplicate=not args.no_deduplicate,
    )

    save_csv(rows, output_path)

    summary = build_summary(
        rows=rows,
        input_file=str(input_path),
        output_file=str(output_path),
        include_target=args.include_target,
        deduplicate=not args.no_deduplicate,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Source dataset: {input_path}")
    print(f"MF dataset CSV: {output_path}")
    print(f"Summary:        {summary_path}")
    print(f"Rows:           {summary['num_rows']}")
    print(f"Users:          {summary['num_users']}")
    print(f"Items:          {summary['num_items']}")


if __name__ == "__main__":
    main()