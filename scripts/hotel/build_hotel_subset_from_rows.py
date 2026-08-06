import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict) and "rows" in data:
            rows = data["rows"]
            if isinstance(rows, list):
                return rows

        if isinstance(data, dict):
            return [data]

        raise ValueError("Unsupported JSON structure.")

    except json.JSONDecodeError:
        rows = []

        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON/JSONL at line {line_number}: {exc}"
                    ) from exc

        return rows


def save_jsonl(records: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_review_text(row: dict[str, Any]) -> str:
    parts = []

    headline = row.get("headline")
    pros = row.get("pros")
    cons = row.get("cons")

    if headline:
        parts.append(str(headline))

    if pros:
        parts.append(f"Pros: {pros}")

    if cons:
        parts.append(f"Cons: {cons}")

    return "\n".join(parts).strip()


def to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_attributes(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "city": row.get("city"),
        "country_code": row.get("country_code"),
        "star_rating": row.get("star_rating"),
        "hotel_type": row.get("hotel_type_name"),
        "chain": row.get("chain_name"),
        "child_allowed": row.get("child_allowed"),
        "pets_allowed": row.get("pets_allowed"),
        "parking": row.get("parking"),
        "checkin_time": row.get("checkin_time"),
        "checkout_time": row.get("checkout_time"),
    }


def build_categories(row: dict[str, Any]) -> list[str]:
    categories = ["Hotels"]

    hotel_type = row.get("hotel_type_name")
    if hotel_type:
        categories.append(str(hotel_type))

    star_rating = row.get("star_rating")
    if star_rating is not None:
        categories.append(f"{star_rating}_star_hotel")

    return categories


def build_item(row: dict[str, Any], is_target: bool = False) -> dict[str, Any]:
    item = {
        "business_id": str(row.get("liteapi_id")),
        "name": str(row.get("hotel_name") or row.get("liteapi_id")),
        "categories": build_categories(row),
        "attributes": build_attributes(row),
    }

    rating = to_float(row.get("rating"))

    if is_target:
        item.update(
            {
                "global_stars": to_float(row.get("hotel_global_rating")),
                "review_count": to_int(row.get("review_count")),
                "user_target_stars": rating,
                "target_review_text": build_review_text(row),
            }
        )
    else:
        item.update(
            {
                "user_stars": rating,
                "review_text": build_review_text(row),
            }
        )

    return item


def main():
    parser = argparse.ArgumentParser(
        description="Convert exported hotel BigQuery rows to internal JSONL recommendation format."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)

    by_user = defaultdict(list)

    for row in rows:
        user_id = row.get("pseudo_user_id")
        liteapi_id = row.get("liteapi_id")
        rating = row.get("rating")

        if not user_id or not liteapi_id or rating is None:
            continue

        by_user[str(user_id)].append(row)

    examples = []

    for user_id, user_rows in by_user.items():
        user_rows.sort(key=lambda row: row.get("review_date") or "")

        if len(user_rows) < args.history_size + 1:
            continue

        selected_rows = user_rows[-(args.history_size + 1):]
        history_rows = selected_rows[:-1]
        target_row = selected_rows[-1]

        example = {
            "user_id": user_id,
            "history": [
                build_item(row, is_target=False)
                for row in history_rows
            ],
            "target_item": build_item(target_row, is_target=True),
        }

        examples.append(example)

        if args.max_users is not None and len(examples) >= args.max_users:
            break

    save_jsonl(examples, output_path)

    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "num_rows_loaded": len(rows),
        "num_users_found": len(by_user),
        "num_examples_saved": len(examples),
        "history_size": args.history_size,
        "max_users": args.max_users,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded rows: {len(rows)}")
    print(f"Users found: {len(by_user)}")
    print(f"Examples saved: {len(examples)}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()