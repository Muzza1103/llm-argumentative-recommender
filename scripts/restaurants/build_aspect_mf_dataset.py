import argparse
import csv
import json
from pathlib import Path


ATTRIBUTE_LABELS = {
    "RestaurantsPriceRange2": "price",
    "RestaurantsTakeOut": "takeout",
    "RestaurantsDelivery": "delivery",
    "OutdoorSeating": "outdoor_seating",
    "RestaurantsAttire": "attire",
    "Alcohol": "alcohol",
    "NoiseLevel": "noise",
    "RestaurantsGoodForGroups": "good_for_groups",
    "GoodForKids": "good_for_kids",
    "RestaurantsReservations": "reservations",
}

IMPORTANT_ATTRIBUTES = list(ATTRIBUTE_LABELS.keys())


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def clean_attribute_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        value = value.replace("u'", "'")
        value = value.strip("'")

        if value.lower() == "none":
            return None

    return value


def normalize_category(category: str) -> str:
    normalized = category.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("(", "")
    normalized = normalized.replace(")", "")
    normalized = normalized.replace(",", "")
    normalized = "_".join(part for part in normalized.split("_") if part)

    return f"category_{normalized}"


def extract_category_aspects(item: dict) -> set[str]:
    aspects = set()

    for category in item.get("categories", []):
        if not isinstance(category, str):
            continue

        category = category.strip()
        if not category:
            continue

        aspects.add(normalize_category(category))

    return aspects


def extract_attribute_aspects(item: dict) -> set[str]:
    aspects = set()
    attributes = item.get("attributes", {})

    if not isinstance(attributes, dict):
        return aspects

    for raw_key in IMPORTANT_ATTRIBUTES:
        if raw_key not in attributes:
            continue

        value = clean_attribute_value(attributes[raw_key])
        if value is None:
            continue

        label = ATTRIBUTE_LABELS[raw_key]

        if isinstance(value, str):
            value_lower = value.lower()

            if value_lower == "true":
                aspects.add(label)
            elif value_lower == "false":
                continue
            else:
                normalized_value = value_lower.replace(" ", "_")
                aspects.add(f"{label}_{normalized_value}")
        else:
            aspects.add(f"{label}_{value}")

    return aspects


def extract_review_aspects(item: dict) -> set[str]:
    aspects = set()

    for aspect_record in item.get("review_aspects", []):
        if not isinstance(aspect_record, dict):
            continue

        name = aspect_record.get("name")
        if not isinstance(name, str):
            continue

        name = name.strip().lower()
        if not name:
            continue

        aspects.add(name)

    return aspects


def build_rows(examples: list[dict]) -> list[dict]:
    rows = []

    for example in examples:
        user_id = example.get("user_id")
        if not user_id:
            continue

        for item in example.get("history", []):
            rating = item.get("user_stars")
            if rating is None:
                continue

            aspects = set()
            aspects.update(extract_category_aspects(item))
            aspects.update(extract_attribute_aspects(item))
            aspects.update(extract_review_aspects(item))

            for aspect in sorted(aspects):
                rows.append(
                    {
                        "user_id": user_id,
                        "aspect": aspect,
                        "rating": float(rating),
                        "business_id": item.get("business_id"),
                        "item_name": item.get("name"),
                    }
                )

    return rows


def save_csv(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["user_id", "aspect", "rating", "business_id", "item_name"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict], input_file: str, output_file: str) -> dict:
    unique_users = set()
    unique_aspects = set()
    unique_items = set()

    for row in rows:
        unique_users.add(row["user_id"])
        unique_aspects.add(row["aspect"])
        unique_items.add(row["business_id"])

    return {
        "input_file": input_file,
        "output_file": output_file,
        "num_rows": len(rows),
        "num_unique_users": len(unique_users),
        "num_unique_aspects": len(unique_aspects),
        "num_unique_items": len(unique_items),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a user-aspect dataset for aspect-based MF."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the aspect-enriched dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    examples = load_jsonl(input_path)
    rows = build_rows(examples)

    save_csv(rows, output_path)

    summary = build_summary(
        rows=rows,
        input_file=str(input_path),
        output_file=str(output_path),
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded examples: {len(examples)}")
    print(f"Built rows:      {len(rows)}")
    print(f"Saved CSV:       {output_path}")
    print(f"Saved summary:   {summary_path}")


if __name__ == "__main__":
    main()