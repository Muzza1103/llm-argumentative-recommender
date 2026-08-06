import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def normalize_rating(rating: float | None) -> float | None:
    if rating is None:
        return None

    return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-only direct score predictions."
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-summary", type=str, required=True)

    args = parser.parse_args()

    predictions = load_jsonl(Path(args.input))
    dataset = load_jsonl(Path(args.dataset))

    dataset_by_index = {
        index: example
        for index, example in enumerate(dataset)
    }

    rows = []

    for prediction in predictions:
        if not prediction.get("is_valid", False):
            continue

        index = prediction.get("index")
        example = dataset_by_index.get(index)

        if example is None:
            continue

        target_item = example.get("target_item", {})
        target_rating = target_item.get("user_target_stars")
        gold_score = normalize_rating(target_rating)

        if gold_score is None:
            continue

        predicted_score = prediction.get("score")

        if not isinstance(predicted_score, (int, float)):
            continue

        predicted_score = float(predicted_score)
        absolute_error = abs(predicted_score - gold_score)
        squared_error = (predicted_score - gold_score) ** 2

        rows.append(
            {
                "index": index,
                "user_id": prediction.get("user_id"),
                "target_name": prediction.get("target_name"),
                "target_rating": target_rating,
                "normalized_target_rating": gold_score,
                "predicted_score": predicted_score,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                "reason": prediction.get("reason"),
            }
        )

    absolute_errors = [row["absolute_error"] for row in rows]
    squared_errors = [row["squared_error"] for row in rows]

    summary = {
        "input_file": args.input,
        "dataset_file": args.dataset,
        "num_examples": len(rows),
        "mae": mean(absolute_errors) if absolute_errors else None,
        "mse": mean(squared_errors) if squared_errors else None,
        "rmse": (mean(squared_errors) ** 0.5) if squared_errors else None,
        "mean_predicted_score": (
            mean([row["predicted_score"] for row in rows])
            if rows
            else None
        ),
        "mean_normalized_target_rating": (
            mean([row["normalized_target_rating"] for row in rows])
            if rows
            else None
        ),
    }

    save_csv(rows, Path(args.output_csv))
    save_json(summary, Path(args.output_summary))

    print(f"\nExamples evaluated: {len(rows)}")
    print(f"MAE:  {summary['mae']}")
    print(f"MSE:  {summary['mse']}")
    print(f"RMSE: {summary['rmse']}")
    print(f"CSV:  {args.output_csv}")
    print(f"JSON: {args.output_summary}")


if __name__ == "__main__":
    main()