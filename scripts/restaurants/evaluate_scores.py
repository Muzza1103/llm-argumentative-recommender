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


def get_mean_argument_score(record: dict, key: str) -> float | None:
    values = []

    for arg in record.get("scored_arguments", []):
        value = arg.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))

    return mean(values) if values else None


def safe_abs_error(pred: float, gold: float) -> float:
    return abs(pred - gold)


def safe_squared_error(pred: float, gold: float) -> float:
    return (pred - gold) ** 2


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
        description="Evaluate predicted recommendation scores against normalized target ratings."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to scored arguments JSONL, usually output of score_batch.py.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to source dataset JSONL containing target user ratings.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save per-example evaluation CSV.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        required=True,
        help="Path to save global metrics JSON.",
    )
    parser.add_argument(
        "--prediction-source",
        type=str,
        default="mean_combined_score",
        choices=[
            "mean_combined_score",
            "mean_llm_score",
            "mean_mf_score",
        ],
        help="Which score to compare against the normalized target rating.",
    )
    args = parser.parse_args()

    scored_records = load_jsonl(Path(args.input))
    dataset_records = load_jsonl(Path(args.dataset))

    dataset_by_index = {
        index: example
        for index, example in enumerate(dataset_records)
    }

    rows = []

    for record in scored_records:
        index = record.get("index")
        example = dataset_by_index.get(index)

        if example is None:
            continue

        target_item = example.get("target_item", {})
        target_rating = target_item.get("user_target_stars")
        gold_score = normalize_rating(target_rating)

        if gold_score is None:
            continue

        mean_llm_score = get_mean_argument_score(record, "llm_score")
        mean_mf_score = get_mean_argument_score(record, "mf_score")
        mean_combined_score = get_mean_argument_score(record, "combined_score")

        prediction = {
            "mean_llm_score": mean_llm_score,
            "mean_mf_score": mean_mf_score,
            "mean_combined_score": mean_combined_score,
        }[args.prediction_source]

        if prediction is None:
            continue

        abs_error = safe_abs_error(prediction, gold_score)
        squared_error = safe_squared_error(prediction, gold_score)

        rows.append(
            {
                "index": index,
                "user_id": record.get("user_id"),
                "target_name": record.get("target_name"),
                "target_rating": target_rating,
                "normalized_target_rating": gold_score,
                "prediction_source": args.prediction_source,
                "predicted_score": prediction,
                "absolute_error": abs_error,
                "squared_error": squared_error,
                "mean_llm_score": mean_llm_score,
                "mean_mf_score": mean_mf_score,
                "mean_combined_score": mean_combined_score,
                "num_arguments": len(record.get("scored_arguments", [])),
            }
        )

    absolute_errors = [row["absolute_error"] for row in rows]
    squared_errors = [row["squared_error"] for row in rows]

    summary = {
        "input_file": args.input,
        "dataset_file": args.dataset,
        "prediction_source": args.prediction_source,
        "num_examples": len(rows),
        "mae": mean(absolute_errors) if absolute_errors else None,
        "mse": mean(squared_errors) if squared_errors else None,
        "rmse": (mean(squared_errors) ** 0.5) if squared_errors else None,
        "min_absolute_error": min(absolute_errors) if absolute_errors else None,
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
        "mean_predicted_score": mean([row["predicted_score"] for row in rows]) if rows else None,
        "mean_normalized_target_rating": mean([row["normalized_target_rating"] for row in rows]) if rows else None,
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