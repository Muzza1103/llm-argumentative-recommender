import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD


def load_mf_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    expected_columns = {"user_id", "business_id", "rating"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in MF dataset CSV: {sorted(missing)}"
        )

    return df[["user_id", "business_id", "rating"]]


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def normalize_rating(rating: float) -> float:
    return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))


def train_svd_model(
    df: pd.DataFrame,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    random_state: int,
):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[["user_id", "business_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    model.fit(trainset)

    return model


def build_target_predictions(
    model,
    source_examples: list[dict],
) -> list[dict]:
    predictions = []

    for example in source_examples:
        user_id = example.get("user_id")
        target_item = example.get("target_item", {})
        business_id = target_item.get("business_id")

        if not isinstance(user_id, str) or not isinstance(business_id, str):
            continue

        pred = model.predict(user_id, business_id)
        predicted_rating = float(pred.est)
        normalized_score = normalize_rating(predicted_rating)

        predictions.append(
            {
                "user_id": user_id,
                "business_id": business_id,
                "predicted_rating": predicted_rating,
                "score": normalized_score,
            }
        )

    return predictions


def build_summary(
    mf_dataset_file: str,
    source_dataset_file: str,
    output_file: str,
    num_train_rows: int,
    num_predictions: int,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    random_state: int,
) -> dict:
    return {
        "mf_dataset_file": mf_dataset_file,
        "source_dataset_file": source_dataset_file,
        "output_file": output_file,
        "num_train_rows": num_train_rows,
        "num_predictions": num_predictions,
        "model": "surprise.SVD",
        "n_factors": n_factors,
        "n_epochs": n_epochs,
        "lr_all": lr_all,
        "reg_all": reg_all,
        "random_state": random_state,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a matrix factorization model and generate target-item predictions."
    )
    parser.add_argument(
        "--mf-data",
        type=str,
        required=True,
        help="Path to the MF training CSV file.",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        required=True,
        help="Path to the source JSONL dataset containing target items to score.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON file containing MF predictions.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=20,
        help="Number of latent factors.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr-all",
        type=float,
        default=0.005,
        help="Global learning rate.",
    )
    parser.add_argument(
        "--reg-all",
        type=float,
        default=0.02,
        help="Global regularization coefficient.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    mf_data_path = Path(args.mf_data)
    source_dataset_path = Path(args.source_dataset)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    df = load_mf_csv(mf_data_path)
    source_examples = load_jsonl(source_dataset_path)

    model = train_svd_model(
        df=df,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        random_state=args.random_state,
    )

    predictions = build_target_predictions(
        model=model,
        source_examples=source_examples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    summary = build_summary(
        mf_dataset_file=str(mf_data_path),
        source_dataset_file=str(source_dataset_path),
        output_file=str(output_path),
        num_train_rows=len(df),
        num_predictions=len(predictions),
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        random_state=args.random_state,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"MF training data: {mf_data_path}")
    print(f"Source dataset:    {source_dataset_path}")
    print(f"Predictions:       {output_path}")
    print(f"Summary:           {summary_path}")
    print(f"Train rows:        {len(df)}")
    print(f"Predictions:       {len(predictions)}")


if __name__ == "__main__":
    main()