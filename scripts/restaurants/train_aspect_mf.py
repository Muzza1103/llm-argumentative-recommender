import argparse
import json
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD


def normalize_score(
    rating: float,
    min_rating: float = 1.0,
    max_rating: float = 5.0,
) -> float:
    return (rating - min_rating) / (max_rating - min_rating)


def load_mf_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"user_id", "aspect", "rating"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=["user_id", "aspect", "rating"])
    df["user_id"] = df["user_id"].astype(str)
    df["aspect"] = df["aspect"].astype(str)
    df["rating"] = df["rating"].astype(float)

    return df


def train_svd(
    df: pd.DataFrame,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
    random_state: int,
) -> SVD:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "aspect", "rating"]], reader)

    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )

    model.fit(trainset)

    return model


def build_predictions(
    df: pd.DataFrame,
    model: SVD,
) -> list[dict]:
    users = sorted(df["user_id"].unique())
    aspects = sorted(df["aspect"].unique())

    predictions = []

    for user_id in users:
        for aspect in aspects:
            prediction = model.predict(user_id, aspect)
            raw_score = float(prediction.est)
            normalized_score = normalize_score(raw_score)

            predictions.append(
                {
                    "user_id": user_id,
                    "aspect": aspect,
                    "predicted_rating": raw_score,
                    "score": normalized_score,
                }
            )

    return predictions


def save_json(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def build_summary(
    df: pd.DataFrame,
    predictions: list[dict],
    input_file: str,
    output_file: str,
    n_factors: int,
    n_epochs: int,
    lr_all: float,
    reg_all: float,
) -> dict:
    return {
        "input_file": input_file,
        "output_file": output_file,
        "num_training_rows": len(df),
        "num_unique_users": int(df["user_id"].nunique()),
        "num_unique_aspects": int(df["aspect"].nunique()),
        "num_predictions": len(predictions),
        "rating_min": float(df["rating"].min()),
        "rating_max": float(df["rating"].max()),
        "rating_mean": float(df["rating"].mean()),
        "n_factors": n_factors,
        "n_epochs": n_epochs,
        "lr_all": lr_all,
        "reg_all": reg_all,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train an aspect-based Matrix Factorization model using SVD."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the aspect MF dataset CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output aspect MF predictions JSON file.",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=20,
        help="Number of latent factors for SVD.",
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
        help="Learning rate for all parameters.",
    )
    parser.add_argument(
        "--reg-all",
        type=float,
        default=0.02,
        help="Regularization term for all parameters.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    df = load_mf_dataset(input_path)

    model = train_svd(
        df=df,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        random_state=args.random_state,
    )

    predictions = build_predictions(
        df=df,
        model=model,
    )

    save_json(predictions, output_path)

    summary = build_summary(
        df=df,
        predictions=predictions,
        input_file=str(input_path),
        output_file=str(output_path),
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Training rows:     {len(df)}")
    print(f"Unique users:      {df['user_id'].nunique()}")
    print(f"Unique aspects:    {df['aspect'].nunique()}")
    print(f"Predictions:       {len(predictions)}")
    print(f"Output:            {output_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()