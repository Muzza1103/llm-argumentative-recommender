import argparse
import json
import random
from pathlib import Path


SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_candidate_pool(dataset: list[dict]) -> list[dict]:
    candidates = []

    for example in dataset:
        target = example.get("target_item")

        if not isinstance(target, dict):
            continue

        business_id = target.get("business_id")

        if not business_id:
            continue

        candidates.append(target)

    return candidates


def sample_negative_items(
    candidate_pool: list[dict],
    forbidden_business_ids: set[str],
    num_negatives: int,
) -> list[dict]:
    valid_candidates = [
        item
        for item in candidate_pool
        if item.get("business_id") not in forbidden_business_ids
    ]

    if len(valid_candidates) < num_negatives:
        return valid_candidates

    return random.sample(valid_candidates, num_negatives)


def build_ranking_examples(
    dataset: list[dict],
    num_negatives: int,
) -> list[dict]:
    ranking_examples = []

    candidate_pool = build_candidate_pool(dataset)

    for idx, example in enumerate(dataset):
        user_id = example.get("user_id")
        history = example.get("history", [])
        positive_target = example.get("target_item")

        if not isinstance(positive_target, dict):
            continue

        positive_business_id = positive_target.get("business_id")

        if not positive_business_id:
            continue

        forbidden_business_ids = {
            positive_business_id,
        }

        for item in history:
            business_id = item.get("business_id")

            if business_id:
                forbidden_business_ids.add(business_id)

        ranking_group_id = f"group_{idx}"

        positive_example = {
            "ranking_group_id": ranking_group_id,
            "candidate_label": 1,
            "user_id": user_id,
            "history": history,
            "target_item": positive_target,
        }

        ranking_examples.append(positive_example)

        negative_targets = sample_negative_items(
            candidate_pool=candidate_pool,
            forbidden_business_ids=forbidden_business_ids,
            num_negatives=num_negatives,
        )

        for negative_target in negative_targets:
            negative_example = {
                "ranking_group_id": ranking_group_id,
                "candidate_label": 0,
                "user_id": user_id,
                "history": history,
                "target_item": negative_target,
            }

            ranking_examples.append(negative_example)

    return ranking_examples


def build_summary(
    ranking_examples: list[dict],
    input_file: str,
    output_file: str,
    num_negatives: int,
) -> dict:
    positive_count = 0
    negative_count = 0
    ranking_groups = set()

    for example in ranking_examples:
        label = example.get("candidate_label")

        if label == 1:
            positive_count += 1
        elif label == 0:
            negative_count += 1

        group_id = example.get("ranking_group_id")

        if isinstance(group_id, str):
            ranking_groups.add(group_id)

    return {
        "input_file": input_file,
        "output_file": output_file,
        "num_examples": len(ranking_examples),
        "num_groups": len(ranking_groups),
        "num_positive_candidates": positive_count,
        "num_negative_candidates": negative_count,
        "num_negatives_per_group": num_negatives,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build ranking candidates with sampled negatives."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the dataset JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output ranking dataset JSONL file.",
    )

    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Number of negative candidates per user.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optional number of users/examples to keep from the input dataset.",
    )

    args = parser.parse_args()

    random.seed(SEED)

    input_path = Path(args.input)
    output_path = Path(args.output)

    summary_path = output_path.with_name(
        f"{output_path.stem}_summary.json"
    )

    dataset = load_jsonl(input_path)

    if args.num_examples is not None:
        dataset = dataset[:args.num_examples]

    ranking_examples = build_ranking_examples(
        dataset=dataset,
        num_negatives=args.num_negatives,
    )

    save_jsonl(ranking_examples, output_path)

    summary = build_summary(
        ranking_examples=ranking_examples,
        input_file=str(input_path),
        output_file=str(output_path),
        num_negatives=args.num_negatives,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded examples:      {len(dataset)}")
    print(f"Built ranking rows:   {len(ranking_examples)}")
    print(f"Saved dataset:        {output_path}")
    print(f"Saved summary:        {summary_path}")


if __name__ == "__main__":
    main()