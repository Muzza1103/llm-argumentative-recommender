import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path


SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_history_items(dataset: list[dict]) -> Counter:
    counts = Counter()

    for example in dataset:
        for item in example.get("history", []):
            business_id = item.get("business_id")
            if business_id:
                counts[business_id] += 1

    return counts


def build_candidate_pool(
    dataset: list[dict],
    source: str,
    history_item_counts: Counter,
    min_candidate_occurrences: int,
) -> list[dict]:
    candidates_by_id = {}

    for example in dataset:
        if source == "target":
            target = example.get("target_item")
            if isinstance(target, dict):
                business_id = target.get("business_id")
                if (
                    business_id
                    and history_item_counts.get(business_id, 0) >= min_candidate_occurrences
                ):
                    candidates_by_id[business_id] = target

        elif source == "history":
            for item in example.get("history", []):
                if isinstance(item, dict):
                    business_id = item.get("business_id")
                    if (
                        business_id
                        and history_item_counts.get(business_id, 0) >= min_candidate_occurrences
                    ):
                        candidates_by_id[business_id] = item

        else:
            raise ValueError(f"Unknown candidate source: {source}")

    return list(candidates_by_id.values())


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
        return []

    return random.sample(valid_candidates, num_negatives)


def build_ranking_examples(
    dataset: list[dict],
    num_examples: int | None,
    num_negatives: int,
    candidate_source: str,
    min_candidate_occurrences: int,
    require_positive_seen: bool,
) -> tuple[list[dict], dict]:
    ranking_examples = []

    history_item_counts = count_history_items(dataset)

    candidate_pool = build_candidate_pool(
        dataset=dataset,
        source=candidate_source,
        history_item_counts=history_item_counts,
        min_candidate_occurrences=min_candidate_occurrences,
    )

    skipped_positive_unseen = 0
    skipped_not_enough_negatives = 0
    candidate_examples_considered = 0

    for example in dataset:
        user_id = example.get("user_id")
        history = example.get("history", [])
        positive_target = example.get("target_item")

        if not isinstance(positive_target, dict):
            continue

        positive_business_id = positive_target.get("business_id")
        if not positive_business_id:
            continue

        candidate_examples_considered += 1
        positive_seen_count = history_item_counts.get(positive_business_id, 0)

        if require_positive_seen and positive_seen_count < min_candidate_occurrences:
            skipped_positive_unseen += 1
            continue

        forbidden_business_ids = {positive_business_id}

        for item in history:
            business_id = item.get("business_id")
            if business_id:
                forbidden_business_ids.add(business_id)

        negative_targets = sample_negative_items(
            candidate_pool=candidate_pool,
            forbidden_business_ids=forbidden_business_ids,
            num_negatives=num_negatives,
        )

        if len(negative_targets) < num_negatives:
            skipped_not_enough_negatives += 1
            continue

        ranking_group_id = f"group_{len(ranking_examples) // (num_negatives + 1)}"

        ranking_examples.append(
            {
                "ranking_group_id": ranking_group_id,
                "candidate_label": 1,
                "positive_seen_count": positive_seen_count,
                "user_id": user_id,
                "history": history,
                "target_item": positive_target,
            }
        )

        for negative_target in negative_targets:
            negative_business_id = negative_target.get("business_id")

            ranking_examples.append(
                {
                    "ranking_group_id": ranking_group_id,
                    "candidate_label": 0,
                    "negative_seen_count": history_item_counts.get(
                        negative_business_id,
                        0,
                    ),
                    "user_id": user_id,
                    "history": history,
                    "target_item": negative_target,
                }
            )

        if num_examples is not None:
            current_groups = len(ranking_examples) // (num_negatives + 1)
            if current_groups >= num_examples:
                break

    metadata = {
        "candidate_examples_considered": candidate_examples_considered,
        "candidate_pool_size": len(candidate_pool),
        "skipped_positive_unseen": skipped_positive_unseen,
        "skipped_not_enough_negatives": skipped_not_enough_negatives,
    }

    return ranking_examples, metadata


def build_summary(
    ranking_examples: list[dict],
    input_file: str,
    output_file: str,
    num_negatives: int,
    candidate_source: str,
    min_candidate_occurrences: int,
    require_positive_seen: bool,
    metadata: dict,
    runtime_seconds: float,
) -> dict:
    positive_count = 0
    negative_count = 0
    ranking_groups = set()
    positive_seen_counts = []
    negative_seen_counts = []

    for example in ranking_examples:
        label = example.get("candidate_label")

        if label == 1:
            positive_count += 1
            positive_seen_counts.append(example.get("positive_seen_count", 0))
        elif label == 0:
            negative_count += 1
            negative_seen_counts.append(example.get("negative_seen_count", 0))

        group_id = example.get("ranking_group_id")
        if isinstance(group_id, str):
            ranking_groups.add(group_id)

    return {
        "input_file": input_file,
        "output_file": output_file,
        "candidate_source": candidate_source,
        "min_candidate_occurrences": min_candidate_occurrences,
        "require_positive_seen": require_positive_seen,
        "num_examples": len(ranking_examples),
        "num_groups": len(ranking_groups),
        "num_positive_candidates": positive_count,
        "num_negative_candidates": negative_count,
        "num_negatives_per_group": num_negatives,
        "positive_seen_count_min": min(positive_seen_counts) if positive_seen_counts else None,
        "positive_seen_count_mean": (
            sum(positive_seen_counts) / len(positive_seen_counts)
            if positive_seen_counts else None
        ),
        "negative_seen_count_min": min(negative_seen_counts) if negative_seen_counts else None,
        "negative_seen_count_mean": (
            sum(negative_seen_counts) / len(negative_seen_counts)
            if negative_seen_counts else None
        ),
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60,
        **metadata,
    }


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Build ranking candidates with sampled negatives."
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

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
        help="Optional number of ranking groups to build after filtering.",
    )

    parser.add_argument(
        "--candidate-source",
        choices=["target", "history"],
        default="target",
        help="Source used to sample negative candidates.",
    )

    parser.add_argument(
        "--min-candidate-occurrences",
        type=int,
        default=1,
        help=(
            "Minimum number of appearances in the input histories required "
            "for positive and negative ranking candidates."
        ),
    )

    parser.add_argument(
        "--require-positive-seen",
        action="store_true",
        help=(
            "Keep only ranking groups whose positive target appears at least "
            "--min-candidate-occurrences times in the input histories."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    dataset = load_jsonl(input_path)

    ranking_examples, metadata = build_ranking_examples(
        dataset=dataset,
        num_examples=args.num_examples,
        num_negatives=args.num_negatives,
        candidate_source=args.candidate_source,
        min_candidate_occurrences=args.min_candidate_occurrences,
        require_positive_seen=args.require_positive_seen,
    )

    save_jsonl(ranking_examples, output_path)

    runtime_seconds = time.perf_counter() - start_time

    summary = build_summary(
        ranking_examples=ranking_examples,
        input_file=str(input_path),
        output_file=str(output_path),
        num_negatives=args.num_negatives,
        candidate_source=args.candidate_source,
        min_candidate_occurrences=args.min_candidate_occurrences,
        require_positive_seen=args.require_positive_seen,
        metadata=metadata,
        runtime_seconds=runtime_seconds,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loaded examples:              {len(dataset)}")
    print(f"Candidate source:             {args.candidate_source}")
    print(f"Min candidate occurrences:    {args.min_candidate_occurrences}")
    print(f"Require positive seen:        {args.require_positive_seen}")
    print(f"Candidate pool size:          {metadata['candidate_pool_size']}")
    print(f"Built ranking rows:           {len(ranking_examples)}")
    print(f"Built ranking groups:         {summary['num_groups']}")
    print(f"Skipped positive unseen:      {metadata['skipped_positive_unseen']}")
    print(f"Skipped not enough negatives: {metadata['skipped_not_enough_negatives']}")
    print(f"Saved dataset:                {output_path}")
    print(f"Saved summary:                {summary_path}")


if __name__ == "__main__":
    main()
