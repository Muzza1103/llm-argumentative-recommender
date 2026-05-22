import argparse
import json
import math
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON input must contain a list of records.")

        return data

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_score(
    record: dict,
    score_source: str,
    score_key: str,
) -> float | None:
    if score_source == "direct":
        score = record.get(score_key)

    elif score_source == "dfquad":
        dfquad = record.get("dfquad", {})
        score = dfquad.get(score_key)

    else:
        raise ValueError(f"Unknown score_source: {score_source}")

    if isinstance(score, (int, float)):
        return float(score)

    return None


def dcg_at_k(labels: list[int], k: int) -> float:
    score = 0.0

    for i, label in enumerate(labels[:k]):
        if label <= 0:
            continue

        rank = i + 1
        score += label / math.log2(rank + 1)

    return score


def ndcg_at_k(labels: list[int], k: int) -> float:
    dcg = dcg_at_k(labels, k)
    ideal_labels = sorted(labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_labels, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def attach_ranking_metadata(
    records: list[dict],
    dataset_records: list[dict],
) -> list[dict]:
    dataset_by_index = {
        index: example
        for index, example in enumerate(dataset_records)
    }

    dataset_by_pair = {}

    for example in dataset_records:
        user_id = example.get("user_id")
        target_item = example.get("target_item", {})
        business_id = target_item.get("business_id")

        if isinstance(user_id, str) and isinstance(business_id, str):
            dataset_by_pair[(user_id, business_id)] = example

    enriched_records = []

    for record in records:
        enriched = dict(record)

        ranking_group_id = enriched.get("ranking_group_id")
        candidate_label = enriched.get("candidate_label")

        if ranking_group_id is None or candidate_label is None:
            example = None

            dataset_index = enriched.get("index")
            if isinstance(dataset_index, int):
                example = dataset_by_index.get(dataset_index)

            if example is None:
                user_id = enriched.get("user_id")
                business_id = enriched.get("business_id")

                if isinstance(user_id, str) and isinstance(business_id, str):
                    example = dataset_by_pair.get((user_id, business_id))

            if example is not None:
                enriched["ranking_group_id"] = example.get("ranking_group_id")
                enriched["candidate_label"] = example.get("candidate_label")

                target_item = example.get("target_item", {})
                enriched.setdefault("target_name", target_item.get("name"))

        enriched_records.append(enriched)

    return enriched_records


def evaluate_group(
    records: list[dict],
    ks: list[int],
    score_source: str,
    score_key: str,
) -> dict:
    scored = []

    for record in records:
        score = get_score(
            record=record,
            score_source=score_source,
            score_key=score_key,
        )

        if score is None:
            continue

        label = int(record.get("candidate_label", 0))

        scored.append(
            {
                "score": score,
                "label": label,
                "target_name": record.get("target_name"),
                "index": record.get("index"),
            }
        )

    if not scored:
        return {}

    ranked = sorted(
        scored,
        key=lambda x: x["score"],
        reverse=True,
    )

    labels = [item["label"] for item in ranked]

    positive_rank = None
    for i, label in enumerate(labels, start=1):
        if label == 1:
            positive_rank = i
            break

    result = {
        "num_candidates": len(ranked),
        "positive_rank": positive_rank,
        "mrr": 1.0 / positive_rank if positive_rank is not None else 0.0,
    }

    for k in ks:
        result[f"hitrate@{k}"] = (
            1.0
            if positive_rank is not None and positive_rank <= k
            else 0.0
        )
        result[f"ndcg@{k}"] = ndcg_at_k(labels, k)

    return result


def mean(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ranking metrics from scored candidate records."
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Original ranking candidates JSONL file used to recover ranking metadata.",
    )
    parser.add_argument("--output-summary", type=str, required=True)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Ranking cutoffs, e.g. --k 1 3 5",
    )
    parser.add_argument(
        "--require-full-groups",
        action="store_true",
        help="Evaluate only groups with the maximum observed number of candidates.",
    )
    parser.add_argument(
        "--score-source",
        choices=["dfquad", "direct"],
        default="dfquad",
        help="Where to read the ranking score from.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="final_score",
        help="Score field to use. For DF-QuAD use final_score; for MF-only use score.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    dataset_path = Path(args.dataset)
    output_summary_path = Path(args.output_summary)

    records = load_records(input_path)
    dataset_records = load_records(dataset_path)

    records = attach_ranking_metadata(
        records=records,
        dataset_records=dataset_records,
    )

    groups: dict[str, list[dict]] = {}
    skipped_without_group = 0

    for record in records:
        group_id = record.get("ranking_group_id")

        if not isinstance(group_id, str):
            skipped_without_group += 1
            continue

        groups.setdefault(group_id, []).append(record)

    max_group_size = max((len(group) for group in groups.values()), default=0)

    group_results = []
    skipped_incomplete_groups = 0

    for group_id, group_records in groups.items():
        if args.require_full_groups and len(group_records) < max_group_size:
            skipped_incomplete_groups += 1
            continue

        result = evaluate_group(
            records=group_records,
            ks=args.k,
            score_source=args.score_source,
            score_key=args.score_key,
        )

        if not result:
            continue

        result["ranking_group_id"] = group_id
        group_results.append(result)

    summary = {
        "input_file": str(input_path),
        "dataset_file": str(dataset_path),
        "score_source": args.score_source,
        "score_key": args.score_key,
        "num_records": len(records),
        "num_dataset_records": len(dataset_records),
        "num_groups": len(groups),
        "num_groups_evaluated": len(group_results),
        "num_records_without_group": skipped_without_group,
        "num_incomplete_groups_skipped": skipped_incomplete_groups,
        "max_group_size": max_group_size,
        "require_full_groups": args.require_full_groups,
        "k_values": args.k,
        "mrr": mean([r["mrr"] for r in group_results]),
    }

    for k in args.k:
        summary[f"hitrate@{k}"] = mean(
            [r[f"hitrate@{k}"] for r in group_results]
        )
        summary[f"ndcg@{k}"] = mean(
            [r[f"ndcg@{k}"] for r in group_results]
        )

    save_json(summary, output_summary_path)

    print(f"\nGroups found: {len(groups)}")
    print(f"Groups evaluated: {len(group_results)}")
    print(f"Skipped without group: {skipped_without_group}")
    print(f"Skipped incomplete groups: {skipped_incomplete_groups}")
    print(f"MRR: {summary['mrr']}")

    for k in args.k:
        print(f"HitRate@{k}: {summary[f'hitrate@{k}']}")
        print(f"NDCG@{k}:    {summary[f'ndcg@{k}']}")

    print(f"Summary: {output_summary_path}")


if __name__ == "__main__":
    main()