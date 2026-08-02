import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"

CATEGORY_FILTER = "Restaurants"
MAX_USER_REVIEWS = 50
MIN_REVIEW_LENGTH = 30
SEED = 42


def load_restaurant_businesses():
    businesses = {}

    with BUSINESS_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading businesses"):
            obj = json.loads(line)

            categories = obj.get("categories")
            if not categories:
                continue

            category_list = [c.strip() for c in categories.split(",")]
            if CATEGORY_FILTER not in category_list:
                continue

            businesses[obj["business_id"]] = {
                "business_id": obj["business_id"],
                "name": obj.get("name", ""),
                "categories": category_list,
                "attributes": obj.get("attributes", {}) or {},
                "stars": obj.get("stars"),
                "review_count": obj.get("review_count"),
            }

    return businesses


def load_filtered_reviews(valid_business_ids, min_item_reviews):
    raw_reviews = []
    item_counts = Counter()

    with REVIEW_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reviews"):
            obj = json.loads(line)

            business_id = obj["business_id"]
            if business_id not in valid_business_ids:
                continue

            text = obj.get("text", "").strip()
            if len(text) < MIN_REVIEW_LENGTH:
                continue

            review = {
                "user_id": obj["user_id"],
                "business_id": business_id,
                "user_stars": obj.get("stars"),
                "review_text": text,
                "date": obj.get("date", ""),
            }

            raw_reviews.append(review)
            item_counts[business_id] += 1

    eligible_business_ids = {
        business_id
        for business_id, count in item_counts.items()
        if count >= min_item_reviews
    }

    user_reviews = {}

    for review in raw_reviews:
        business_id = review["business_id"]

        if business_id not in eligible_business_ids:
            continue

        user_id = review["user_id"]
        user_reviews.setdefault(user_id, []).append(
            {
                "business_id": business_id,
                "user_stars": review["user_stars"],
                "review_text": review["review_text"],
                "date": review["date"],
            }
        )

    return user_reviews, item_counts, eligible_business_ids


def build_single_example(businesses, user_id, reviews, history_size):
    reviews = sorted(reviews, key=lambda x: x["date"])

    if len(reviews) < history_size + 1:
        return None

    target_review = reviews[-1]
    candidate_history = reviews[:-1]

    positive_reviews = [r for r in candidate_history if r["user_stars"] >= 4]
    negative_reviews = [r for r in candidate_history if r["user_stars"] <= 2]

    history_reviews = []
    history_reviews.extend(positive_reviews[-2:])
    history_reviews.extend(negative_reviews[-2:])

    if len(history_reviews) < history_size:
        already_selected_ids = {
            (r["business_id"], r["date"], r["review_text"])
            for r in history_reviews
        }

        fallback_reviews = list(reversed(candidate_history))

        for review in fallback_reviews:
            key = (
                review["business_id"],
                review["date"],
                review["review_text"],
            )

            if key in already_selected_ids:
                continue

            history_reviews.append(review)
            already_selected_ids.add(key)

            if len(history_reviews) >= history_size:
                break

    history_reviews = sorted(history_reviews, key=lambda x: x["date"])

    target_business_id = target_review["business_id"]
    if target_business_id not in businesses:
        return None

    history = []

    for review in history_reviews:
        business_id = review["business_id"]

        if business_id not in businesses:
            return None

        business_info = businesses[business_id]

        history.append(
            {
                "business_id": business_id,
                "name": business_info["name"],
                "categories": business_info["categories"],
                "attributes": business_info["attributes"],
                "user_stars": review["user_stars"],
                "review_text": review["review_text"],
            }
        )

    if len(history) < history_size:
        return None

    target_business = businesses[target_business_id]

    return {
        "user_id": user_id,
        "history": history,
        "target_item": {
            "business_id": target_business_id,
            "name": target_business["name"],
            "categories": target_business["categories"],
            "attributes": target_business["attributes"],
            "global_stars": target_business["stars"],
            "review_count": target_business["review_count"],
            "user_target_stars": target_review["user_stars"],
            "target_review_text": target_review["review_text"],
        },
    }


def count_history_items(examples):
    counts = Counter()

    for example in examples:
        for item in example.get("history", []):
            business_id = item.get("business_id")
            if business_id:
                counts[business_id] += 1

    return counts


def is_mf_compatible_example(
    example,
    history_item_counts,
    min_history_item_occurrences,
):
    target_id = example["target_item"]["business_id"]

    if history_item_counts.get(target_id, 0) < min_history_item_occurrences:
        return False

    for item in example.get("history", []):
        business_id = item.get("business_id")

        if history_item_counts.get(business_id, 0) < min_history_item_occurrences:
            return False

    return True


def build_examples(
    businesses,
    user_reviews,
    nb_users,
    history_size,
    min_user_reviews,
    min_history_item_occurrences,
    candidate_multiplier,
):
    valid_users = [
        user_id
        for user_id, reviews in user_reviews.items()
        if min_user_reviews <= len(reviews) <= MAX_USER_REVIEWS
    ]

    random.shuffle(valid_users)

    candidate_examples = []
    max_candidates = nb_users * candidate_multiplier if candidate_multiplier > 0 else None

    for user_id in valid_users:
        example = build_single_example(
            businesses=businesses,
            user_id=user_id,
            reviews=user_reviews[user_id],
            history_size=history_size,
        )

        if example is None:
            continue

        candidate_examples.append(example)

        if max_candidates is not None and len(candidate_examples) >= max_candidates:
            break

    history_item_counts = count_history_items(candidate_examples)

    examples = [
        example
        for example in candidate_examples
        if is_mf_compatible_example(
            example=example,
            history_item_counts=history_item_counts,
            min_history_item_occurrences=min_history_item_occurrences,
        )
    ]

    return examples[:nb_users], candidate_examples, history_item_counts


def save_jsonl(examples, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def save_summary(
    examples,
    candidate_examples,
    output_file,
    min_item_reviews,
    min_history_item_occurrences,
    eligible_business_ids,
    item_counts,
    history_item_counts,
    runtime_seconds,
):
    target_ids = [
        ex["target_item"]["business_id"]
        for ex in examples
    ]

    history_ids = [
        item["business_id"]
        for ex in examples
        for item in ex["history"]
    ]

    target_raw_counts = [
        item_counts[business_id]
        for business_id in target_ids
    ]

    target_history_counts = [
        history_item_counts.get(business_id, 0)
        for business_id in target_ids
    ]

    selected_history_counts = [
        history_item_counts.get(business_id, 0)
        for business_id in history_ids
    ]

    summary = {
        "output_file": str(output_file),
        "num_examples": len(examples),
        "num_candidate_examples": len(candidate_examples),
        "min_item_reviews": min_item_reviews,
        "min_history_item_occurrences": min_history_item_occurrences,
        "num_eligible_businesses": len(eligible_business_ids),
        "num_unique_target_items": len(set(target_ids)),
        "num_unique_history_items": len(set(history_ids)),
        "target_item_count_min": min(target_raw_counts) if target_raw_counts else None,
        "target_item_count_mean": (
            sum(target_raw_counts) / len(target_raw_counts)
        ) if target_raw_counts else None,
        "target_history_count_min": (
            min(target_history_counts) if target_history_counts else None
        ),
        "target_history_count_mean": (
            sum(target_history_counts) / len(target_history_counts)
        ) if target_history_counts else None,
        "selected_history_count_min": (
            min(selected_history_counts) if selected_history_counts else None
        ),
        "selected_history_count_mean": (
            sum(selected_history_counts) / len(selected_history_counts)
        ) if selected_history_counts else None,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60,
    }

    summary_path = output_file.with_name(f"{output_file.stem}_summary.json")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {summary_path}")


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Build a Yelp subset with item-frequency constraints."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DIR / "yelp_subset.jsonl"),
    )

    parser.add_argument(
        "--nb-users",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--history-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-user-reviews",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--min-item-reviews",
        type=int,
        default=1,
        help="Minimum number of raw reviews required for a business to be kept.",
    )

    parser.add_argument(
        "--min-history-item-occurrences",
        type=int,
        default=1,
        help=(
            "Minimum number of appearances required in the candidate histories "
            "for both target items and selected history items."
        ),
    )

    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=20,
        help=(
            "Number of candidate examples to build before MF-compatibility filtering, "
            "as a multiple of --nb-users. Use 0 to scan all valid users."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    output_file = Path(args.output)

    print("Loading restaurant businesses")
    businesses = load_restaurant_businesses()
    print(f"Loaded {len(businesses)} restaurant businesses.")

    print("Loading filtered reviews")
    user_reviews, item_counts, eligible_business_ids = load_filtered_reviews(
        valid_business_ids=set(businesses.keys()),
        min_item_reviews=args.min_item_reviews,
    )

    print(f"Eligible businesses: {len(eligible_business_ids)}")
    print(f"Loaded reviews for {len(user_reviews)} users.")

    print("Building examples")
    examples, candidate_examples, history_item_counts = build_examples(
        businesses=businesses,
        user_reviews=user_reviews,
        nb_users=args.nb_users,
        history_size=args.history_size,
        min_user_reviews=args.min_user_reviews,
        min_history_item_occurrences=args.min_history_item_occurrences,
        candidate_multiplier=args.candidate_multiplier,
    )

    print(f"Candidate examples: {len(candidate_examples)}")
    print(f"Built {len(examples)} examples.")

    print("Saving subset")
    save_jsonl(examples, output_file)
    print(f"Saved subset to: {output_file}")

    runtime_seconds = time.perf_counter() - start_time

    save_summary(
        examples=examples,
        candidate_examples=candidate_examples,
        output_file=output_file,
        min_item_reviews=args.min_item_reviews,
        min_history_item_occurrences=args.min_history_item_occurrences,
        eligible_business_ids=eligible_business_ids,
        item_counts=item_counts,
        history_item_counts=history_item_counts,
        runtime_seconds=runtime_seconds,
    )


if __name__ == "__main__":
    main()
