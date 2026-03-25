import json
import random
from pathlib import Path

from tqdm import tqdm


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "yelp_subset.jsonl"

BUSINESS_FILE = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_FILE = RAW_DIR / "yelp_academic_dataset_review.json"

CATEGORY_FILTER = "Restaurants"
MIN_USER_REVIEWS = 5
MAX_USER_REVIEWS = 50
MIN_REVIEW_LENGTH = 30
NB_USERS = 100
HISTORY_SIZE = 5
SEED = 42
NB_POSITIVE_HISTORY = 2
NB_NEGATIVE_HISTORY = 2


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


def load_filtered_reviews(valid_business_ids):
    user_reviews = {}

    with REVIEW_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading reviews"):
            obj = json.loads(line)

            business_id = obj["business_id"]
            if business_id not in valid_business_ids:
                continue

            text = obj.get("text", "").strip()
            if len(text) < MIN_REVIEW_LENGTH:
                continue

            user_id = obj["user_id"]

            if user_id not in user_reviews:
                user_reviews[user_id] = []

            user_reviews[user_id].append(
                {
                    "business_id": business_id,
                    "user_stars": obj.get("stars"),
                    "review_text": text,
                    "date": obj.get("date", ""),
                }
            )

    return user_reviews


def build_examples(businesses, user_reviews):
    examples = []

    valid_users = [
        user_id
        for user_id, reviews in user_reviews.items()
        if MIN_USER_REVIEWS <= len(reviews) <= MAX_USER_REVIEWS
    ]

    random.shuffle(valid_users)

    for user_id in valid_users[:NB_USERS]:
        reviews = sorted(user_reviews[user_id], key=lambda x: x["date"])

        if len(reviews) < HISTORY_SIZE + 1:
            continue

        target_review = reviews[-1]
        
        candidate_history = reviews[:-1]

        positive_reviews = [r for r in candidate_history if r["user_stars"] >= 4]
        negative_reviews = [r for r in candidate_history if r["user_stars"] <= 2]
        neutral_reviews = [r for r in candidate_history if r["user_stars"] == 3]

        history_reviews = []

        history_reviews.extend(positive_reviews[-2:])
        history_reviews.extend(negative_reviews[-2:])

        if len(history_reviews) < HISTORY_SIZE:
            already_selected_ids = {
                (r["business_id"], r["date"], r["review_text"]) for r in history_reviews
            }

            fallback_reviews = list(reversed(candidate_history))
            for r in fallback_reviews:
                key = (r["business_id"], r["date"], r["review_text"])
                if key in already_selected_ids:
                    continue
                history_reviews.append(r)
                already_selected_ids.add(key)
                if len(history_reviews) >= HISTORY_SIZE:
                    break

        history_reviews = sorted(history_reviews, key=lambda x: x["date"])

        target_business_id = target_review["business_id"]
        if target_business_id not in businesses:
            continue

        history = []
        skip_user = False

        for review in history_reviews:
            business_id = review["business_id"]

            if business_id not in businesses:
                skip_user = True
                break

            business_info = businesses[business_id]

            history.append(
                {
                    "business_id": business_id,
                    "name": business_info["name"],
                    "categories": business_info["categories"],
                    "user_stars": review["user_stars"],
                    "review_text": review["review_text"],
                }
            )

        if skip_user:
            continue

        target_business = businesses[target_business_id]

        example = {
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

        examples.append(example)

    return examples


def save_jsonl(examples, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    print("Loading restaurant businesses")
    businesses = load_restaurant_businesses()
    print(f"Loaded {len(businesses)} restaurant businesses.")

    print("Loading filtered reviews")
    user_reviews = load_filtered_reviews(set(businesses.keys()))
    print(f"Loaded reviews for {len(user_reviews)} users.")

    print("Building examples")
    examples = build_examples(businesses, user_reviews)
    print(f"Built {len(examples)} examples.")

    print("Saving subset")
    save_jsonl(examples, OUTPUT_FILE)
    print(f"Saved subset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()