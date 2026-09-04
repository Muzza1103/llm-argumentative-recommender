"""Shared Yelp context rendering for DF-QuAD command-line tools."""

from src.prompting.formatters import get_filtered_attributes


def normalize_rating(rating: float | None) -> float | None:
    if rating is None:
        return None
    return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))


def format_review_aspects(item: dict) -> list[str]:
    aspects = []
    for aspect in item.get("review_aspects", []):
        if not isinstance(aspect, dict):
            continue
        name = aspect.get("name")
        polarity = aspect.get("polarity", "neutral")
        if name:
            aspects.append(f"{name} ({polarity})")
    return aspects


def build_context_summary(example: dict) -> dict:
    target_item = example.get("target_item", {})
    history = example.get("history", [])
    user_target_stars = target_item.get("user_target_stars")

    history_summary = [
        {
            "name": item.get("name"),
            "user_stars": item.get("user_stars"),
            "categories": item.get("categories", []),
            "attributes": get_filtered_attributes(
                item.get("attributes", {})
            ),
            "review_aspects": format_review_aspects(item),
        }
        for item in history
    ]

    return {
        "user_id": example.get("user_id"),
        "target_item": {
            "name": target_item.get("name"),
            "categories": target_item.get("categories", []),
            "attributes": get_filtered_attributes(
                target_item.get("attributes", {})
            ),
            "review_aspects": format_review_aspects(target_item),
            "global_stars": target_item.get("global_stars"),
            "user_target_stars": user_target_stars,
            "normalized_user_target_score": normalize_rating(
                user_target_stars
            ),
        },
        "history": history_summary,
    }
