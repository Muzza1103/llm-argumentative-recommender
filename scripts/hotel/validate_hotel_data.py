import argparse
import json
from collections import Counter
from pathlib import Path

from src.hotel import (
    HotelProfileDataset,
    HotelReview,
    load_hotel_profiles,
    load_review_annotations,
    validate_reviews_match_profiles,
)


def build_validation_summary(
    dataset: HotelProfileDataset,
    reviews: tuple[HotelReview, ...] | None = None,
) -> dict:
    embedded_reviews = tuple(
        review
        for hotel in dataset.hotels
        for review in hotel.reviews
    )
    effective_reviews = reviews if reviews is not None else embedded_reviews
    status_counts = Counter(
        review.annotation_status.value
        for review in effective_reviews
    )
    aspect_counts = [len(hotel.aspects) for hotel in dataset.hotels]

    return {
        "schema_version": dataset.schema_version,
        "dataset_name": dataset.dataset_name,
        "n_hotels": len(dataset.hotels),
        "n_profile_reviews": len(embedded_reviews),
        "n_review_annotations": (
            len(reviews) if reviews is not None else None
        ),
        "cross_file_match": reviews is not None,
        "annotation_status_counts": dict(sorted(status_counts.items())),
        "n_signals": sum(
            len(review.signals)
            for review in effective_reviews
        ),
        "min_observed_aspects_per_hotel": (
            min(aspect_counts) if aspect_counts else None
        ),
        "max_observed_aspects_per_hotel": (
            max(aspect_counts) if aspect_counts else None
        ),
        "n_hotels_missing_city": sum(
            hotel.metadata.city is None
            for hotel in dataset.hotels
        ),
        "n_hotels_without_policies": sum(
            not hotel.metadata.policies
            for hotel in dataset.hotels
        ),
        "n_raw_facilities": sum(
            len(hotel.metadata.source_facility_ids)
            for hotel in dataset.hotels
        ),
        "n_deduplicated_facilities": sum(
            len(hotel.metadata.facilities)
            for hotel in dataset.hotels
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate local enriched hotel profiles and review annotations."
    )
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--reviews")
    parser.add_argument("--output-summary")
    args = parser.parse_args()

    dataset = load_hotel_profiles(args.profiles)
    reviews = None

    if args.reviews:
        reviews = load_review_annotations(args.reviews)
        validate_reviews_match_profiles(dataset, reviews)

    summary = build_validation_summary(dataset, reviews)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output_summary:
        output_path = Path(args.output_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
