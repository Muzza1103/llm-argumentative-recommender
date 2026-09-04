from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .aspects import HOTEL_ASPECT_SET, validate_hotel_aspect
from .errors import HotelDataValidationError
from .models import (
    AnnotationStatus,
    AspectLabel,
    AspectProfile,
    Facility,
    HotelMetadata,
    HotelPolicy,
    HotelProfile,
    HotelProfileDataset,
    HotelReview,
    HotelStats,
    ReviewSignal,
    Stance,
)


PathLike = str | Path


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HotelDataValidationError("expected an object", path=path)
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise HotelDataValidationError("expected a list", path=path)
    return value


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HotelDataValidationError(
            "expected a non-empty string",
            path=path,
        )
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HotelDataValidationError(
            "expected a string or null",
            path=path,
        )
    return value if value.strip() else None


def _integer(value: object, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HotelDataValidationError("expected an integer", path=path)
    if value < minimum:
        raise HotelDataValidationError(
            f"expected a value >= {minimum}",
            path=path,
        )
    return value


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HotelDataValidationError("expected a number", path=path)
    number = float(value)
    if minimum is not None and number < minimum:
        raise HotelDataValidationError(
            f"expected a value >= {minimum}",
            path=path,
        )
    if maximum is not None and number > maximum:
        raise HotelDataValidationError(
            f"expected a value <= {maximum}",
            path=path,
        )
    return number


def _optional_unit_score(value: object, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=0.0, maximum=1.0)


def _enum_value(enum_type, value: object, path: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise HotelDataValidationError(
            f"expected one of: {allowed}",
            path=path,
        ) from exc


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    return tuple(
        _required_string(item, f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )


def _official_aspect_tuple(value: object, path: str) -> tuple[str, ...]:
    aspects = tuple(
        validate_hotel_aspect(item, path=f"{path}[{index}]")
        for index, item in enumerate(_sequence(value, path))
    )
    if len(aspects) != len(set(aspects)):
        raise HotelDataValidationError(
            "contains duplicate aspects",
            path=path,
        )
    return aspects


def _normalize_facility_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def deduplicate_facilities(
    raw_facilities: Sequence[Any],
    *,
    path: str = "facilities",
) -> tuple[Facility, ...]:
    """Merge facility records by normalized name and retain all source ids."""
    grouped: dict[str, tuple[str, list[int]]] = {}

    for index, raw_facility in enumerate(raw_facilities):
        item_path = f"{path}[{index}]"
        facility = _mapping(raw_facility, item_path)
        facility_id = _integer(
            facility.get("facility_id"),
            f"{item_path}.facility_id",
        )
        raw_name = _required_string(
            facility.get("name"),
            f"{item_path}.name",
        )
        display_name = re.sub(r"\s+", " ", raw_name).strip()
        key = _normalize_facility_name(raw_name)

        if key not in grouped:
            grouped[key] = (display_name, [])

        ids = grouped[key][1]
        if facility_id not in ids:
            ids.append(facility_id)

    return tuple(
        Facility(name=name, facility_ids=tuple(ids))
        for name, ids in grouped.values()
    )


def hotel_review_from_dict(
    raw_review: object,
    *,
    path: str = "review",
) -> HotelReview:
    review = _mapping(raw_review, path)
    status = _enum_value(
        AnnotationStatus,
        review.get("annotation_status"),
        f"{path}.annotation_status",
    )

    signals = []
    for index, raw_signal in enumerate(
        _sequence(review.get("signals"), f"{path}.signals")
    ):
        signal_path = f"{path}.signals[{index}]"
        signal = _mapping(raw_signal, signal_path)
        signals.append(
            ReviewSignal(
                aspect=validate_hotel_aspect(
                    signal.get("aspect"),
                    path=f"{signal_path}.aspect",
                ),
                stance=_enum_value(
                    Stance,
                    signal.get("stance"),
                    f"{signal_path}.stance",
                ),
                evidence=_required_string(
                    signal.get("evidence"),
                    f"{signal_path}.evidence",
                ),
            )
        )

    if status is AnnotationStatus.ANNOTATED and not signals:
        raise HotelDataValidationError(
            "annotated reviews must contain at least one signal",
            path=f"{path}.signals",
        )
    if status is not AnnotationStatus.ANNOTATED and signals:
        raise HotelDataValidationError(
            f"{status.value} reviews must not contain signals",
            path=f"{path}.signals",
        )

    return HotelReview(
        dataset_index=_integer(
            review.get("dataset_index"),
            f"{path}.dataset_index",
        ),
        review_id=_required_string(review.get("review_id"), f"{path}.review_id"),
        hotel_id=_required_string(review.get("hotel_id"), f"{path}.hotel_id"),
        liteapi_id=_required_string(
            review.get("liteapi_id"),
            f"{path}.liteapi_id",
        ),
        hotel_country_code=_required_string(
            review.get("hotel_country_code"),
            f"{path}.hotel_country_code",
        ),
        author_country_code=_optional_string(
            review.get("author_country_code"),
            f"{path}.author_country_code",
        ),
        review_language=_required_string(
            review.get("review_language"),
            f"{path}.review_language",
        ),
        rating=_number(
            review.get("rating"),
            f"{path}.rating",
            minimum=0.0,
            maximum=10.0,
        ),
        headline=_optional_string(review.get("headline"), f"{path}.headline"),
        pros=_optional_string(review.get("pros"), f"{path}.pros"),
        cons=_optional_string(review.get("cons"), f"{path}.cons"),
        review_text=_required_string(
            review.get("review_text"),
            f"{path}.review_text",
        ),
        review_date=_required_string(
            review.get("review_date"),
            f"{path}.review_date",
        ),
        annotation_status=status,
        signals=tuple(signals),
    )


def _aspect_profile_from_dict(raw_aspect: object, path: str) -> AspectProfile:
    aspect = _mapping(raw_aspect, path)
    profile = AspectProfile(
        aspect=validate_hotel_aspect(
            aspect.get("aspect"),
            path=f"{path}.aspect",
        ),
        label=_enum_value(
            AspectLabel,
            aspect.get("label"),
            f"{path}.label",
        ),
        score=_number(
            aspect.get("score"),
            f"{path}.score",
            minimum=0.0,
            maximum=1.0,
        ),
        confidence=_number(
            aspect.get("confidence"),
            f"{path}.confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        wilson_lower=_optional_unit_score(
            aspect.get("wilson_lower"),
            f"{path}.wilson_lower",
        ),
        wilson_upper=_optional_unit_score(
            aspect.get("wilson_upper"),
            f"{path}.wilson_upper",
        ),
        wilson_width=_optional_unit_score(
            aspect.get("wilson_width"),
            f"{path}.wilson_width",
        ),
        n_mentions=_integer(aspect.get("n_mentions"), f"{path}.n_mentions"),
        support_count=_integer(
            aspect.get("support_count"),
            f"{path}.support_count",
        ),
        attack_count=_integer(
            aspect.get("attack_count"),
            f"{path}.attack_count",
        ),
        neutral_count=_integer(
            aspect.get("neutral_count"),
            f"{path}.neutral_count",
        ),
        n_raw_signals_total=_integer(
            aspect.get("n_raw_signals_total"),
            f"{path}.n_raw_signals_total",
        ),
        evidence_examples=_string_tuple(
            aspect.get("evidence_examples"),
            f"{path}.evidence_examples",
        ),
    )

    stance_total = (
        profile.support_count + profile.attack_count + profile.neutral_count
    )
    if profile.n_mentions != stance_total:
        raise HotelDataValidationError(
            "n_mentions must equal support_count + attack_count + neutral_count",
            path=path,
        )
    if profile.n_raw_signals_total < profile.n_mentions:
        raise HotelDataValidationError(
            "n_raw_signals_total must be >= n_mentions",
            path=path,
        )

    wilson_values = (
        profile.wilson_lower,
        profile.wilson_upper,
        profile.wilson_width,
    )
    if any(value is None for value in wilson_values) and not all(
        value is None for value in wilson_values
    ):
        raise HotelDataValidationError(
            "Wilson fields must be either all present or all null",
            path=path,
        )

    return profile


def _hotel_stats_from_dict(raw_stats: object, path: str) -> HotelStats:
    stats = _mapping(raw_stats, path)
    result = HotelStats(
        n_reviews_total=_integer(
            stats.get("n_reviews_total"),
            f"{path}.n_reviews_total",
        ),
        n_reviews_annotated=_integer(
            stats.get("n_reviews_annotated"),
            f"{path}.n_reviews_annotated",
        ),
        n_reviews_missing_or_error=_integer(
            stats.get("n_reviews_missing_or_error"),
            f"{path}.n_reviews_missing_or_error",
        ),
        average_rating=_number(
            stats.get("average_rating"),
            f"{path}.average_rating",
            minimum=0.0,
            maximum=10.0,
        ),
        median_rating=_number(
            stats.get("median_rating"),
            f"{path}.median_rating",
            minimum=0.0,
            maximum=10.0,
        ),
        min_rating=_number(
            stats.get("min_rating"),
            f"{path}.min_rating",
            minimum=0.0,
            maximum=10.0,
        ),
        max_rating=_number(
            stats.get("max_rating"),
            f"{path}.max_rating",
            minimum=0.0,
            maximum=10.0,
        ),
        rating_std=_number(
            stats.get("rating_std"),
            f"{path}.rating_std",
            minimum=0.0,
        ),
        n_aspects_mentioned=_integer(
            stats.get("n_aspects_mentioned"),
            f"{path}.n_aspects_mentioned",
        ),
        total_mentions=_integer(
            stats.get("total_mentions"),
            f"{path}.total_mentions",
        ),
        total_support=_integer(
            stats.get("total_support"),
            f"{path}.total_support",
        ),
        total_attack=_integer(
            stats.get("total_attack"),
            f"{path}.total_attack",
        ),
        total_neutral=_integer(
            stats.get("total_neutral"),
            f"{path}.total_neutral",
        ),
        weighted_aspect_score=_number(
            stats.get("weighted_aspect_score"),
            f"{path}.weighted_aspect_score",
            minimum=0.0,
            maximum=1.0,
        ),
    )

    if not (
        result.min_rating
        <= result.median_rating
        <= result.max_rating
    ):
        raise HotelDataValidationError(
            "expected min_rating <= median_rating <= max_rating",
            path=path,
        )
    if not result.min_rating <= result.average_rating <= result.max_rating:
        raise HotelDataValidationError(
            "average_rating must be between min_rating and max_rating",
            path=path,
        )

    return result


def _policy_from_dict(raw_policy: object, path: str) -> HotelPolicy:
    policy = _mapping(raw_policy, path)
    return HotelPolicy(
        source_id=_optional_integer(policy.get("id"), f"{path}.id"),
        name=_required_string(policy.get("name"), f"{path}.name"),
        policy_type=_optional_string(
            policy.get("policy_type"),
            f"{path}.policy_type",
        ),
        description=_required_string(
            policy.get("description"),
            f"{path}.description",
        ),
        parking=_optional_string(policy.get("parking"), f"{path}.parking"),
        pets_allowed=_optional_string(
            policy.get("pets_allowed"),
            f"{path}.pets_allowed",
        ),
        child_allowed=_optional_string(
            policy.get("child_allowed"),
            f"{path}.child_allowed",
        ),
    )


def _metadata_from_dict(raw_metadata: object, path: str) -> HotelMetadata:
    metadata = _mapping(raw_metadata, path)
    raw_policies = _sequence(metadata.get("policies"), f"{path}.policies")
    raw_facility_ids = _sequence(
        metadata.get("facility_ids"),
        f"{path}.facility_ids",
    )
    raw_facilities = _sequence(
        metadata.get("facilities"),
        f"{path}.facilities",
    )
    completeness = _mapping(
        metadata.get("data_completeness"),
        f"{path}.data_completeness",
    )

    return HotelMetadata(
        liteapi_id=_required_string(
            metadata.get("liteapi_id"),
            f"{path}.liteapi_id",
        ),
        name=_required_string(metadata.get("name"), f"{path}.name"),
        description=_required_string(
            metadata.get("description"),
            f"{path}.description",
        ),
        city=_optional_string(metadata.get("city"), f"{path}.city"),
        policies=tuple(
            _policy_from_dict(policy, f"{path}.policies[{index}]")
            for index, policy in enumerate(raw_policies)
        ),
        source_facility_ids=tuple(
            _integer(item, f"{path}.facility_ids[{index}]")
            for index, item in enumerate(raw_facility_ids)
        ),
        facilities=deduplicate_facilities(
            raw_facilities,
            path=f"{path}.facilities",
        ),
        data_completeness=dict(completeness),
    )


def _validate_hotel_consistency(hotel: HotelProfile, path: str) -> None:
    stats = hotel.stats
    reviews = hotel.reviews
    aspects = hotel.aspects

    if stats.n_reviews_total != len(reviews):
        raise HotelDataValidationError(
            "n_reviews_total does not match the embedded review count",
            path=f"{path}.hotel_stats.n_reviews_total",
        )

    annotated_count = sum(
        review.annotation_status is not AnnotationStatus.MISSING_OR_ERROR
        for review in reviews
    )
    error_count = sum(
        review.annotation_status is AnnotationStatus.MISSING_OR_ERROR
        for review in reviews
    )
    if stats.n_reviews_annotated != annotated_count:
        raise HotelDataValidationError(
            "n_reviews_annotated does not match review statuses",
            path=f"{path}.hotel_stats.n_reviews_annotated",
        )
    if stats.n_reviews_missing_or_error != error_count:
        raise HotelDataValidationError(
            "n_reviews_missing_or_error does not match review statuses",
            path=f"{path}.hotel_stats.n_reviews_missing_or_error",
        )

    if any(review.hotel_id != hotel.hotel_id for review in reviews):
        raise HotelDataValidationError(
            "an embedded review belongs to another hotel",
            path=f"{path}.reviews",
        )
    if hotel.metadata.liteapi_id != hotel.hotel_id:
        raise HotelDataValidationError(
            "hotel_metadata.liteapi_id must match hotel_id",
            path=f"{path}.hotel_metadata.liteapi_id",
        )

    aspect_names = [aspect.aspect for aspect in aspects]
    if len(aspect_names) != len(set(aspect_names)):
        raise HotelDataValidationError(
            "contains duplicate aspect profiles",
            path=f"{path}.aspects",
        )
    if stats.n_aspects_mentioned != len(aspects):
        raise HotelDataValidationError(
            "n_aspects_mentioned does not match the aspect profile count",
            path=f"{path}.hotel_stats.n_aspects_mentioned",
        )

    expected_totals = {
        "total_mentions": sum(aspect.n_mentions for aspect in aspects),
        "total_support": sum(aspect.support_count for aspect in aspects),
        "total_attack": sum(aspect.attack_count for aspect in aspects),
        "total_neutral": sum(aspect.neutral_count for aspect in aspects),
    }
    for field_name, expected in expected_totals.items():
        if getattr(stats, field_name) != expected:
            raise HotelDataValidationError(
                f"{field_name} does not match aspect profile counts",
                path=f"{path}.hotel_stats.{field_name}",
            )

    available_aspects = set(aspect_names)
    ranked_aspects = hotel.top_positive_aspects + hotel.top_negative_aspects
    if any(aspect not in available_aspects for aspect in ranked_aspects):
        raise HotelDataValidationError(
            "top aspect lists reference an unobserved aspect",
            path=path,
        )


def _hotel_profile_from_dict(raw_hotel: object, path: str) -> HotelProfile:
    hotel = _mapping(raw_hotel, path)
    raw_aspects = _sequence(hotel.get("aspects"), f"{path}.aspects")
    raw_reviews = _sequence(hotel.get("reviews"), f"{path}.reviews")

    profile = HotelProfile(
        hotel_id=_required_string(hotel.get("hotel_id"), f"{path}.hotel_id"),
        stats=_hotel_stats_from_dict(
            hotel.get("hotel_stats"),
            f"{path}.hotel_stats",
        ),
        top_positive_aspects=_official_aspect_tuple(
            hotel.get("top_positive_aspects"),
            f"{path}.top_positive_aspects",
        ),
        top_negative_aspects=_official_aspect_tuple(
            hotel.get("top_negative_aspects"),
            f"{path}.top_negative_aspects",
        ),
        aspects=tuple(
            _aspect_profile_from_dict(item, f"{path}.aspects[{index}]")
            for index, item in enumerate(raw_aspects)
        ),
        reviews=tuple(
            hotel_review_from_dict(item, path=f"{path}.reviews[{index}]")
            for index, item in enumerate(raw_reviews)
        ),
        metadata=_metadata_from_dict(
            hotel.get("hotel_metadata"),
            f"{path}.hotel_metadata",
        ),
    )
    _validate_hotel_consistency(profile, path)
    return profile


def hotel_profile_dataset_from_dict(
    raw_dataset: object,
    *,
    path: str = "root",
) -> HotelProfileDataset:
    dataset = _mapping(raw_dataset, path)
    raw_catalog = _mapping(
        dataset.get("aspect_catalog"),
        f"{path}.aspect_catalog",
    )
    catalog_keys = set(raw_catalog)
    if catalog_keys != HOTEL_ASPECT_SET:
        missing = sorted(HOTEL_ASPECT_SET - catalog_keys)
        unknown = sorted(catalog_keys - HOTEL_ASPECT_SET)
        raise HotelDataValidationError(
            f"catalog mismatch (missing={missing}, unknown={unknown})",
            path=f"{path}.aspect_catalog",
        )
    aspect_catalog = {
        aspect: _required_string(
            description,
            f"{path}.aspect_catalog.{aspect}",
        )
        for aspect, description in raw_catalog.items()
    }

    raw_hotels = _sequence(dataset.get("hotels"), f"{path}.hotels")
    hotels = tuple(
        _hotel_profile_from_dict(hotel, f"{path}.hotels[{index}]")
        for index, hotel in enumerate(raw_hotels)
    )
    n_hotels = _integer(dataset.get("n_hotels"), f"{path}.n_hotels")
    n_reviews = _integer(dataset.get("n_reviews"), f"{path}.n_reviews")

    if n_hotels != len(hotels):
        raise HotelDataValidationError(
            "declared n_hotels does not match the hotel list",
            path=f"{path}.n_hotels",
        )
    embedded_review_count = sum(len(hotel.reviews) for hotel in hotels)
    if n_reviews != embedded_review_count:
        raise HotelDataValidationError(
            "declared n_reviews does not match embedded reviews",
            path=f"{path}.n_reviews",
        )

    hotel_ids = [hotel.hotel_id for hotel in hotels]
    if len(hotel_ids) != len(set(hotel_ids)):
        raise HotelDataValidationError(
            "hotel ids must be unique",
            path=f"{path}.hotels",
        )
    review_ids = [review.review_id for hotel in hotels for review in hotel.reviews]
    if len(review_ids) != len(set(review_ids)):
        raise HotelDataValidationError(
            "embedded review ids must be unique",
            path=f"{path}.hotels",
        )

    enrichment = _mapping(
        dataset.get("metadata_enrichment"),
        f"{path}.metadata_enrichment",
    )

    return HotelProfileDataset(
        schema_version=_required_string(
            dataset.get("schema_version"),
            f"{path}.schema_version",
        ),
        dataset_name=_required_string(
            dataset.get("dataset_name"),
            f"{path}.dataset_name",
        ),
        model_id=_required_string(
            dataset.get("model_id"),
            f"{path}.model_id",
        ),
        annotation_method=_required_string(
            dataset.get("annotation_method"),
            f"{path}.annotation_method",
        ),
        n_hotels=n_hotels,
        n_reviews=n_reviews,
        aspect_catalog=aspect_catalog,
        hotels=hotels,
        metadata_enrichment=dict(enrichment),
    )


def load_hotel_profiles(path: PathLike) -> HotelProfileDataset:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            raw_dataset = json.load(stream)
    except json.JSONDecodeError as exc:
        raise HotelDataValidationError(
            f"invalid JSON: {exc.msg}",
            path=str(input_path),
        ) from exc
    return hotel_profile_dataset_from_dict(raw_dataset, path=str(input_path))


def iter_review_annotations(path: PathLike) -> Iterator[HotelReview]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw_review = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HotelDataValidationError(
                    f"invalid JSON: {exc.msg}",
                    path=f"{input_path}:{line_number}",
                ) from exc
            yield hotel_review_from_dict(
                raw_review,
                path=f"{input_path}:{line_number}",
            )


def _validate_unique_reviews(
    reviews: Iterable[HotelReview],
    *,
    path: str,
) -> tuple[HotelReview, ...]:
    result = tuple(reviews)
    review_ids = [review.review_id for review in result]
    indices = [review.dataset_index for review in result]
    if len(review_ids) != len(set(review_ids)):
        raise HotelDataValidationError("review ids must be unique", path=path)
    if len(indices) != len(set(indices)):
        raise HotelDataValidationError(
            "dataset_index values must be unique",
            path=path,
        )
    return result


def load_review_annotations(path: PathLike) -> tuple[HotelReview, ...]:
    return _validate_unique_reviews(
        iter_review_annotations(path),
        path=str(path),
    )


def validate_reviews_match_profiles(
    dataset: HotelProfileDataset,
    reviews: Iterable[HotelReview],
) -> None:
    """Validate that standalone JSONL reviews equal embedded profile reviews."""
    standalone = _validate_unique_reviews(reviews, path="review_annotations")
    embedded = tuple(review for hotel in dataset.hotels for review in hotel.reviews)
    standalone_by_id = {review.review_id: review for review in standalone}
    embedded_by_id = {review.review_id: review for review in embedded}

    missing = sorted(set(embedded_by_id) - set(standalone_by_id))
    extra = sorted(set(standalone_by_id) - set(embedded_by_id))
    if missing or extra:
        raise HotelDataValidationError(
            f"review id mismatch (missing={missing[:5]}, extra={extra[:5]})",
            path="review_annotations",
        )

    for review_id, embedded_review in embedded_by_id.items():
        if standalone_by_id[review_id] != embedded_review:
            raise HotelDataValidationError(
                "standalone and embedded review records differ",
                path=f"review_annotations[{review_id}]",
            )
