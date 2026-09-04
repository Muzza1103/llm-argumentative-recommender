from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .aspects import validate_hotel_aspect


class AnnotationStatus(str, Enum):
    ANNOTATED = "annotated"
    ANNOTATED_NO_ASPECT = "annotated_no_aspect"
    MISSING_OR_ERROR = "missing_or_error"


class Stance(str, Enum):
    SUPPORT = "support"
    ATTACK = "attack"
    NEUTRAL = "neutral"


class AspectLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class ReviewSignal:
    aspect: str
    stance: Stance
    evidence: str


@dataclass(frozen=True, slots=True)
class HotelReview:
    dataset_index: int
    review_id: str
    hotel_id: str
    liteapi_id: str
    hotel_country_code: str
    author_country_code: str | None
    review_language: str
    rating: float
    headline: str | None
    pros: str | None
    cons: str | None
    review_text: str
    review_date: str
    annotation_status: AnnotationStatus
    signals: tuple[ReviewSignal, ...] = field(default_factory=tuple)

    @property
    def has_aspect_annotations(self) -> bool:
        return self.annotation_status is AnnotationStatus.ANNOTATED

    @property
    def annotation_failed(self) -> bool:
        return self.annotation_status is AnnotationStatus.MISSING_OR_ERROR


@dataclass(frozen=True, slots=True)
class AspectProfile:
    aspect: str
    label: AspectLabel
    score: float
    confidence: float
    wilson_lower: float | None
    wilson_upper: float | None
    wilson_width: float | None
    n_mentions: int
    support_count: int
    attack_count: int
    neutral_count: int
    n_raw_signals_total: int
    evidence_examples: tuple[str, ...] = field(default_factory=tuple)

    @property
    def raw_stance_score(self) -> float:
        """Unshrunk opinion score; neutral mentions contribute 0.5."""
        if self.n_mentions == 0:
            return 0.5
        return (
            self.support_count + 0.5 * self.neutral_count
        ) / self.n_mentions


@dataclass(frozen=True, slots=True)
class HotelStats:
    n_reviews_total: int
    n_reviews_annotated: int
    n_reviews_missing_or_error: int
    average_rating: float
    median_rating: float
    min_rating: float
    max_rating: float
    rating_std: float
    n_aspects_mentioned: int
    total_mentions: int
    total_support: int
    total_attack: int
    total_neutral: int
    weighted_aspect_score: float


@dataclass(frozen=True, slots=True)
class Facility:
    """A factual facility deduplicated by name, retaining every source id."""

    name: str
    facility_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HotelPolicy:
    source_id: int | None
    name: str
    policy_type: str | None
    description: str
    parking: str | None = None
    pets_allowed: str | None = None
    child_allowed: str | None = None

    def semantic_payload(self) -> dict[str, str]:
        """Return prompt-safe semantic fields, excluding ids and empty values."""
        payload = {
            "name": self.name,
            "policy_type": self.policy_type,
            "description": self.description,
            "parking": self.parking,
            "pets_allowed": self.pets_allowed,
            "child_allowed": self.child_allowed,
        }
        return {
            key: value.strip()
            for key, value in payload.items()
            if isinstance(value, str) and value.strip()
        }


@dataclass(frozen=True, slots=True)
class HotelMetadata:
    """Factual hotel data, deliberately separate from review opinions."""

    liteapi_id: str
    name: str
    description: str
    city: str | None
    policies: tuple[HotelPolicy, ...]
    source_facility_ids: tuple[int, ...]
    facilities: tuple[Facility, ...]
    data_completeness: dict[str, Any]

    def facility_names(self) -> tuple[str, ...]:
        return tuple(facility.name for facility in self.facilities)


@dataclass(frozen=True, slots=True)
class HotelProfile:
    hotel_id: str
    stats: HotelStats
    top_positive_aspects: tuple[str, ...]
    top_negative_aspects: tuple[str, ...]
    aspects: tuple[AspectProfile, ...]
    reviews: tuple[HotelReview, ...]
    metadata: HotelMetadata

    def get_aspect(self, aspect: str) -> AspectProfile | None:
        """Return None when an official aspect was not observed in reviews."""
        validated = validate_hotel_aspect(aspect)
        return next(
            (profile for profile in self.aspects if profile.aspect == validated),
            None,
        )


@dataclass(frozen=True, slots=True)
class HotelProfileDataset:
    schema_version: str
    dataset_name: str
    model_id: str
    annotation_method: str
    n_hotels: int
    n_reviews: int
    aspect_catalog: dict[str, str]
    hotels: tuple[HotelProfile, ...]
    metadata_enrichment: dict[str, Any]

    def get_hotel(self, hotel_id: str) -> HotelProfile | None:
        return next(
            (hotel for hotel in self.hotels if hotel.hotel_id == hotel_id),
            None,
        )
