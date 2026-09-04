from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.argumentation.schema import Argument

from .constraints import ConstraintOutcome, ConstraintStatus
from .models import HotelProfile, Stance
from .preferences import ABSOLUTE_5_WEIGHTING_METHOD, SessionPreferences
from .wilson import wilson_lower_bound


EMPIRICAL_ASPECT = "empirical_aspect"
STRUCTURED_FACT = "structured_fact"
SEMANTIC_EXTRA = "semantic_extra"


class SemanticArgumentProvider(Protocol):
    """Future extension point; no implementation or score is assumed here."""

    def build_arguments(
        self,
        hotel: HotelProfile,
        preferences: SessionPreferences,
    ) -> Sequence[Argument]:
        ...


def select_review_sources(
    hotel: HotelProfile,
    *,
    aspect: str,
    stance: Stance,
    limit: int = 2,
) -> list[dict[str, str]]:
    """Select deterministic, distinct evidence from compatible reviews."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

    reviews = sorted(hotel.reviews, key=lambda review: review.review_id)
    reviews = sorted(
        reviews,
        key=lambda review: review.review_date or "",
        reverse=True,
    )
    selected: list[dict[str, str]] = []
    seen_evidence: set[str] = set()

    for review in reviews:
        if review.hotel_id != hotel.hotel_id:
            continue
        for signal in review.signals:
            if signal.aspect != aspect or signal.stance is not stance:
                continue
            dedupe_key = " ".join(signal.evidence.casefold().split())
            if dedupe_key in seen_evidence:
                continue
            seen_evidence.add(dedupe_key)
            selected.append(
                {
                    "review_id": review.review_id,
                    "evidence": signal.evidence,
                    "stance": signal.stance.value,
                    "aspect": signal.aspect,
                }
            )
            if len(selected) >= limit:
                return selected
    return selected


def _empirical_argument(
    hotel: HotelProfile,
    *,
    aspect: str,
    stance: Stance,
    importance_raw: float,
    normalized_weight: float,
) -> Argument | None:
    profile = hotel.get_aspect(aspect)
    if profile is None:
        return None

    n_decisive = profile.support_count + profile.attack_count
    stance_count = (
        profile.support_count
        if stance is Stance.SUPPORT
        else profile.attack_count
    )
    if n_decisive == 0 or stance_count == 0:
        return None

    sources = select_review_sources(
        hotel,
        aspect=aspect,
        stance=stance,
        limit=2,
    )
    if not sources:
        return None

    evidence_score = wilson_lower_bound(stance_count, n_decisive)
    intrinsic_strength = normalized_weight * evidence_score
    force_components = {
        "importance_raw": importance_raw,
        "importance_coefficient": normalized_weight,
        "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
        "wilson_lower_bound": evidence_score,
    }
    direction = "positive" if stance is Stance.SUPPORT else "negative"
    text = (
        f"Guest review signals support a {direction} assessment of "
        f"{aspect}: {profile.support_count} support and "
        f"{profile.attack_count} attack signals."
    )
    return Argument(
        id=f"EMPIRICAL::{aspect}::{stance.value}",
        arg_type=stance.value,
        text=text,
        evidence=[source["evidence"] for source in sources],
        aspect_effect=(
            "present_preferred"
            if stance is Stance.SUPPORT
            else "present_disliked"
        ),
        used_aspects=[aspect],
        target_item_name=hotel.metadata.name,
        argument_family=EMPIRICAL_ASPECT,
        aspect=aspect,
        intrinsic_strength=intrinsic_strength,
        importance_raw=importance_raw,
        normalized_weight=normalized_weight,
        evidence_score=evidence_score,
        n_support=profile.support_count,
        n_attack=profile.attack_count,
        n_neutral=profile.neutral_count,
        review_sources=sources,
        metadata={
            "hotel_id": hotel.hotel_id,
            "wilson_successes": stance_count,
            "wilson_trials": n_decisive,
            "wilson_z": 1.96,
            "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
            "force_formula": "(importance_raw / 5) * wilson_lower_bound",
            "force_components": force_components,
            "final_force": intrinsic_strength,
            "inclusion_reason": "compatible_review_evidence_available",
        },
    )


def build_empirical_arguments(
    hotel: HotelProfile,
    preferences: SessionPreferences,
) -> list[Argument]:
    """Build support then attack arguments in canonical aspect order."""
    arguments = []
    for preference in preferences.active_aspect_preferences:
        for stance in (Stance.SUPPORT, Stance.ATTACK):
            argument = _empirical_argument(
                hotel,
                aspect=preference.aspect,
                stance=stance,
                importance_raw=preference.importance_raw,
                normalized_weight=preference.normalized_weight,
            )
            if argument is not None:
                arguments.append(argument)
    return arguments


_FACT_ASPECTS = {
    "parking": "parking_voiture",
    "parking_voiture": "parking_voiture",
    "piscine": "piscine_spa_bien_etre",
    "pool": "piscine_spa_bien_etre",
    "swimming_pool": "piscine_spa_bien_etre",
    "spa": "piscine_spa_bien_etre",
    "gym": "piscine_spa_bien_etre",
    "wifi": "wifi_internet",
    "wifi_internet": "wifi_internet",
    "accessibilite": "accessibilite_batiment",
    "accessibility": "accessibilite_batiment",
    "accessibilite_batiment": "accessibilite_batiment",
    "accessible_facilities": "accessibilite_batiment",
    "air_conditioning": "climatisation_chauffage_temperature",
    "restaurant": "petit_dejeuner_restauration",
    "breakfast": "petit_dejeuner_restauration",
    "family_room": "chambre_taille_confort",
}


def build_structured_fact_arguments(
    hotel: HotelProfile,
    outcomes: tuple[ConstraintOutcome, ...],
) -> list[Argument]:
    arguments = []
    for index, outcome in enumerate(outcomes):
        constraint = outcome.constraint
        if (
            constraint.hard
            or constraint.normalized_weight <= 0.0
            or outcome.status is ConstraintStatus.UNKNOWN
        ):
            continue
        arg_type = (
            "support"
            if outcome.status is ConstraintStatus.SATISFIED
            else "attack"
        )
        importance_coefficient = constraint.normalized_weight
        direction = "satisfies" if arg_type == "support" else "conflicts with"
        target = constraint.canonical_target
        arguments.append(
            Argument(
                id=f"FACT::{index:02d}::{target}",
                arg_type=arg_type,
                text=(
                    f"Hotel metadata explicitly {direction} the constraint: "
                    f"{constraint.text}."
                ),
                evidence=list(outcome.evidence),
                aspect_effect=(
                    "present_preferred"
                    if arg_type == "support"
                    else "missing_preferred"
                ),
                used_aspects=[],
                target_item_name=hotel.metadata.name,
                argument_family=STRUCTURED_FACT,
                aspect=_FACT_ASPECTS.get(target),
                intrinsic_strength=importance_coefficient,
                importance_raw=constraint.importance_raw,
                normalized_weight=importance_coefficient,
                evidence_score=1.0,
                n_support=0,
                n_attack=0,
                n_neutral=0,
                review_sources=[],
                metadata={
                    "hotel_id": hotel.hotel_id,
                    "constraint_id": constraint.preference_ref,
                    "constraint_mode": constraint.mode,
                    "constraint_field": target,
                    "constraint_value": constraint.value,
                    "constraint_qualifiers": dict(constraint.qualifiers),
                    "constraint_status": outcome.status.value,
                    "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                    "force_formula": "importance_raw / 5",
                    "force_components": {
                        "importance_raw": constraint.importance_raw,
                        "importance_coefficient": importance_coefficient,
                        "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                        "deterministic_fact_confidence": 1.0,
                    },
                    "final_force": importance_coefficient,
                    "inclusion_reason": (
                        "known_soft_fact_satisfied"
                        if outcome.status is ConstraintStatus.SATISFIED
                        else "known_soft_fact_violated"
                    ),
                    "source_refs": [
                        source.get("source_ref")
                        for source in outcome.fact_sources
                        if source.get("source_ref")
                    ],
                    "fact_sources": [
                        dict(source) for source in outcome.fact_sources
                    ],
                },
            )
        )
    return arguments
