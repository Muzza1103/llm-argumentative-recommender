from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping

from .facility_ontology import normalize_facility_text


FACILITY_ASPECTS: Mapping[str, str] = MappingProxyType(
    {
        "parking": "parking_voiture",
        "wifi": "wifi_internet",
        "swimming_pool": "piscine_spa_bien_etre",
        "spa": "piscine_spa_bien_etre",
        "gym": "piscine_spa_bien_etre",
        "air_conditioning": "climatisation_chauffage_temperature",
        "accessible_facilities": "accessibilite_batiment",
        "restaurant": "petit_dejeuner_restauration",
        "breakfast": "petit_dejeuner_restauration",
        "family_room": "chambre_taille_confort",
    }
)


_QUALITY_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "wifi": (
            r"\bbon(?:ne)?\b",
            r"\bfiable\b",
            r"\brapid(?:e)?\b",
            r"\bstable\b",
            r"\bperformant(?:e)?\b",
            r"\bqualite\b",
            r"\bgood\b",
            r"\breliable\b",
            r"\bfast\b",
            r"\bhigh speed\b",
            r"\bstrong\b",
        ),
        "parking": (
            r"\bpratique\b",
            r"\bfacile d acces\b",
            r"\bdifficile d acces\b",
            r"\bsecurise\b",
            r"\bconvenient\b",
            r"\beasy to access\b",
            r"\bdifficult to access\b",
            r"\bsecure\b",
        ),
    }
)


_FACT_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "wifi": (
            r"\bavec\b.*\b(?:wi ?fi|connexion)\b",
            r"\bwith\b.*\b(?:wi ?fi|internet)\b",
            r"\b(?:wi ?fi|connexion|internet)\b.*\b(?:disponible|gratuit(?:e)?|free|available|included|utile|necessaire|indispensable|obligatoire|required)\b",
            r"\b(?:voudrais|veux|souhaite|besoin)\b.*\b(?:wi ?fi|connexion|internet)\b",
            r"\b(?:want|need|would like)\b.*\b(?:wi ?fi|internet)\b",
        ),
        "parking": (
            r"\bavec\b.*\bparking\b",
            r"\bwith\b.*\bparking\b",
            r"\bparking\b.*\b(?:disponible|gratuit|gratuitement|free|available|utile|necessaire|indispensable|obligatoire|required)\b",
            r"\b(?:voudrais|veux|souhaite|besoin)\b.*\bparking\b",
            r"\b(?:want|need|would like)\b.*\bparking\b",
        ),
    }
)


def _normalized_source(value: object) -> str:
    return normalize_facility_text(value) if isinstance(value, str) else ""


def _same_source_intention(left: object, right: object) -> bool:
    normalized_left = _normalized_source(left)
    normalized_right = _normalized_source(right)
    if not normalized_left or not normalized_right:
        return False
    return (
        normalized_left == normalized_right
        or normalized_left in normalized_right
        or normalized_right in normalized_left
    )


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _semantic_cues(capability: str, source_text: str) -> tuple[bool, bool]:
    normalized = _normalized_source(source_text)
    quality = _matches_any(
        normalized,
        _QUALITY_PATTERNS.get(capability, ()),
    )
    factual = _matches_any(
        normalized,
        _FACT_PATTERNS.get(capability, ()),
    )
    return quality, factual


def deduplicate_preference_intentions(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Remove duplicate scoring interpretations without mutating ``payload``.

    The input is the already schema- and ontology-validated Gemini payload.
    Only exact/nested source excerpts for the same canonical facility/aspect
    pair are compared.  Qualitative cues keep the aspect representation;
    factual presence/request cues keep the constraint representation.  A
    qualified fact plus an explicit quality cue is retained as two genuinely
    distinct intentions (for example, free and fast Wi-Fi).
    """
    aspects = {
        str(aspect): dict(details)
        for aspect, details in payload["aspect_preferences"].items()
    }
    constraints = [dict(row) for row in payload["constraints"]]
    uninterpreted = [dict(row) for row in payload["uninterpreted_items"]]
    decisions: list[dict[str, Any]] = []
    dropped_constraints: set[int] = set()

    # First collapse repeated factual interpretations of the same excerpt.
    for index, constraint in enumerate(constraints):
        if index in dropped_constraints or constraint.get("hard"):
            continue
        for other_index in range(index + 1, len(constraints)):
            other = constraints[other_index]
            if other_index in dropped_constraints or other.get("hard"):
                continue
            if (
                constraint.get("target") != other.get("target")
                or constraint.get("operator") != other.get("operator")
                or not _same_source_intention(
                    constraint.get("source_text"),
                    other.get("source_text"),
                )
            ):
                continue
            qualifiers = dict(constraint.get("qualifiers") or {})
            other_qualifiers = dict(other.get("qualifiers") or {})
            if qualifiers == other_qualifiers:
                drop_index = other_index
            elif all(
                other_qualifiers.get(key) == value
                for key, value in qualifiers.items()
            ):
                drop_index = index
            elif all(
                qualifiers.get(key) == value
                for key, value in other_qualifiers.items()
            ):
                drop_index = other_index
            else:
                continue
            dropped_constraints.add(drop_index)
            dropped = constraints[drop_index]
            decisions.append(
                {
                    "action": "drop_constraint",
                    "reason": "duplicate_factual_intention",
                    "constraint_id": dropped.get("constraint_id"),
                    "source_text": dropped.get("source_text"),
                }
            )
            if drop_index == index:
                break

    dropped_aspects: set[str] = set()
    for index, constraint in enumerate(constraints):
        if index in dropped_constraints:
            continue
        capability = constraint.get("target")
        aspect = FACILITY_ASPECTS.get(str(capability))
        if aspect is None or aspect not in aspects:
            continue
        aspect_source = aspects[aspect].get("source_text")
        constraint_source = constraint.get("source_text")
        if not _same_source_intention(aspect_source, constraint_source):
            continue

        combined_source = max(
            (str(aspect_source), str(constraint_source)),
            key=len,
        )
        quality_cue, factual_cue = _semantic_cues(
            str(capability),
            combined_source,
        )
        qualifiers = dict(constraint.get("qualifiers") or {})

        if constraint.get("hard"):
            if not quality_cue:
                dropped_aspects.add(aspect)
                decisions.append(
                    {
                        "action": "drop_aspect",
                        "reason": "hard_factual_intention",
                        "aspect": aspect,
                        "constraint_id": constraint.get("constraint_id"),
                        "source_text": combined_source,
                    }
                )
            continue

        if quality_cue and factual_cue and qualifiers:
            decisions.append(
                {
                    "action": "keep_both",
                    "reason": "distinct_qualified_fact_and_quality",
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": combined_source,
                }
            )
        elif quality_cue:
            dropped_constraints.add(index)
            decisions.append(
                {
                    "action": "drop_constraint",
                    "reason": "qualitative_request_is_aspect_only",
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": combined_source,
                }
            )
        else:
            # An ontology-validated facility constraint is the conservative
            # default when no explicit qualitative cue was found.
            dropped_aspects.add(aspect)
            decisions.append(
                {
                    "action": "drop_aspect",
                    "reason": (
                        "factual_request_is_constraint_only"
                        if factual_cue
                        else "validated_facility_constraint_is_authoritative"
                    ),
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": combined_source,
                }
            )

    cleaned = {
        "aspect_preferences": {
            aspect: details
            for aspect, details in aspects.items()
            if aspect not in dropped_aspects
        },
        "constraints": [
            constraint
            for index, constraint in enumerate(constraints)
            if index not in dropped_constraints
        ],
        "uninterpreted_items": uninterpreted,
    }
    return cleaned, tuple(decisions)
