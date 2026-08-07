from __future__ import annotations

import re
from dataclasses import dataclass
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


_CAPABILITY_TERMS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "wifi": (
            r"\bwi ?fi\b",
            r"\binternet\b",
            r"\bconnexion\b",
            r"\bconnection\b",
        ),
        "parking": (r"\bparking\b", r"\bcar park\b", r"\bgarage\b"),
        "swimming_pool": (r"\bpiscine\b", r"\bpool\b"),
        "spa": (r"\bspa\b", r"\bwellness cent(?:er|re)\b"),
        "gym": (r"\bgym\b", r"\bfitness\b", r"\bsalle de sport\b"),
        "air_conditioning": (r"\bclimatisation\b", r"\bair conditioning\b"),
        "accessible_facilities": (
            r"\baccessible\b",
            r"\baccessibilite\b",
            r"\bwheelchair\b",
        ),
        "restaurant": (r"\brestaurant\b",),
        "breakfast": (r"\bpetit dejeuner\b", r"\bbreakfast\b"),
        "family_room": (r"\bchambre familiale\b", r"\bfamily room\b"),
    }
)


_COMMON_QUALITY_PATTERNS = (
    r"\bbon(?:ne)?\b",
    r"\bmauvais(?:e)?\b",
    r"\bexcellent(?:e)?\b",
    r"\bqualite\b",
    r"\bpratique\b",
    r"\bgood\b",
    r"\bbad\b",
    r"\bexcellent\b",
    r"\bquality\b",
    r"\bconvenient\b",
)


_QUALITY_PATTERNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "wifi": (
            r"\bbon(?:ne)?\b",
            r"\bfiable\b",
            r"\brapid(?:e)?\b",
            r"\blent(?:e)?\b",
            r"\bstable\b",
            r"\bperformant(?:e)?\b",
            r"\bqualite\b",
            r"\bgood\b",
            r"\breliable\b",
            r"\bfast\b",
            r"\bslow\b",
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


_AVAILABILITY_PATTERNS = (
    r"\bdisponible\b",
    r"\bdisponibilite\b",
    r"\bavailable\b",
    r"\bincluded\b",
)
_REQUEST_PATTERNS = (
    r"\bvoudrais\b",
    r"\bveux\b",
    r"\bsouhaite\b",
    r"\bbesoin\b",
    r"\bwant\b",
    r"\bwould like\b",
    r"\bneed\b",
)
_OPTIONAL_PATTERNS = (
    r"\bserait utile\b",
    r"\bde preference\b",
    r"\bsi possible\b",
    r"\bidealement\b",
    r"\bwould be useful\b",
    r"\bnice to have\b",
    r"\bpreferably\b",
    r"\bif possible\b",
    r"\bideally\b",
)
_STRONG_SOFT_PATTERNS = (
    r"\btres important(?:e)?\b",
    r"\bimportant(?:e)?\b",
    r"\bprioritaire\b",
    r"\bvery important\b",
    r"\breally important\b",
    r"\bhigh priority\b",
)
_ABSOLUTE_NECESSITY_PATTERNS = (
    r"\bindispensable\b",
    r"\bobligatoire\b",
    r"\bimperatif\b",
    r"\babsolument necessaire\b",
    r"\bnecessaire\b",
    r"\bmust[ -]?have\b",
    r"\bmandatory\b",
    r"\brequired\b",
    r"\bstrictly required\b",
    r"\bnon[ -]?negotiable\b",
    r"\bcannot (?:do|stay|travel) without\b",
)
_PRICE_QUALIFIED_CAPABILITIES = frozenset(
    {"parking", "wifi", "spa", "gym", "breakfast", "airport_shuttle"}
)
_FREE_PATTERNS = (
    r"\bgratuit(?:e|ement)?\b",
    r"\bsans frais\b",
    r"\bfree\b",
    r"\bcomplimentary\b",
)
_PAID_PATTERNS = (
    r"\bpayant(?:e)?\b",
    r"\bavec supplement\b",
    r"\bpaid\b",
    r"\bsurcharge\b",
)


@dataclass(frozen=True, slots=True)
class _SemanticCues:
    quality: bool
    factual: bool
    qualified_fact: bool
    absolute_necessity: bool


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


def _mentions_capability(capability: str, source_text: object) -> bool:
    normalized = _normalized_source(source_text)
    patterns = _CAPABILITY_TERMS.get(
        capability,
        (rf"\b{re.escape(capability.replace('_', ' '))}\b",),
    )
    return _matches_any(normalized, patterns)


def _intent_scope(capability: str, source_text: object) -> str:
    normalized = _normalized_source(source_text)
    if not normalized:
        return ""
    clauses = [
        clause.strip()
        for clause in re.split(r"[.!?;,\n]+", normalized)
        if clause.strip()
    ]
    matching = [
        clause
        for clause in clauses
        if _mentions_capability(capability, clause)
    ]
    return min(matching, key=len) if matching else normalized


def _semantic_cues(capability: str, source_text: object) -> _SemanticCues:
    scoped = _intent_scope(capability, source_text)
    quality = _matches_any(
        scoped,
        _QUALITY_PATTERNS.get(capability, ()) + _COMMON_QUALITY_PATTERNS,
    )
    qualified_fact = _matches_any(scoped, _FREE_PATTERNS + _PAID_PATTERNS)
    absolute = _matches_any(scoped, _ABSOLUTE_NECESSITY_PATTERNS)
    explicit_availability = _matches_any(scoped, _AVAILABILITY_PATTERNS)
    optional_presence = _matches_any(scoped, _OPTIONAL_PATTERNS)
    requested_presence = _matches_any(scoped, _REQUEST_PATTERNS)
    connector = re.search(r"\b(?:avec|with)\b", scoped) is not None

    # "avec un bon Wi-Fi" uses a connector to introduce a quality, not a
    # second availability intention. An explicit qualifier or availability
    # cue remains a genuinely separate fact (for example free and fast Wi-Fi).
    factual = (
        qualified_fact
        or absolute
        or explicit_availability
        or optional_presence
        or ((requested_presence or connector) and not quality)
    )
    return _SemanticCues(
        quality=quality,
        factual=factual,
        qualified_fact=qualified_fact,
        absolute_necessity=absolute,
    )


def calibrate_importance_from_source(
    source_text: str,
    *,
    hard: bool = False,
    eligibility: bool = False,
) -> float:
    """Apply the deterministic 2/3/4/5 importance convention."""
    normalized = _normalized_source(source_text)
    if hard or eligibility or _matches_any(
        normalized,
        _ABSOLUTE_NECESSITY_PATTERNS,
    ):
        return 5.0
    if _matches_any(normalized, _OPTIONAL_PATTERNS):
        return 2.0
    if _matches_any(normalized, _STRONG_SOFT_PATTERNS):
        return 4.0
    return 3.0


def _inferred_qualifiers(capability: str, source_text: object) -> dict[str, Any]:
    if capability not in _PRICE_QUALIFIED_CAPABILITIES:
        return {}
    scoped = _intent_scope(capability, source_text)
    if _matches_any(scoped, _FREE_PATTERNS):
        return {"price": "free"}
    if _matches_any(scoped, _PAID_PATTERNS):
        return {"price": "paid"}
    return {}


def _aspect_intent_scope(aspect: str, source_text: str) -> str:
    capabilities = [
        capability
        for capability, mapped_aspect in FACILITY_ASPECTS.items()
        if mapped_aspect == aspect
        and _mentions_capability(capability, source_text)
    ]
    if len(capabilities) == 1:
        return _intent_scope(capabilities[0], source_text)
    return source_text


def _next_constraint_id(existing_ids: set[str], capability: str) -> str:
    prefix = f"deterministic_{capability}"
    index = 1
    while f"{prefix}_{index:02d}" in existing_ids:
        index += 1
    result = f"{prefix}_{index:02d}"
    existing_ids.add(result)
    return result


def _constraint_is_hard(constraint: Mapping[str, Any]) -> bool:
    target = str(constraint.get("target", ""))
    target_type = constraint.get(
        "target_type",
        "facility" if target in FACILITY_ASPECTS else None,
    )
    operator = constraint.get("operator")
    if target_type == "metadata" and target == "city" and operator == "equals":
        return True
    if target_type != "facility":
        return False
    return _semantic_cues(
        target,
        constraint.get("source_text"),
    ).absolute_necessity


def _ensure_aspect(
    aspects: dict[str, dict[str, Any]],
    *,
    aspect: str,
    capability: str,
    source_text: str,
    decisions: list[dict[str, Any]],
) -> None:
    importance = calibrate_importance_from_source(
        _intent_scope(capability, source_text)
    )
    current = aspects.get(aspect)
    if current is None:
        aspects[aspect] = {
            "importance_raw": importance,
            "source_text": source_text,
        }
        decisions.append(
            {
                "action": "create_aspect",
                "reason": "source_text_expresses_subjective_quality",
                "aspect": aspect,
                "source_text": source_text,
            }
        )
        return

    current_source = str(current.get("source_text", ""))
    current_cues = _semantic_cues(capability, current_source)
    if not current_cues.quality:
        current["source_text"] = source_text
    current["importance_raw"] = max(
        float(current.get("importance_raw", 0.0)),
        importance,
    )


def _new_constraint_from_aspect(
    *,
    capability: str,
    source_text: str,
    existing_ids: set[str],
) -> dict[str, Any]:
    cues = _semantic_cues(capability, source_text)
    hard = cues.absolute_necessity
    return {
        "constraint_id": _next_constraint_id(existing_ids, capability),
        "target_type": "facility",
        "target": capability,
        "operator": "present",
        "qualifiers": _inferred_qualifiers(capability, source_text),
        "value": None,
        "hard": hard,
        "importance_raw": calibrate_importance_from_source(
            _intent_scope(capability, source_text),
            hard=hard,
        ),
        "source_text": source_text,
        "text": source_text,
    }


def deduplicate_preference_intentions(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Correct and deduplicate Gemini interpretations from validated excerpts.

    The function never scans the full request to add an unrelated aspect. It
    only recalibrates or converts an entry whose validated ``source_text``
    explicitly contains the corresponding capability and semantic cue.
    """
    aspects = {
        str(aspect): dict(details)
        for aspect, details in payload["aspect_preferences"].items()
    }
    constraints = [dict(row) for row in payload["constraints"]]
    uninterpreted = [dict(row) for row in payload["uninterpreted_items"]]
    decisions: list[dict[str, Any]] = []
    dropped_constraints: set[int] = set()
    dropped_aspects: set[str] = set()
    existing_ids = {
        str(row.get("constraint_id"))
        for row in constraints
        if row.get("constraint_id")
    }

    # Importance is local to each validated excerpt; Gemini's numeric proposal
    # is retained in the trace only when it differs from the calibrated value.
    for aspect, details in aspects.items():
        source_text = str(details.get("source_text", ""))
        proposed = details.get("importance_raw")
        calibrated = calibrate_importance_from_source(
            _aspect_intent_scope(aspect, source_text)
        )
        details["importance_raw"] = calibrated
        if proposed != calibrated:
            decisions.append(
                {
                    "action": "calibrate_importance",
                    "intent": aspect,
                    "proposed": proposed,
                    "calibrated": calibrated,
                    "source_text": source_text,
                }
            )

    for index, constraint in enumerate(constraints):
        source_text = str(constraint.get("source_text", ""))
        proposed_hard = bool(constraint.get("hard"))
        hard = _constraint_is_hard(constraint)
        constraint["hard"] = hard
        proposed_importance = constraint.get("importance_raw")
        target = str(constraint.get("target", ""))
        calibration_source = (
            _intent_scope(target, source_text)
            if constraint.get("target_type") == "facility"
            else source_text
        )
        calibrated = calibrate_importance_from_source(
            calibration_source,
            hard=hard,
            eligibility=(
                constraint.get("target_type") == "metadata"
                and constraint.get("target") == "city"
            ),
        )
        constraint["importance_raw"] = calibrated
        if proposed_hard != hard:
            decisions.append(
                {
                    "action": "correct_constraint_mode",
                    "constraint_id": constraint.get("constraint_id"),
                    "proposed_hard": proposed_hard,
                    "hard": hard,
                    "source_text": source_text,
                }
            )
        if proposed_importance is not None and proposed_importance != calibrated:
            decisions.append(
                {
                    "action": "calibrate_importance",
                    "intent": constraint.get("constraint_id"),
                    "proposed": proposed_importance,
                    "calibrated": calibrated,
                    "source_text": source_text,
                }
            )

        capability = target
        target_type = constraint.get(
            "target_type",
            "facility" if capability in FACILITY_ASPECTS else None,
        )
        if (
            target_type == "metadata"
            and capability == "city"
            and "localisation_transport" in aspects
            and _same_source_intention(
                aspects["localisation_transport"].get("source_text"),
                source_text,
            )
        ):
            dropped_aspects.add("localisation_transport")
            decisions.append(
                {
                    "action": "drop_aspect",
                    "reason": "eligibility_location_is_constraint_only",
                    "aspect": "localisation_transport",
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": source_text,
                }
            )
        if target_type != "facility":
            continue
        inferred = _inferred_qualifiers(capability, source_text)
        qualifiers = dict(constraint.get("qualifiers") or {})
        for key, value in inferred.items():
            qualifiers[key] = value
        constraint["qualifiers"] = qualifiers

        aspect = FACILITY_ASPECTS.get(capability)
        if aspect is None:
            continue
        cues = _semantic_cues(capability, source_text)
        matching_aspect = (
            aspect in aspects
            and _same_source_intention(
                aspects[aspect].get("source_text"),
                source_text,
            )
        )
        if cues.quality and not cues.factual:
            _ensure_aspect(
                aspects,
                aspect=aspect,
                capability=capability,
                source_text=source_text,
                decisions=decisions,
            )
            dropped_constraints.add(index)
            decisions.append(
                {
                    "action": "drop_constraint",
                    "reason": "qualitative_request_is_aspect_only",
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": source_text,
                }
            )
        elif cues.quality and cues.factual:
            _ensure_aspect(
                aspects,
                aspect=aspect,
                capability=capability,
                source_text=source_text,
                decisions=decisions,
            )
            decisions.append(
                {
                    "action": "keep_both",
                    "reason": "distinct_fact_and_quality",
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": source_text,
                }
            )
        elif matching_aspect:
            dropped_aspects.add(aspect)
            decisions.append(
                {
                    "action": "drop_aspect",
                    "reason": "factual_request_is_constraint_only",
                    "aspect": aspect,
                    "constraint_id": constraint.get("constraint_id"),
                    "source_text": source_text,
                }
            )

    # Correct the inverse Gemini error: an aspect entry that clearly asks only
    # for facility presence becomes a constraint. Ambiguous aspect text stays
    # untouched instead of inventing a facility or an additional preference.
    for aspect, details in list(aspects.items()):
        if aspect in dropped_aspects:
            continue
        source_text = str(details.get("source_text", ""))
        candidates = [
            capability
            for capability, mapped_aspect in FACILITY_ASPECTS.items()
            if mapped_aspect == aspect
            and _mentions_capability(capability, source_text)
        ]
        if len(candidates) != 1:
            continue
        capability = candidates[0]
        cues = _semantic_cues(capability, source_text)
        if not cues.factual:
            continue
        matching_constraint = next(
            (
                index
                for index, constraint in enumerate(constraints)
                if index not in dropped_constraints
                and constraint.get("target") == capability
                and _same_source_intention(
                    constraint.get("source_text"),
                    source_text,
                )
            ),
            None,
        )
        if matching_constraint is None:
            generated = _new_constraint_from_aspect(
                capability=capability,
                source_text=source_text,
                existing_ids=existing_ids,
            )
            constraints.append(generated)
            decisions.append(
                {
                    "action": "create_constraint",
                    "reason": "source_text_expresses_factual_presence",
                    "aspect": aspect,
                    "constraint_id": generated["constraint_id"],
                    "source_text": source_text,
                }
            )
        if not cues.quality:
            dropped_aspects.add(aspect)
            decisions.append(
                {
                    "action": "drop_aspect",
                    "reason": "factual_request_is_constraint_only",
                    "aspect": aspect,
                    "source_text": source_text,
                }
            )

    # Collapse repeated factual interpretations after all deterministic
    # conversions. Prefer the entry with the richer compatible qualifier set.
    for index, constraint in enumerate(constraints):
        if index in dropped_constraints:
            continue
        for other_index in range(index + 1, len(constraints)):
            if other_index in dropped_constraints:
                continue
            other = constraints[other_index]
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
            decisions.append(
                {
                    "action": "drop_constraint",
                    "reason": "duplicate_factual_intention",
                    "constraint_id": constraints[drop_index].get("constraint_id"),
                    "source_text": constraints[drop_index].get("source_text"),
                }
            )
            if drop_index == index:
                break

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
