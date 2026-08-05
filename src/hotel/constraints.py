from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import HotelProfile
from .preferences import SessionConstraint


class ConstraintStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConstraintOutcome:
    constraint: SessionConstraint
    status: ConstraintStatus
    reason: str
    evidence: tuple[str, ...] = ()
    fact_sources: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "fact_sources": [dict(source) for source in self.fact_sources],
        }


_FIELD_ALIASES = {
    "facilities": "facility",
    "parking_voiture": "parking",
    "pool": "piscine",
    "swimming_pool": "piscine",
    "wifi_internet": "wifi",
    "wi-fi": "wifi",
    "accessibility": "accessibilite",
    "accessibilite_batiment": "accessibilite",
    "ville": "city",
}

_FACILITY_TERMS = {
    "parking": ("parking", "car park", "garage"),
    "piscine": ("swimming pool", "pool"),
    "spa": ("spa",),
    "wifi": ("wifi", "wi fi", "wireless internet"),
    "accessibilite": (
        "wheelchair accessible",
        "wheelchair access",
        "accessible room",
        "disabled access",
    ),
}

_FALSE_TEXTS = {
    "false",
    "no",
    "none",
    "not available",
    "unavailable",
    "no parking",
    "parking not available",
    "parking is not available",
}


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def _expected_boolean(value: Any) -> bool | None:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = _normalize(value)
        if normalized in {"true", "yes", "required", "present", "available"}:
            return True
        if normalized in {"false", "no", "absent", "unavailable"}:
            return False
    return None


def _status_for_observation(
    *,
    expected_present: bool,
    observed_present: bool,
) -> ConstraintStatus:
    if expected_present == observed_present:
        return ConstraintStatus.SATISFIED
    return ConstraintStatus.VIOLATED


def _evaluate_city(
    hotel: HotelProfile,
    constraint: SessionConstraint,
) -> ConstraintOutcome:
    if not isinstance(constraint.value, str) or not constraint.value.strip():
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city constraint has no comparable expected value",
        )

    expected_city = _normalize(constraint.value)
    if not expected_city:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city constraint has no comparable expected value",
        )

    city = hotel.metadata.city
    if city is None or not _normalize(city):
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city metadata is absent",
        )

    source = ({"source": "city", "value": city},)
    evidence = (f"city: {city}",)
    actual_city = _normalize(city)
    return ConstraintOutcome(
        constraint=constraint,
        status=(
            ConstraintStatus.SATISFIED
            if expected_city == actual_city
            else ConstraintStatus.VIOLATED
        ),
        reason="city metadata was compared with the requested city",
        evidence=evidence,
        fact_sources=source,
    )


def _matching_facilities(
    hotel: HotelProfile,
    terms: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_terms = tuple(
        normalized_term
        for term in terms
        if (normalized_term := _normalize(term))
    )
    matches = []
    for facility in hotel.metadata.facilities:
        normalized_name = _normalize(facility.name)
        if any(
            re.search(
                rf"(?:^|\s){re.escape(term)}(?:$|\s)",
                normalized_name,
            )
            is not None
            for term in normalized_terms
        ):
            matches.append(facility.name)
    return tuple(matches)


def _parking_policy_observation(
    hotel: HotelProfile,
) -> tuple[bool, str, dict[str, Any]] | None:
    for policy in hotel.metadata.policies:
        if policy.parking is None:
            continue
        normalized = _normalize(policy.parking)
        present = not (
            normalized in _FALSE_TEXTS
            or normalized.startswith("no ")
            or "not available" in normalized
            or "unavailable" in normalized
        )
        evidence = f"parking policy: {policy.parking}"
        source = {
            "source": "policy.parking",
            "policy_name": policy.name,
            "value": policy.parking,
        }
        return present, evidence, source
    return None


def _evaluate_named_facility(
    hotel: HotelProfile,
    constraint: SessionConstraint,
    field: str,
) -> ConstraintOutcome:
    expected_present = _expected_boolean(constraint.value)
    if expected_present is None:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="facility constraint value is not a supported boolean",
        )

    matches = _matching_facilities(hotel, _FACILITY_TERMS[field])
    policy_observation = (
        _parking_policy_observation(hotel)
        if field == "parking"
        else None
    )
    if (
        matches
        and policy_observation is not None
        and policy_observation[0] is False
    ):
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="facility and policy metadata conflict",
            evidence=(
                f"facility: {matches[0]}",
                policy_observation[1],
            ),
            fact_sources=(
                {"source": "facility", "value": matches[0]},
                policy_observation[2],
            ),
        )
    if matches:
        evidence = tuple(f"facility: {name}" for name in matches[:2])
        sources = tuple(
            {"source": "facility", "value": name}
            for name in matches[:2]
        )
        return ConstraintOutcome(
            constraint=constraint,
            status=_status_for_observation(
                expected_present=expected_present,
                observed_present=True,
            ),
            reason="a matching facility is explicitly declared",
            evidence=evidence,
            fact_sources=sources,
        )

    if policy_observation is not None:
        observed_present, evidence, source = policy_observation
        return ConstraintOutcome(
            constraint=constraint,
            status=_status_for_observation(
                expected_present=expected_present,
                observed_present=observed_present,
            ),
            reason="an explicit parking policy was found",
            evidence=(evidence,),
            fact_sources=(source,),
        )

    return ConstraintOutcome(
        constraint=constraint,
        status=ConstraintStatus.UNKNOWN,
        reason=(
            "no explicit matching fact was found; an omitted facility is not "
            "evidence of absence"
        ),
    )


def _evaluate_generic_facility(
    hotel: HotelProfile,
    constraint: SessionConstraint,
) -> ConstraintOutcome:
    if not isinstance(constraint.value, str) or not constraint.value.strip():
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="generic facility constraints require a facility name",
        )
    expected = _normalize(constraint.value)
    for facility in hotel.metadata.facilities:
        if _normalize(facility.name) == expected:
            return ConstraintOutcome(
                constraint=constraint,
                status=ConstraintStatus.SATISFIED,
                reason="the requested facility is explicitly declared",
                evidence=(f"facility: {facility.name}",),
                fact_sources=(
                    {"source": "facility", "value": facility.name},
                ),
            )
    return ConstraintOutcome(
        constraint=constraint,
        status=ConstraintStatus.UNKNOWN,
        reason=(
            "the facility is not explicitly listed; the list is treated as "
            "non-exhaustive"
        ),
    )


def evaluate_constraint(
    hotel: HotelProfile,
    constraint: SessionConstraint,
) -> ConstraintOutcome:
    field = _FIELD_ALIASES.get(constraint.field, constraint.field)
    if field == "city":
        return _evaluate_city(hotel, constraint)
    if field == "facility":
        return _evaluate_generic_facility(hotel, constraint)
    if field in _FACILITY_TERMS:
        return _evaluate_named_facility(hotel, constraint, field)
    return ConstraintOutcome(
        constraint=constraint,
        status=ConstraintStatus.UNKNOWN,
        reason=f"unsupported factual field: {constraint.field}",
    )


def evaluate_constraints(
    hotel: HotelProfile,
    constraints: tuple[SessionConstraint, ...],
) -> tuple[ConstraintOutcome, ...]:
    return tuple(
        evaluate_constraint(hotel, constraint)
        for constraint in constraints
    )
