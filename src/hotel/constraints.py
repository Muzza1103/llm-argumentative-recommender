from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .facility_ontology import (
    FacilityNormalizationResult,
    FacilityOntology,
    load_default_facility_ontology,
    normalize_facility_text,
    normalize_hotel_facilities,
)
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
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "fact_sources": [dict(source) for source in self.fact_sources],
            "warnings": list(self.warnings),
        }


_FIELD_ALIASES = {
    "facilities": "facility",
    "parking_voiture": "parking",
    "pool": "swimming_pool",
    "piscine": "swimming_pool",
    "wifi_internet": "wifi",
    "wi-fi": "wifi",
    "accessibility": "accessible_facilities",
    "accessibilite": "accessible_facilities",
    "accessibilite_batiment": "accessible_facilities",
    "ville": "city",
}


def _expected_boolean(value: Any) -> bool | None:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = normalize_facility_text(value)
        if normalized in {"true", "yes", "required", "present", "available"}:
            return True
        if normalized in {"false", "no", "absent", "unavailable"}:
            return False
    return None


def _evaluate_city(
    hotel: HotelProfile,
    constraint: SessionConstraint,
) -> ConstraintOutcome:
    if constraint.operator not in {None, "equals"}:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city constraints support only the equals operator",
        )
    if not isinstance(constraint.value, str) or not constraint.value.strip():
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city constraint has no comparable expected value",
        )

    expected_city = normalize_facility_text(constraint.value)
    if not expected_city:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city constraint has no comparable expected value",
        )
    city = hotel.metadata.city
    if city is None or not normalize_facility_text(city):
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="city metadata is absent",
        )
    source_ref = "METADATA::city"
    return ConstraintOutcome(
        constraint=constraint,
        status=(
            ConstraintStatus.SATISFIED
            if expected_city == normalize_facility_text(city)
            else ConstraintStatus.VIOLATED
        ),
        reason="city metadata was compared with the requested city",
        evidence=(f"city: {city}",),
        fact_sources=(
            {
                "source_ref": source_ref,
                "source": "city",
                "value": city,
            },
        ),
    )


def _source_payload(observation) -> dict[str, Any]:
    return {
        "source_ref": observation.source_ref,
        "source": observation.source_kind,
        "capability": observation.capability,
        "present": observation.present,
        "qualifiers": dict(observation.qualifiers),
        "value": observation.evidence,
        "facility_id": observation.facility_id,
    }


def _evaluate_canonical_facility(
    constraint: SessionConstraint,
    capability: str,
    ontology: FacilityOntology,
    normalization: FacilityNormalizationResult,
) -> ConstraintOutcome:
    if constraint.operator not in {None, "present"}:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="facility constraints support only the present operator",
        )
    qualifier_errors = ontology.qualifier_errors(
        capability,
        constraint.qualifiers,
    )
    if qualifier_errors:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="; ".join(qualifier_errors),
        )
    expected_present = _expected_boolean(constraint.value)
    if expected_present is None:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="facility constraint value is not a supported boolean",
        )

    fact = normalization.get_fact(capability)
    if fact is None:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason=(
                "no explicit matching fact was found; an omitted facility "
                "is not evidence of absence"
            ),
        )
    observations = fact.observations
    evidence = tuple(item.evidence for item in observations[:3])
    sources = tuple(_source_payload(item) for item in observations[:3])
    if fact.status == "contradictory":
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="positive and negative structured sources conflict",
            evidence=evidence,
            fact_sources=sources,
            warnings=("contradictory_capability_evidence",),
        )

    positives = tuple(item for item in observations if item.present)
    negatives = tuple(item for item in observations if not item.present)
    if not expected_present:
        if positives:
            return ConstraintOutcome(
                constraint=constraint,
                status=ConstraintStatus.VIOLATED,
                reason="the facility is explicitly declared present",
                evidence=tuple(item.evidence for item in positives[:3]),
                fact_sources=tuple(
                    _source_payload(item) for item in positives[:3]
                ),
            )
        if negatives:
            return ConstraintOutcome(
                constraint=constraint,
                status=ConstraintStatus.SATISFIED,
                reason="the facility is explicitly declared absent",
                evidence=tuple(item.evidence for item in negatives[:3]),
                fact_sources=tuple(
                    _source_payload(item) for item in negatives[:3]
                ),
            )

    if negatives and not positives:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.VIOLATED,
            reason="the facility is explicitly declared absent",
            evidence=tuple(item.evidence for item in negatives[:3]),
            fact_sources=tuple(
                _source_payload(item) for item in negatives[:3]
            ),
        )
    if not positives:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="no positive facility observation is available",
        )

    requested_qualifiers = constraint.qualifiers
    if not requested_qualifiers:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.SATISFIED,
            reason="the canonical capability is explicitly confirmed",
            evidence=tuple(item.evidence for item in positives[:3]),
            fact_sources=tuple(
                _source_payload(item) for item in positives[:3]
            ),
        )

    matching = tuple(
        item
        for item in positives
        if all(
            key in item.qualifiers and item.qualifiers[key] == value
            for key, value in requested_qualifiers.items()
        )
    )
    if matching:
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.SATISFIED,
            reason="the capability and every requested qualifier are confirmed",
            evidence=tuple(item.evidence for item in matching[:3]),
            fact_sources=tuple(
                _source_payload(item) for item in matching[:3]
            ),
        )

    return ConstraintOutcome(
        constraint=constraint,
        status=ConstraintStatus.UNKNOWN,
        reason=(
            "the capability is present but the requested qualifiers are not "
            "fully confirmed; non-matching observations do not prove that "
            "another valid option is absent"
        ),
        evidence=tuple(item.evidence for item in positives[:3]),
        fact_sources=tuple(
            _source_payload(item) for item in positives[:3]
        ),
        warnings=("qualifier_not_fully_verified",),
    )


def _evaluate_generic_facility(
    hotel: HotelProfile,
    constraint: SessionConstraint,
) -> ConstraintOutcome:
    if not isinstance(constraint.value, str) or not constraint.value.strip():
        return ConstraintOutcome(
            constraint=constraint,
            status=ConstraintStatus.UNKNOWN,
            reason="generic facility constraints require an exact facility name",
        )
    expected = normalize_facility_text(constraint.value)
    for facility in hotel.metadata.facilities:
        if normalize_facility_text(facility.name) == expected:
            return ConstraintOutcome(
                constraint=constraint,
                status=ConstraintStatus.SATISFIED,
                reason="the exact requested facility is explicitly declared",
                evidence=(facility.name,),
                fact_sources=(
                    {
                        "source_ref": (
                            f"FACILITY_RAW::{facility.facility_ids[0]}"
                            if facility.facility_ids
                            else "FACILITY_RAW::unknown"
                        ),
                        "source": "facility",
                        "value": facility.name,
                    },
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
    *,
    ontology: FacilityOntology | None = None,
    normalization: FacilityNormalizationResult | None = None,
) -> ConstraintOutcome:
    resolved_ontology = ontology or load_default_facility_ontology()
    resolved_normalization = normalization or normalize_hotel_facilities(
        hotel,
        resolved_ontology,
    )
    field = _FIELD_ALIASES.get(
        constraint.canonical_target,
        constraint.canonical_target,
    )
    if field == "city":
        return _evaluate_city(hotel, constraint)
    if field == "facility":
        return _evaluate_generic_facility(hotel, constraint)
    if field in resolved_ontology.capabilities:
        return _evaluate_canonical_facility(
            constraint,
            field,
            resolved_ontology,
            resolved_normalization,
        )
    return ConstraintOutcome(
        constraint=constraint,
        status=ConstraintStatus.UNKNOWN,
        reason=f"unsupported factual field: {constraint.canonical_target}",
    )


def evaluate_constraints(
    hotel: HotelProfile,
    constraints: tuple[SessionConstraint, ...],
    *,
    ontology: FacilityOntology | None = None,
) -> tuple[ConstraintOutcome, ...]:
    resolved_ontology = ontology or load_default_facility_ontology()
    normalization = normalize_hotel_facilities(hotel, resolved_ontology)
    return tuple(
        evaluate_constraint(
            hotel,
            constraint,
            ontology=resolved_ontology,
            normalization=normalization,
        )
        for constraint in constraints
    )
