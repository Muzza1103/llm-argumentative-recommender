from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .errors import HotelDataValidationError
from .models import HotelPolicy, HotelProfile


DEFAULT_FACILITY_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "hotel_facility_ontology.json"
)


def normalize_facility_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True, slots=True)
class FacilityMapping:
    facility_id: int
    expected_names: tuple[str, ...]
    capability: str
    qualifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CanonicalFacilityObservation:
    capability: str
    present: bool
    qualifiers: dict[str, Any]
    source_ref: str
    source_kind: str
    evidence: str
    facility_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "present": self.present,
            "qualifiers": dict(self.qualifiers),
            "source_ref": self.source_ref,
            "source_kind": self.source_kind,
            "evidence": self.evidence,
            "facility_id": self.facility_id,
        }


@dataclass(frozen=True, slots=True)
class CanonicalFacilityFact:
    capability: str
    status: str
    observations: tuple[CanonicalFacilityObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }


@dataclass(frozen=True, slots=True)
class FacilityNormalizationResult:
    facts: tuple[CanonicalFacilityFact, ...]
    unmapped_facilities: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]

    def get_fact(self, capability: str) -> CanonicalFacilityFact | None:
        return next(
            (fact for fact in self.facts if fact.capability == capability),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [fact.to_dict() for fact in self.facts],
            "unmapped_facilities": [
                dict(item) for item in self.unmapped_facilities
            ],
            "warnings": [dict(item) for item in self.warnings],
        }


class FacilityOntology:
    def __init__(
        self,
        *,
        schema_version: str,
        ontology_version: str,
        capabilities: dict[str, dict[str, Any]],
        operators: tuple[str, ...],
        mappings: tuple[FacilityMapping, ...],
    ) -> None:
        self.schema_version = schema_version
        self.ontology_version = ontology_version
        self.capabilities = capabilities
        self.operators = operators
        self.mappings = mappings
        self._mapping_by_id = {
            mapping.facility_id: mapping for mapping in mappings
        }
        self._ids_by_expected_name: dict[str, set[int]] = defaultdict(set)
        for mapping in mappings:
            for name in mapping.expected_names:
                self._ids_by_expected_name[
                    normalize_facility_text(name)
                ].add(mapping.facility_id)

    @classmethod
    def from_dict(cls, raw: object) -> "FacilityOntology":
        if not isinstance(raw, dict):
            raise HotelDataValidationError(
                "facility ontology must be a JSON object"
            )
        capabilities = raw.get("capabilities")
        if not isinstance(capabilities, dict) or not capabilities:
            raise HotelDataValidationError(
                "facility ontology requires capabilities"
            )
        operators = raw.get("operators")
        if not isinstance(operators, list) or not all(
            isinstance(operator, str) and operator
            for operator in operators
        ):
            raise HotelDataValidationError(
                "facility ontology operators must be non-empty strings"
            )

        mappings = []
        seen_ids: set[int] = set()
        raw_mappings = raw.get("facility_mappings")
        if not isinstance(raw_mappings, list):
            raise HotelDataValidationError(
                "facility_mappings must be a list"
            )
        for index, item in enumerate(raw_mappings):
            path = f"facility_mappings[{index}]"
            if not isinstance(item, dict):
                raise HotelDataValidationError("expected object", path=path)
            facility_id = item.get("facility_id")
            if isinstance(facility_id, bool) or not isinstance(
                facility_id, int
            ):
                raise HotelDataValidationError(
                    "facility_id must be an integer",
                    path=path,
                )
            if facility_id in seen_ids:
                raise HotelDataValidationError(
                    f"duplicate facility_id {facility_id}",
                    path=path,
                )
            seen_ids.add(facility_id)
            names = item.get("expected_names")
            if not isinstance(names, list) or not names or not all(
                isinstance(name, str) and name.strip() for name in names
            ):
                raise HotelDataValidationError(
                    "expected_names must contain non-empty strings",
                    path=path,
                )
            capability = item.get("capability")
            if capability not in capabilities:
                raise HotelDataValidationError(
                    f"unknown canonical capability {capability!r}",
                    path=path,
                )
            qualifiers = item.get("qualifiers", {})
            if not isinstance(qualifiers, dict):
                raise HotelDataValidationError(
                    "qualifiers must be an object",
                    path=path,
                )
            mappings.append(
                FacilityMapping(
                    facility_id=facility_id,
                    expected_names=tuple(name.strip() for name in names),
                    capability=capability,
                    qualifiers=dict(qualifiers),
                )
            )

        ontology = cls(
            schema_version=str(raw.get("schema_version", "")),
            ontology_version=str(raw.get("ontology_version", "")),
            capabilities={
                str(name): dict(specification)
                for name, specification in capabilities.items()
                if isinstance(specification, dict)
            },
            operators=tuple(operators),
            mappings=tuple(mappings),
        )
        for mapping in ontology.mappings:
            errors = ontology.qualifier_errors(
                mapping.capability,
                mapping.qualifiers,
                allow_unknown=True,
            )
            if errors:
                raise HotelDataValidationError(
                    "; ".join(errors),
                    path=f"facility_id={mapping.facility_id}",
                )
        return ontology

    @classmethod
    def load(cls, path: str | Path) -> "FacilityOntology":
        input_path = Path(path)
        try:
            raw = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HotelDataValidationError(
                f"invalid facility ontology JSON: {exc.msg}",
                path=str(input_path),
            ) from exc
        return cls.from_dict(raw)

    @property
    def capability_names(self) -> tuple[str, ...]:
        return tuple(self.capabilities)

    def get_mapping(self, facility_id: int) -> FacilityMapping | None:
        return self._mapping_by_id.get(facility_id)

    def expected_ids_for_name(self, name: str) -> tuple[int, ...]:
        return tuple(
            sorted(
                self._ids_by_expected_name.get(
                    normalize_facility_text(name),
                    set(),
                )
            )
        )

    def qualifier_errors(
        self,
        capability: str,
        qualifiers: object,
        *,
        allow_unknown: bool = False,
    ) -> list[str]:
        if capability not in self.capabilities:
            return [f"unsupported facility target: {capability}"]
        if not isinstance(qualifiers, dict):
            return ["qualifiers must be an object"]
        specification = self.capabilities[capability].get("qualifiers", {})
        errors = []
        for key, value in qualifiers.items():
            if key not in specification:
                errors.append(
                    f"unsupported qualifier {key!r} for {capability}"
                )
                continue
            allowed = specification[key]
            if not isinstance(allowed, list) or value not in allowed:
                errors.append(
                    f"unsupported value {value!r} for "
                    f"{capability}.{key}"
                )
            elif value == "unknown" and not allow_unknown:
                errors.append(
                    f"unknown is not a verifiable requested value for "
                    f"{capability}.{key}"
                )
        return errors

    def compact_prompt_contract(self) -> dict[str, Any]:
        return {
            "capabilities": {
                name: {
                    "qualifiers": {
                        qualifier: [
                            value
                            for value in values
                            if value != "unknown"
                        ]
                        for qualifier, values in specification.get(
                            "qualifiers", {}
                        ).items()
                    }
                }
                for name, specification in self.capabilities.items()
            },
            "operators": list(self.operators),
        }


@lru_cache(maxsize=1)
def load_default_facility_ontology() -> FacilityOntology:
    return FacilityOntology.load(DEFAULT_FACILITY_ONTOLOGY_PATH)


def _policy_source_ref(policy: HotelPolicy, index: int) -> str:
    source_id = policy.source_id if policy.source_id is not None else index
    return f"POLICY::{source_id}::description"


def _observation_from_policy(
    policy: HotelPolicy,
    index: int,
) -> CanonicalFacilityObservation | None:
    evidence = policy.description.strip()
    if not evidence:
        return None
    policy_type = normalize_facility_text(policy.policy_type or "")
    name = normalize_facility_text(policy.name)
    text = normalize_facility_text(
        " ".join(
            value
            for value in (
                policy.name,
                policy.description,
                policy.parking,
                policy.pets_allowed,
            )
            if isinstance(value, str) and value.strip()
        )
    )
    source_ref = _policy_source_ref(policy, index)

    if policy_type == "policy hotel parking" or name == "parking":
        negative = re.search(
            r"\b(?:no parking|parking (?:is )?not available|"
            r"parking unavailable)\b",
            text,
        )
        positive = re.search(
            r"\bparking\b.*\b(?:possible|available|provided|"
            r"height restrictions? apply)\b",
            text,
        )
        if negative is None and positive is None:
            return None
        qualifiers: dict[str, Any] = {}
        if re.search(r"\bfree\b", text):
            qualifiers["price"] = "free"
        elif re.search(r"\b(?:costs?|surcharge|fee|charges?)\b", text):
            qualifiers["price"] = "paid"
        if re.search(r"\b(?:on site|onsite)\b", text):
            qualifiers["location"] = "on_site"
        elif re.search(r"\b(?:nearby|offsite|off site)\b", text):
            qualifiers["location"] = "nearby"
        if re.search(r"\bprivate\b", text):
            qualifiers["access"] = "private"
        elif re.search(r"\bpublic\b", text):
            qualifiers["access"] = "public"
        for marker, parking_type in (
            ("garage", "garage"),
            ("valet", "valet"),
            ("street", "street"),
        ):
            if re.search(rf"\b{re.escape(marker)}\b", text):
                qualifiers["type"] = parking_type
                break
        return CanonicalFacilityObservation(
            capability="parking",
            present=negative is None,
            qualifiers=qualifiers,
            source_ref=source_ref,
            source_kind="policy",
            evidence=evidence,
        )

    if policy_type == "policy hotel internet" or name == "internet":
        if re.search(r"\bwi fi\b|\bwifi\b", text) is None:
            return None
        negative = re.search(
            r"\b(?:no wi fi|no wifi|wi fi is not available|"
            r"wifi is not available)\b",
            text,
        )
        positive = re.search(r"\bavailable\b", text)
        if negative is None and positive is None:
            return None
        qualifiers = {}
        if re.search(r"\bfree(?: of charge)?\b", text):
            qualifiers["price"] = "free"
        elif re.search(r"\b(?:costs?|surcharge|fee|charges?)\b", text):
            qualifiers["price"] = "paid"
        if re.search(r"\ball areas\b", text):
            qualifiers["coverage"] = "all_areas"
        elif re.search(r"\b(?:public|common) areas\b", text):
            qualifiers["coverage"] = "common_areas"
        elif re.search(r"\brooms?\b", text):
            qualifiers["coverage"] = "rooms"
        return CanonicalFacilityObservation(
            capability="wifi",
            present=negative is None,
            qualifiers=qualifiers,
            source_ref=source_ref,
            source_kind="policy",
            evidence=evidence,
        )

    if policy_type == "policy hotel pets" or name == "pets":
        if re.search(r"\bpets? (?:are )?not allowed\b", text):
            present = False
        elif re.search(r"\bpets? (?:are )?allowed\b", text):
            present = True
        else:
            return None
        return CanonicalFacilityObservation(
            capability="pets_allowed",
            present=present,
            qualifiers={},
            source_ref=source_ref,
            source_kind="policy",
            evidence=evidence,
        )
    return None


def normalize_hotel_facilities(
    hotel: HotelProfile,
    ontology: FacilityOntology | None = None,
) -> FacilityNormalizationResult:
    resolved_ontology = ontology or load_default_facility_ontology()
    observations: list[CanonicalFacilityObservation] = []
    unmapped = []
    warnings = []

    for facility in hotel.metadata.facilities:
        for facility_id in facility.facility_ids:
            mapping = resolved_ontology.get_mapping(facility_id)
            normalized_name = normalize_facility_text(facility.name)
            if mapping is None:
                expected_ids = resolved_ontology.expected_ids_for_name(
                    facility.name
                )
                if expected_ids:
                    warnings.append(
                        {
                            "code": "known_facility_name_with_unmapped_id",
                            "facility_id": facility_id,
                            "facility_name": facility.name,
                            "expected_ids": list(expected_ids),
                        }
                    )
                unmapped.append(
                    {
                        "facility_id": facility_id,
                        "facility_name": facility.name,
                    }
                )
                continue
            expected_names = {
                normalize_facility_text(name)
                for name in mapping.expected_names
            }
            if normalized_name not in expected_names:
                warnings.append(
                    {
                        "code": "facility_id_name_mismatch",
                        "facility_id": facility_id,
                        "facility_name": facility.name,
                        "expected_names": list(mapping.expected_names),
                    }
                )
                unmapped.append(
                    {
                        "facility_id": facility_id,
                        "facility_name": facility.name,
                    }
                )
                continue
            observations.append(
                CanonicalFacilityObservation(
                    capability=mapping.capability,
                    present=True,
                    qualifiers=dict(mapping.qualifiers),
                    source_ref=f"FACILITY_RAW::{facility_id}",
                    source_kind="facility",
                    evidence=facility.name,
                    facility_id=facility_id,
                )
            )

    for index, policy in enumerate(hotel.metadata.policies):
        observation = _observation_from_policy(policy, index)
        if observation is not None:
            observations.append(observation)

    unique_observations = []
    seen_observations = set()
    for observation in observations:
        key = (
            observation.capability,
            observation.present,
            tuple(sorted(observation.qualifiers.items())),
            observation.source_ref,
            observation.evidence,
        )
        if key in seen_observations:
            continue
        seen_observations.add(key)
        unique_observations.append(observation)

    grouped: dict[str, list[CanonicalFacilityObservation]] = defaultdict(list)
    for observation in unique_observations:
        grouped[observation.capability].append(observation)

    facts = []
    for capability in resolved_ontology.capability_names:
        capability_observations = grouped.get(capability, [])
        if not capability_observations:
            continue
        has_positive = any(item.present for item in capability_observations)
        has_negative = any(not item.present for item in capability_observations)
        if has_positive and has_negative:
            status = "contradictory"
            warnings.append(
                {
                    "code": "contradictory_capability_evidence",
                    "capability": capability,
                    "source_refs": [
                        item.source_ref for item in capability_observations
                    ],
                }
            )
        elif has_positive:
            status = "confirmed"
        else:
            status = "explicitly_absent"
        facts.append(
            CanonicalFacilityFact(
                capability=capability,
                status=status,
                observations=tuple(capability_observations),
            )
        )

    unique_unmapped = {
        (item["facility_id"], item["facility_name"]): item
        for item in unmapped
    }
    return FacilityNormalizationResult(
        facts=tuple(facts),
        unmapped_facilities=tuple(
            unique_unmapped[key] for key in sorted(unique_unmapped)
        ),
        warnings=tuple(warnings),
    )
