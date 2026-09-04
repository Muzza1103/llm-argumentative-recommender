from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.argumentation.schema import Argument

from .argument_builder import (
    EMPIRICAL_ASPECT,
    STRUCTURED_FACT,
    build_empirical_arguments,
    build_structured_fact_arguments,
)
from .constraints import ConstraintOutcome, ConstraintStatus, evaluate_constraints
from .errors import HotelHybridValidationError
from .facility_ontology import (
    FacilityNormalizationResult,
    FacilityOntology,
    load_default_facility_ontology,
    normalize_hotel_facilities,
)
from .models import HotelProfile
from .preferences import ABSOLUTE_5_WEIGHTING_METHOD, SessionPreferences


HYBRID_ARGUMENT_KINDS = frozenset(
    {"opinion", "fact", "contextual", "tradeoff", "summary"}
)
HYBRID_ARGUMENT_TYPES = frozenset({"support", "attack"})
EXPLANATORY_KINDS = frozenset({"contextual", "tradeoff", "summary"})
MAX_HYBRID_ARGUMENTS = 8
MAX_HYBRID_RELATIONS = 8
MAX_HYBRID_PREFERENCE_REFS = 5
MAX_HYBRID_SOURCE_REFS = 8
MAX_HYBRID_SCORING_UNIT_REFS = 1
MAX_FACILITY_SOURCES_PER_CAPABILITY = 3


@runtime_checkable
class HybridArgumentGenerator(Protocol):
    def propose_arguments(
        self,
        *,
        preferences: SessionPreferences,
        hotel_context: dict[str, Any],
        authorized_sources: list[dict[str, Any]],
        scoring_units: list[dict[str, Any]],
        constraint_outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return one closed JSON-like proposal batch."""
        ...


@dataclass(frozen=True, slots=True)
class AuthorizedSource:
    source_id: str
    kind: str
    payload: dict[str, Any]
    allowed_types: tuple[str, ...]
    scoring_unit_ids: tuple[str, ...] = ()
    evidence_text: str | None = None
    hard_constraint: bool = False
    unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "allowed_types": list(self.allowed_types),
            "scoring_unit_ids": list(self.scoring_unit_ids),
            "evidence_text": self.evidence_text,
            "hard_constraint": self.hard_constraint,
            "unknown": self.unknown,
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("hard_constraint")
        payload.pop("unknown")
        return payload


@dataclass(frozen=True, slots=True)
class ScoringUnit:
    unit_id: str
    kind: str
    arg_type: str
    intrinsic_strength: float
    preference_refs: tuple[str, ...]
    source_refs: tuple[str, ...]
    atomic_argument: Argument = field(repr=False, compare=False)

    def to_dict(self, *, include_strength: bool = True) -> dict[str, Any]:
        metadata = self.atomic_argument.metadata
        payload = {
            "scoring_unit_id": self.unit_id,
            "kind": self.kind,
            "type": self.arg_type,
            "preference_refs": list(self.preference_refs),
            "intent_ref": (
                self.preference_refs[0] if self.preference_refs else None
            ),
            "source_refs": list(self.source_refs),
        }
        if include_strength:
            payload.update(
                {
                    "intrinsic_strength": self.intrinsic_strength,
                    "importance_raw": self.atomic_argument.importance_raw,
                    "normalized_weight": self.atomic_argument.normalized_weight,
                    "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                    "confidence_factor": self.atomic_argument.evidence_score,
                    "force_formula": metadata.get("force_formula"),
                    "force_components": dict(
                        metadata.get("force_components", {})
                    ),
                    "final_force": self.intrinsic_strength,
                    "availability_reason": metadata.get("inclusion_reason"),
                    "availability_status": "available",
                    "weight_active": True,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class PreparedHybridContext:
    hotel_context: dict[str, Any]
    sources: tuple[AuthorizedSource, ...]
    scoring_units: tuple[ScoringUnit, ...]
    constraint_outcomes: tuple[ConstraintOutcome, ...]
    facility_normalization: FacilityNormalizationResult

    def source_lookup(self) -> dict[str, AuthorizedSource]:
        return {source.source_id: source for source in self.sources}

    def unit_lookup(self) -> dict[str, ScoringUnit]:
        return {unit.unit_id: unit for unit in self.scoring_units}

    def prompt_sources(self) -> list[dict[str, Any]]:
        return [source.to_prompt_dict() for source in self.sources]

    def prompt_units(self) -> list[dict[str, Any]]:
        return [
            unit.to_dict(include_strength=False) for unit in self.scoring_units
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotel_context": dict(self.hotel_context),
            "authorized_sources": [
                source.to_dict() for source in self.sources
            ],
            "available_scoring_units": [
                unit.to_dict() for unit in self.scoring_units
            ],
            "constraint_outcomes": [
                outcome.to_dict() for outcome in self.constraint_outcomes
            ],
            "facility_normalization": self.facility_normalization.to_dict(),
        }


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", value)
    return token.strip("_") or "none"


def _qualifier_token(qualifiers: dict[str, Any]) -> str:
    if not qualifiers:
        return "unqualified"
    return ",".join(
        f"{_safe_token(str(key))}={_safe_token(str(value).lower())}"
        for key, value in sorted(qualifiers.items())
    )


def _description_excerpt(description: str, limit: int = 1200) -> str:
    compact = re.sub(r"\s+", " ", description).strip()
    if len(compact) <= limit:
        return compact
    candidate = compact[:limit]
    if " " in candidate:
        candidate = candidate.rsplit(" ", 1)[0]
    return candidate.rstrip(" ,;:")


def _unit_for_argument(
    hotel: HotelProfile,
    argument: Argument,
    source_refs: tuple[str, ...],
) -> ScoringUnit:
    if argument.intrinsic_strength is None:
        raise HotelHybridValidationError(
            f"deterministic argument {argument.id} has no intrinsic strength"
        )
    if argument.argument_family == EMPIRICAL_ASPECT:
        kind = "opinion"
        unit_id = (
            f"OPINION::{hotel.hotel_id}::{argument.aspect}::"
            f"{argument.arg_type}::review_evidence"
        )
        preference_refs = (str(argument.aspect),)
    elif argument.argument_family == STRUCTURED_FACT:
        kind = "fact"
        metadata = argument.metadata
        target = str(metadata.get("constraint_field", "facility"))
        constraint_id = str(metadata.get("constraint_id", "soft_constraint"))
        qualifiers = metadata.get("constraint_qualifiers", {})
        unit_id = (
            f"FACT::{hotel.hotel_id}::{_safe_token(target)}::"
            f"{_qualifier_token(qualifiers)}::{_safe_token(constraint_id)}"
        )
        preference_refs = (constraint_id,)
    else:
        raise HotelHybridValidationError(
            f"unsupported deterministic argument family: "
            f"{argument.argument_family}"
        )
    return ScoringUnit(
        unit_id=unit_id,
        kind=kind,
        arg_type=argument.arg_type,
        intrinsic_strength=float(argument.intrinsic_strength),
        preference_refs=preference_refs,
        source_refs=source_refs,
        atomic_argument=argument,
    )


def prepare_hybrid_context(
    hotel: HotelProfile,
    preferences: SessionPreferences,
    *,
    ontology: FacilityOntology | None = None,
    constraint_outcomes: tuple[ConstraintOutcome, ...] | None = None,
) -> PreparedHybridContext:
    resolved_ontology = ontology or load_default_facility_ontology()
    normalization = normalize_hotel_facilities(hotel, resolved_ontology)
    outcomes = constraint_outcomes or evaluate_constraints(
        hotel,
        preferences.constraints,
        ontology=resolved_ontology,
    )
    empirical_arguments = build_empirical_arguments(hotel, preferences)
    factual_arguments = build_structured_fact_arguments(hotel, outcomes)

    source_builders: dict[str, dict[str, Any]] = {}

    def add_source(
        source_id: str,
        *,
        kind: str,
        payload: dict[str, Any],
        allowed_types: tuple[str, ...],
        evidence_text: str | None = None,
        unit_id: str | None = None,
        hard_constraint: bool = False,
        unknown: bool = False,
    ) -> None:
        builder = source_builders.setdefault(
            source_id,
            {
                "source_id": source_id,
                "kind": kind,
                "payload": dict(payload),
                "allowed_types": set(),
                "scoring_unit_ids": set(),
                "evidence_text": evidence_text,
                "hard_constraint": hard_constraint,
                "unknown": unknown,
            },
        )
        builder["allowed_types"].update(allowed_types)
        if unit_id is not None:
            builder["scoring_unit_ids"].add(unit_id)
        builder["hard_constraint"] = (
            builder["hard_constraint"] or hard_constraint
        )
        builder["unknown"] = builder["unknown"] or unknown

    observation_aliases: dict[str, str] = {}
    for fact in normalization.facts:
        ordered_observations = sorted(
            fact.observations,
            key=lambda item: (item.source_ref, item.evidence),
        )
        for index, observation in enumerate(
            ordered_observations[:MAX_FACILITY_SOURCES_PER_CAPABILITY]
        ):
            if observation.source_kind == "facility":
                source_id = (
                    f"FACILITY::{fact.capability}::"
                    f"{_qualifier_token(observation.qualifiers)}::{index}"
                )
            else:
                source_id = observation.source_ref
            observation_aliases[observation.source_ref] = source_id
            add_source(
                source_id,
                kind=observation.source_kind,
                payload={
                    "capability": observation.capability,
                    "present": observation.present,
                    "qualifiers": dict(observation.qualifiers),
                    "evidence": observation.evidence,
                },
                allowed_types=(
                    ("support",) if observation.present else ("attack",)
                ),
                evidence_text=observation.evidence,
            )

    units: list[ScoringUnit] = []
    for argument in empirical_arguments:
        aspect_source = f"ASPECT::{argument.aspect}::{argument.arg_type.upper()}"
        review_refs = tuple(
            (
                f"REVIEW::{source['review_id']}::{source['aspect']}::"
                f"{source['stance']}::{index}"
            )
            for index, source in enumerate(argument.review_sources)
        )
        source_refs = (aspect_source,) + review_refs
        unit = _unit_for_argument(hotel, argument, source_refs)
        units.append(unit)
        add_source(
            aspect_source,
            kind="aspect_statistics",
            payload={
                "aspect": argument.aspect,
                "stance": argument.arg_type,
                "n_support": argument.n_support,
                "n_attack": argument.n_attack,
                "n_neutral": argument.n_neutral,
                "evidence_basis": "aggregated_aspect_counts_and_citations",
            },
            allowed_types=(argument.arg_type,),
            unit_id=unit.unit_id,
        )
        for review_ref, review_source in zip(
            review_refs,
            argument.review_sources,
        ):
            add_source(
                review_ref,
                kind="review_evidence",
                payload={
                    "review_id": review_source["review_id"],
                    "aspect": review_source["aspect"],
                    "stance": review_source["stance"],
                    "evidence": review_source["evidence"],
                },
                allowed_types=(argument.arg_type,),
                evidence_text=review_source["evidence"],
                unit_id=unit.unit_id,
            )

    outcomes_by_ref = {
        outcome.constraint.preference_ref: outcome for outcome in outcomes
    }
    for argument in factual_arguments:
        constraint_ref = str(argument.metadata["constraint_id"])
        outcome = outcomes_by_ref[constraint_ref]
        for fact_source in outcome.fact_sources:
            raw_source_ref = fact_source.get("source_ref")
            if not isinstance(raw_source_ref, str) or not raw_source_ref:
                continue
            source_ref = observation_aliases.get(
                raw_source_ref,
                raw_source_ref,
            )
            if source_ref in source_builders:
                continue
            source_payload = {
                str(key): value
                for key, value in fact_source.items()
                if key != "source_ref"
            }
            evidence_value = fact_source.get("value")
            add_source(
                source_ref,
                kind=str(fact_source.get("source", "structured_fact")),
                payload=source_payload,
                allowed_types=(argument.arg_type,),
                evidence_text=(
                    evidence_value if isinstance(evidence_value, str) else None
                ),
            )
        fact_refs = tuple(
            observation_aliases.get(
                str(source.get("source_ref")),
                str(source.get("source_ref")),
            )
            for source in outcome.fact_sources
            if source.get("source_ref")
        )
        constraint_source = f"CONSTRAINT::{constraint_ref}"
        source_refs = (constraint_source,) + fact_refs
        unit = _unit_for_argument(hotel, argument, source_refs)
        units.append(unit)
        add_source(
            constraint_source,
            kind="constraint_result",
            payload={
                "constraint_id": constraint_ref,
                "target": outcome.constraint.canonical_target,
                "qualifiers": dict(outcome.constraint.qualifiers),
                "status": outcome.status.value,
                "mode": outcome.constraint.mode,
            },
            allowed_types=(argument.arg_type,),
            evidence_text=outcome.constraint.source_text,
            unit_id=unit.unit_id,
        )
        for source_ref in fact_refs:
            builder = source_builders.get(source_ref)
            if builder is not None:
                builder["scoring_unit_ids"].add(unit.unit_id)

    for outcome in outcomes:
        constraint_ref = outcome.constraint.preference_ref
        source_id = f"CONSTRAINT::{constraint_ref}"
        if outcome.status is ConstraintStatus.SATISFIED:
            allowed_types = ("support",)
        elif outcome.status is ConstraintStatus.VIOLATED:
            allowed_types = ("attack",)
        else:
            allowed_types = ()
        add_source(
            source_id,
            kind="constraint_result",
            payload={
                "constraint_id": constraint_ref,
                "target": outcome.constraint.canonical_target,
                "qualifiers": dict(outcome.constraint.qualifiers),
                "status": outcome.status.value,
                "mode": outcome.constraint.mode,
                "reason": outcome.reason,
            },
            allowed_types=allowed_types,
            evidence_text=outcome.constraint.source_text,
            hard_constraint=outcome.constraint.hard,
            unknown=outcome.status is ConstraintStatus.UNKNOWN,
        )

    if hotel.metadata.city:
        add_source(
            "METADATA::city",
            kind="metadata",
            payload={"field": "city", "value": hotel.metadata.city},
            allowed_types=("support", "attack"),
            evidence_text=hotel.metadata.city,
        )
    description = _description_excerpt(hotel.metadata.description)
    if description:
        add_source(
            "METADATA::description",
            kind="metadata",
            payload={"field": "description", "value": description},
            allowed_types=("support", "attack"),
            evidence_text=description,
        )
    add_source(
        "STAT::hotel",
        kind="hotel_statistics",
        payload={
            "n_reviews_total": hotel.stats.n_reviews_total,
            "average_rating": hotel.stats.average_rating,
            "total_support": hotel.stats.total_support,
            "total_attack": hotel.stats.total_attack,
            "total_neutral": hotel.stats.total_neutral,
        },
        allowed_types=("support", "attack"),
    )

    policy_sources_added = sum(
        builder["kind"] == "policy" for builder in source_builders.values()
    )
    for index, policy in enumerate(hotel.metadata.policies):
        if policy_sources_added >= 8:
            break
        semantic = policy.semantic_payload()
        if not semantic:
            continue
        source_id = (
            f"POLICY::{policy.source_id if policy.source_id is not None else index}::"
            "description"
        )
        if source_id in source_builders:
            continue
        add_source(
            source_id,
            kind="policy",
            payload=semantic,
            allowed_types=("support", "attack"),
            evidence_text=semantic.get("description"),
        )
        policy_sources_added += 1

    sources = tuple(
        AuthorizedSource(
            source_id=builder["source_id"],
            kind=builder["kind"],
            payload=builder["payload"],
            allowed_types=tuple(sorted(builder["allowed_types"])),
            scoring_unit_ids=tuple(sorted(builder["scoring_unit_ids"])),
            evidence_text=builder["evidence_text"],
            hard_constraint=builder["hard_constraint"],
            unknown=builder["unknown"],
        )
        for _, builder in sorted(source_builders.items())
    )
    clean_preferences = preferences.to_dict()
    clean_preferences.pop("interpretation_trace", None)
    hotel_context = {
        "hotel_id": hotel.hotel_id,
        "hotel_name": hotel.metadata.name,
        "preferences": clean_preferences,
        "city_known": hotel.metadata.city is not None,
        "canonical_capabilities": [
            {
                "capability": fact.capability,
                "status": fact.status,
                "observations": [
                    {
                        "present": observation.present,
                        "qualifiers": dict(observation.qualifiers),
                        "source_ref": observation_aliases.get(
                            observation.source_ref,
                            observation.source_ref,
                        ),
                    }
                    for observation in sorted(
                        fact.observations,
                        key=lambda item: (item.source_ref, item.evidence),
                    )[:MAX_FACILITY_SOURCES_PER_CAPABILITY]
                ],
            }
            for fact in normalization.facts
        ],
    }
    return PreparedHybridContext(
        hotel_context=hotel_context,
        sources=sources,
        scoring_units=tuple(units),
        constraint_outcomes=outcomes,
        facility_normalization=normalization,
    )


@dataclass(frozen=True, slots=True)
class AcceptedHybridArgument:
    proposal: dict[str, Any]
    effective_explanatory_only: bool
    scoring_status: str
    scoring_unit_id: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.proposal)
        payload.update(
            {
                "effective_explanatory_only": (
                    self.effective_explanatory_only
                ),
                "scoring_status": self.scoring_status,
                "scoring_unit_id": self.scoring_unit_id,
                "preference_refs_source": (
                    "python_scoring_unit_registry"
                    if self.scoring_unit_id is not None
                    else "validated_generator_references"
                ),
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class RejectedHybridArgument:
    proposal: Any
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ValidatedHybridRelation:
    relation: dict[str, Any]
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.relation,
            "accepted": self.accepted,
            "explanatory_only": True,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class HybridValidationResult:
    proposed_arguments: tuple[Any, ...]
    accepted_arguments: tuple[AcceptedHybridArgument, ...]
    rejected_arguments: tuple[RejectedHybridArgument, ...]
    relations: tuple[ValidatedHybridRelation, ...]
    scoring_arguments: tuple[Argument, ...]
    scoring_units: tuple[dict[str, Any], ...]
    excluded_arguments: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_arguments": list(self.proposed_arguments),
            "accepted_arguments": [
                argument.to_dict() for argument in self.accepted_arguments
            ],
            "rejected_arguments": [
                argument.to_dict() for argument in self.rejected_arguments
            ],
            "relations": [relation.to_dict() for relation in self.relations],
            "scoring_units": [dict(unit) for unit in self.scoring_units],
            "excluded_arguments": [
                dict(argument) for argument in self.excluded_arguments
            ],
        }


def _proposal_reasons(
    proposal: Any,
    *,
    source_lookup: dict[str, AuthorizedSource],
    unit_lookup: dict[str, ScoringUnit],
    preference_refs: set[str],
) -> tuple[list[str], dict[str, Any] | None]:
    if not isinstance(proposal, dict):
        return ["invalid_argument_schema"], None
    required_fields = {
        "id",
        "kind",
        "type",
        "text",
        "source_refs",
        "scoring_unit_refs",
        "explanatory_only",
    }
    allowed_fields = required_fields | {"preference_refs"}
    extra_fields = set(proposal) - allowed_fields
    reasons = []
    if extra_fields & {
        "strength",
        "score",
        "intrinsic_strength",
        "importance_raw",
        "normalized_weight",
        "importance_coefficient",
        "weighting_method",
        "final_force",
        "force_formula",
        "force_components",
    }:
        reasons.append("forbidden_strength_field")
    if extra_fields & {
        "hotel_field",
        "hotel_fields",
        "facility_id",
        "review_id",
    }:
        reasons.append("hallucinated_hotel_field")
    if extra_fields - {
        "strength",
        "score",
        "intrinsic_strength",
        "importance_raw",
        "normalized_weight",
        "importance_coefficient",
        "weighting_method",
        "final_force",
        "force_formula",
        "force_components",
        "hotel_field",
        "hotel_fields",
        "facility_id",
        "review_id",
    }:
        reasons.append("invalid_argument_schema")
    if required_fields - set(proposal):
        reasons.append("invalid_argument_schema")
        return list(dict.fromkeys(reasons)), None

    cleaned = {key: proposal[key] for key in required_fields}
    supplied_preference_refs = proposal.get("preference_refs", [])
    cleaned["preference_refs"] = supplied_preference_refs
    if not isinstance(cleaned["id"], str) or not cleaned["id"].strip():
        reasons.append("invalid_argument_id")
    if cleaned["kind"] not in HYBRID_ARGUMENT_KINDS:
        reasons.append("unsupported_argument_kind")
    if cleaned["type"] not in HYBRID_ARGUMENT_TYPES:
        reasons.append("unsupported_argument_type")
    if not isinstance(cleaned["text"], str) or not cleaned["text"].strip():
        reasons.append("invalid_argument_text")
    reference_limits = {
        "source_refs": MAX_HYBRID_SOURCE_REFS,
        "scoring_unit_refs": MAX_HYBRID_SCORING_UNIT_REFS,
    }
    for field_name, maximum in reference_limits.items():
        value = cleaned[field_name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            reasons.append("invalid_argument_schema")
            continue
        if len(value) > maximum:
            reasons.append("invalid_argument_schema")
        if field_name == "source_refs" and not value:
            reasons.append("invalid_argument_schema")
    if (
        isinstance(cleaned["scoring_unit_refs"], list)
        and len(cleaned["scoring_unit_refs"]) > 1
    ):
        reasons.append("multiple_scoring_units")
    if not isinstance(cleaned["explanatory_only"], bool):
        reasons.append("invalid_argument_schema")
    if reasons:
        return list(dict.fromkeys(reasons)), cleaned

    unknown_sources = [
        reference
        for reference in cleaned["source_refs"]
        if reference not in source_lookup
    ]
    if unknown_sources:
        reasons.append("unknown_source_ref")
    unknown_units = [
        reference
        for reference in cleaned["scoring_unit_refs"]
        if reference not in unit_lookup
    ]
    if unknown_units:
        reasons.append("unknown_scoring_unit_ref")
    if unknown_sources or unknown_units:
        return list(dict.fromkeys(reasons)), cleaned

    sources = [source_lookup[ref] for ref in cleaned["source_refs"]]
    units = [unit_lookup[ref] for ref in cleaned["scoring_unit_refs"]]
    if len(units) == 1:
        # The Python scoring registry is authoritative.  Gemini's optional
        # copy is deliberately ignored, even when it contains extra refs.
        cleaned["preference_refs"] = list(units[0].preference_refs)
    else:
        if not isinstance(supplied_preference_refs, list) or not all(
            isinstance(item, str) and item
            for item in supplied_preference_refs
        ):
            reasons.append("invalid_argument_schema")
        elif not supplied_preference_refs:
            reasons.append("missing_preference_ref")
        elif len(supplied_preference_refs) > MAX_HYBRID_PREFERENCE_REFS:
            reasons.append("invalid_argument_schema")
        elif any(
            reference not in preference_refs
            for reference in supplied_preference_refs
        ):
            reasons.append("unknown_preference_ref")
    if any(
        source.allowed_types
        and cleaned["type"] not in source.allowed_types
        for source in sources
    ):
        reasons.append("polarity_source_mismatch")
    if cleaned["type"] == "attack" and any(source.unknown for source in sources):
        reasons.append("unknown_information_used_as_negative")
    if not cleaned["explanatory_only"] and any(
        source.hard_constraint for source in sources
    ):
        reasons.append("hard_constraint_excluded")
    if cleaned["kind"] in {"opinion", "fact"} and not cleaned[
        "explanatory_only"
    ]:
        if len(units) != 1:
            reasons.append("missing_or_ambiguous_scoring_unit")
        else:
            unit = units[0]
            if unit.kind != cleaned["kind"]:
                reasons.append("scoring_unit_kind_mismatch")
            if unit.arg_type != cleaned["type"]:
                reasons.append("polarity_source_mismatch")
            if not set(unit.source_refs).issubset(cleaned["source_refs"]):
                reasons.append("missing_required_source_ref")
            elif set(unit.source_refs) != set(cleaned["source_refs"]):
                reasons.append("unexpected_source_ref_for_scoring_unit")
            if any(
                cleaned["type"] not in source.allowed_types
                for source in sources
                if source.scoring_unit_ids
                and unit.unit_id in source.scoring_unit_ids
            ):
                reasons.append("polarity_source_mismatch")
    elif cleaned["kind"] in {"opinion", "fact"} and units:
        if any(unit.kind != cleaned["kind"] for unit in units):
            reasons.append("scoring_unit_kind_mismatch")

    return list(dict.fromkeys(reasons)), cleaned


def validate_hybrid_proposals(
    raw_batch: object,
    *,
    prepared: PreparedHybridContext,
    preferences: SessionPreferences,
    hotel: HotelProfile,
) -> HybridValidationResult:
    if not isinstance(raw_batch, dict):
        raise HotelHybridValidationError(
            "hybrid generator output must be a JSON object"
        )
    if set(raw_batch) != {"arguments", "relations"}:
        raise HotelHybridValidationError(
            "hybrid generator output requires only arguments and relations"
        )
    proposed = raw_batch["arguments"]
    raw_relations = raw_batch["relations"]
    if not isinstance(proposed, list) or not isinstance(raw_relations, list):
        raise HotelHybridValidationError(
            "hybrid arguments and relations must be lists"
        )
    if len(proposed) > MAX_HYBRID_ARGUMENTS:
        raise HotelHybridValidationError(
            f"hybrid output exceeds {MAX_HYBRID_ARGUMENTS} arguments"
        )
    if len(raw_relations) > MAX_HYBRID_RELATIONS:
        raise HotelHybridValidationError(
            f"hybrid output exceeds {MAX_HYBRID_RELATIONS} relations"
        )

    source_lookup = prepared.source_lookup()
    unit_lookup = prepared.unit_lookup()
    preference_refs = {
        preference.aspect for preference in preferences.aspect_preferences
    } | {
        constraint.preference_ref for constraint in preferences.constraints
    }
    accepted_candidates = []
    rejected = []
    seen_ids = set()
    for proposal in proposed:
        reasons, cleaned = _proposal_reasons(
            proposal,
            source_lookup=source_lookup,
            unit_lookup=unit_lookup,
            preference_refs=preference_refs,
        )
        if cleaned is not None and isinstance(cleaned.get("id"), str):
            proposal_id = cleaned["id"]
            if proposal_id in seen_ids:
                reasons.append("duplicate_argument_id")
            else:
                seen_ids.add(proposal_id)
        if reasons or cleaned is None:
            rejected.append(
                RejectedHybridArgument(
                    proposal=proposal,
                    reasons=tuple(dict.fromkeys(reasons)),
                )
            )
        else:
            accepted_candidates.append(cleaned)

    scoring_arguments = []
    accepted = []
    excluded = []
    used_units: dict[str, str] = {}
    unit_argument_ids: dict[str, list[str]] = {
        unit.unit_id: [] for unit in prepared.scoring_units
    }
    for proposal in accepted_candidates:
        proposal_id = proposal["id"]
        unit_refs = proposal["scoring_unit_refs"]
        effective_explanatory = (
            proposal["explanatory_only"]
            or proposal["kind"] in EXPLANATORY_KINDS
            or len(unit_refs) != 1
        )
        scoring_unit_id = unit_refs[0] if len(unit_refs) == 1 else None
        if effective_explanatory:
            status = "excluded_explanatory_only"
            excluded.append(
                {
                    "argument_id": proposal_id,
                    "reason": (
                        "composite_or_explanatory_only"
                        if len(unit_refs) != 1
                        or proposal["kind"] in EXPLANATORY_KINDS
                        else "explanatory_only"
                    ),
                }
            )
        elif scoring_unit_id in used_units:
            status = "excluded_duplicate_scoring_unit"
            excluded.append(
                {
                    "argument_id": proposal_id,
                    "reason": "duplicate_scoring_unit",
                    "scoring_unit_id": scoring_unit_id,
                    "counted_argument_id": used_units[scoring_unit_id],
                }
            )
        else:
            status = "included_in_dfquad"
            used_units[scoring_unit_id] = proposal_id
            unit = unit_lookup[scoring_unit_id]
            unit_argument_ids[scoring_unit_id].append(proposal_id)
            evidence = [
                source_lookup[source_ref].evidence_text
                for source_ref in proposal["source_refs"]
                if source_lookup[source_ref].evidence_text
            ]
            unique_evidence = list(dict.fromkeys(evidence))[:2]
            atomic = unit.atomic_argument
            scoring_arguments.append(
                Argument(
                    id=proposal_id,
                    arg_type=proposal["type"],
                    text=proposal["text"],
                    evidence=unique_evidence,
                    aspect_effect=atomic.aspect_effect,
                    used_aspects=list(atomic.used_aspects),
                    target_item_name=hotel.metadata.name,
                    argument_family=atomic.argument_family,
                    aspect=atomic.aspect,
                    intrinsic_strength=unit.intrinsic_strength,
                    importance_raw=atomic.importance_raw,
                    normalized_weight=atomic.normalized_weight,
                    evidence_score=atomic.evidence_score,
                    n_support=atomic.n_support,
                    n_attack=atomic.n_attack,
                    n_neutral=atomic.n_neutral,
                    review_sources=list(atomic.review_sources),
                    source_refs=list(proposal["source_refs"]),
                    preference_refs=list(proposal["preference_refs"]),
                    scoring_unit_id=unit.unit_id,
                    explanatory_only=False,
                    metadata={
                        **dict(atomic.metadata),
                        "hybrid_generator_argument_id": proposal_id,
                        "source_refs": list(proposal["source_refs"]),
                        "scoring_unit_id": unit.unit_id,
                    },
                )
            )
        accepted.append(
            AcceptedHybridArgument(
                proposal=proposal,
                effective_explanatory_only=effective_explanatory,
                scoring_status=status,
                scoring_unit_id=scoring_unit_id,
            )
        )

    accepted_ids = {argument.proposal["id"] for argument in accepted}
    relations = []
    expected_relation_fields = {
        "id",
        "source_argument_id",
        "target_argument_id",
        "relation_type",
    }
    seen_relation_ids = set()
    for relation in raw_relations:
        reasons = []
        if not isinstance(relation, dict) or set(relation) != expected_relation_fields:
            reasons.append("invalid_relation_schema")
            normalized_relation = relation if isinstance(relation, dict) else {}
        else:
            normalized_relation = dict(relation)
            if not all(
                isinstance(relation[field], str) and relation[field].strip()
                for field in (
                    "id",
                    "source_argument_id",
                    "target_argument_id",
                    "relation_type",
                )
            ):
                reasons.append("invalid_relation_schema")
            elif relation["id"] in seen_relation_ids:
                reasons.append("duplicate_relation_id")
            else:
                seen_relation_ids.add(relation["id"])
            if "invalid_relation_schema" not in reasons:
                if relation["source_argument_id"] not in accepted_ids or relation[
                    "target_argument_id"
                ] not in accepted_ids:
                    reasons.append("unknown_relation_argument_ref")
                if relation["source_argument_id"] == relation[
                    "target_argument_id"
                ]:
                    reasons.append("self_relation")
                if relation["relation_type"] not in {
                    "support",
                    "attack",
                    "qualifies",
                    "tradeoff",
                    "synthesis",
                }:
                    reasons.append("unsupported_relation_type")
        if not reasons:
            reasons.append("current_graph_relations_are_root_only")
        relations.append(
            ValidatedHybridRelation(
                relation=normalized_relation,
                accepted=not any(
                    reason
                    for reason in reasons
                    if reason != "current_graph_relations_are_root_only"
                ),
                reasons=tuple(reasons),
            )
        )

    scoring_unit_rows = []
    for unit in prepared.scoring_units:
        row = unit.to_dict()
        attached = [
            argument.proposal["id"]
            for argument in accepted
            if unit.unit_id in argument.proposal["scoring_unit_refs"]
        ]
        row.update(
            {
                "attached_argument_ids": attached,
                "counted_argument_id": used_units.get(unit.unit_id),
                "included_in_dfquad": unit.unit_id in used_units,
                "dfquad_reason": (
                    "counted_once_from_validated_argument"
                    if unit.unit_id in used_units
                    else (
                        "accepted_arguments_were_explanatory_only"
                        if attached
                        else "no_validated_argument_selected"
                    )
                ),
            }
        )
        scoring_unit_rows.append(row)
    return HybridValidationResult(
        proposed_arguments=tuple(proposed),
        accepted_arguments=tuple(accepted),
        rejected_arguments=tuple(rejected),
        relations=tuple(relations),
        scoring_arguments=tuple(scoring_arguments),
        scoring_units=tuple(scoring_unit_rows),
        excluded_arguments=tuple(excluded),
    )


def run_hybrid_generation(
    *,
    hotel: HotelProfile,
    preferences: SessionPreferences,
    generator: HybridArgumentGenerator,
    ontology: FacilityOntology | None = None,
    constraint_outcomes: tuple[ConstraintOutcome, ...] | None = None,
) -> tuple[PreparedHybridContext, HybridValidationResult, dict[str, Any]]:
    prepared = prepare_hybrid_context(
        hotel,
        preferences,
        ontology=ontology,
        constraint_outcomes=constraint_outcomes,
    )
    raw_batch = generator.propose_arguments(
        preferences=preferences,
        hotel_context=prepared.hotel_context,
        authorized_sources=prepared.prompt_sources(),
        scoring_units=prepared.prompt_units(),
        constraint_outcomes=[
            outcome.to_dict() for outcome in prepared.constraint_outcomes
        ],
    )
    validation = validate_hybrid_proposals(
        raw_batch,
        prepared=prepared,
        preferences=preferences,
        hotel=hotel,
    )
    trace = getattr(generator, "last_trace", None)
    return prepared, validation, dict(trace) if isinstance(trace, dict) else {}
