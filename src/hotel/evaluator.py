from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.argumentation.dfquad import evaluate_root_dfquad
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.schema import Argument

from .argument_builder import (
    EMPIRICAL_ASPECT,
    STRUCTURED_FACT,
    build_empirical_arguments,
    build_structured_fact_arguments,
)
from .constraints import ConstraintOutcome, ConstraintStatus, evaluate_constraints
from .facility_ontology import (
    FacilityOntology,
    normalize_hotel_facilities,
)
from .hybrid import HybridArgumentGenerator, run_hybrid_generation
from .models import HotelProfile, HotelProfileDataset
from .preferences import ABSOLUTE_5_WEIGHTING_METHOD, SessionPreferences


ROOT_TEXT = "Recommend this hotel for the current session"
ROOT_BASE_SCORE = 0.5
ARGUMENT_MODES = frozenset({"baseline", "hybrid"})


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    status: str
    hard_constraints: tuple[ConstraintOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hard_constraints": [
                outcome.to_dict() for outcome in self.hard_constraints
            ],
        }


@dataclass(frozen=True, slots=True)
class HotelEvaluationResult:
    hotel_id: str
    hotel_name: str
    argument_mode: str
    session_preferences: SessionPreferences
    eligibility: EligibilityResult
    constraint_outcomes: tuple[ConstraintOutcome, ...]
    ineligibility_reasons: tuple[dict[str, Any], ...]
    observed_preference_aspects: tuple[str, ...]
    missing_preference_aspects: tuple[str, ...]
    unknown_constraints: tuple[ConstraintOutcome, ...]
    preference_coverage: float
    linear_empirical_score: float | None
    dfquad_score: float
    weighting_method: str
    scoring_status: str
    is_personalized: bool
    arguments: tuple[Argument, ...]
    scoring_units: tuple[dict[str, Any], ...]
    graph: dict[str, Any]
    dfquad: dict[str, Any]
    facility_normalization: dict[str, Any]
    hybrid: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "hotel_name": self.hotel_name,
            "argument_mode": self.argument_mode,
            "session_preferences": self.session_preferences.to_dict(),
            "eligibility": self.eligibility.to_dict(),
            "constraint_outcomes": [
                outcome.to_dict() for outcome in self.constraint_outcomes
            ],
            "ineligibility_reasons": [
                dict(reason) for reason in self.ineligibility_reasons
            ],
            "observed_preference_aspects": list(
                self.observed_preference_aspects
            ),
            "missing_preference_aspects": list(
                self.missing_preference_aspects
            ),
            "unknown_constraints": [
                outcome.to_dict() for outcome in self.unknown_constraints
            ],
            "preference_coverage": self.preference_coverage,
            "linear_empirical_score": self.linear_empirical_score,
            "dfquad_score": self.dfquad_score,
            "weighting_method": self.weighting_method,
            "scoring_status": self.scoring_status,
            "is_personalized": self.is_personalized,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "scoring_units": [dict(unit) for unit in self.scoring_units],
            "graph": self.graph,
            "dfquad": self.dfquad,
            "facility_normalization": self.facility_normalization,
            "hybrid": self.hybrid,
        }


def _empirical_baseline(
    hotel: HotelProfile,
    preferences: SessionPreferences,
) -> tuple[tuple[str, ...], tuple[str, ...], float, float | None]:
    """Return observed/missing aspects, coverage, and linear quality score.

    ``AspectProfile.score`` is the positive aspect-quality estimate already
    shrunk toward 0.5 by the profile's confidence.  It is therefore used once
    as ``empirical_quality`` and is never injected into the DF-QuAD root.
    """
    observed = []
    missing = []
    weighted_quality = 0.0
    observed_weight = 0.0
    requested_weight = sum(
        preference.normalized_weight
        for preference in preferences.active_aspect_preferences
    )

    for preference in preferences.active_aspect_preferences:
        profile = hotel.get_aspect(preference.aspect)
        if profile is None or profile.n_mentions == 0:
            missing.append(preference.aspect)
            continue
        observed.append(preference.aspect)
        observed_weight += preference.normalized_weight
        weighted_quality += preference.normalized_weight * profile.score

    coverage = (
        observed_weight / requested_weight
        if requested_weight > 0.0
        else 0.0
    )
    linear_score = (
        weighted_quality / observed_weight
        if observed_weight > 0.0
        else None
    )
    return tuple(observed), tuple(missing), coverage, linear_score


def _argument_scoring_unit_row(argument: Argument) -> dict[str, Any]:
    metadata = argument.metadata
    if argument.argument_family == EMPIRICAL_ASPECT:
        kind = "opinion"
        preference_refs = [argument.aspect] if argument.aspect else []
    elif argument.argument_family == STRUCTURED_FACT:
        kind = "fact"
        constraint_id = metadata.get("constraint_id")
        preference_refs = [constraint_id] if constraint_id else []
    else:
        kind = argument.argument_family or "unknown"
        preference_refs = list(argument.preference_refs)
    source_refs = list(argument.source_refs) or list(
        metadata.get("source_refs", [])
    )
    return {
        "scoring_unit_id": argument.scoring_unit_id or argument.id,
        "kind": kind,
        "type": argument.arg_type,
        "intent_ref": preference_refs[0] if preference_refs else None,
        "preference_refs": preference_refs,
        "source_refs": source_refs,
        "importance_raw": argument.importance_raw,
        "normalized_weight": argument.normalized_weight,
        "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
        "confidence_factor": argument.evidence_score,
        "force_formula": metadata.get("force_formula"),
        "force_components": dict(metadata.get("force_components", {})),
        "final_force": argument.intrinsic_strength,
        "availability_status": "available",
        "availability_reason": metadata.get("inclusion_reason"),
        "weight_active": True,
        "attached_argument_ids": [argument.id],
        "counted_argument_id": argument.id,
        "included_in_dfquad": True,
        "dfquad_reason": "counted_once_from_deterministic_argument",
    }


def _build_scoring_unit_audit(
    hotel: HotelProfile,
    preferences: SessionPreferences,
    outcomes: tuple[ConstraintOutcome, ...],
    arguments: tuple[Argument, ...],
    *,
    hybrid_rows: tuple[dict[str, Any], ...] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows = (
        [dict(row) for row in hybrid_rows]
        if hybrid_rows is not None
        else [_argument_scoring_unit_row(argument) for argument in arguments]
    )
    represented_refs = {
        str(reference)
        for row in rows
        for reference in row.get("preference_refs", [])
        if reference is not None
    }

    for preference in preferences.aspect_preferences:
        if not preference.active:
            continue
        if preference.aspect in represented_refs:
            continue
        rows.append(
            {
                "scoring_unit_id": (
                    f"OPINION::{hotel.hotel_id}::{preference.aspect}::"
                    "unavailable"
                ),
                "kind": "opinion",
                "type": None,
                "intent_ref": preference.aspect,
                "preference_refs": [preference.aspect],
                "source_refs": [],
                "importance_raw": preference.importance_raw,
                "normalized_weight": preference.normalized_weight,
                "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                "confidence_factor": None,
                "force_formula": "(importance_raw / 5) * wilson_lower_bound",
                "force_components": {
                    "importance_raw": preference.importance_raw,
                    "importance_coefficient": preference.normalized_weight,
                    "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                    "wilson_lower_bound": None,
                },
                "final_force": None,
                "availability_status": "no_compatible_empirical_evidence",
                "availability_reason": (
                    "no support or attack argument had compatible review evidence"
                ),
                "weight_active": preference.active,
                "attached_argument_ids": [],
                "counted_argument_id": None,
                "included_in_dfquad": False,
                "dfquad_reason": "no_deterministic_force_available",
            }
        )

    outcomes_by_ref = {
        outcome.constraint.preference_ref: outcome for outcome in outcomes
    }
    for constraint in preferences.constraints:
        if not constraint.hard and constraint.importance_raw <= 0.0:
            continue
        reference = constraint.preference_ref
        if reference in represented_refs:
            continue
        outcome = outcomes_by_ref[reference]
        if constraint.hard:
            availability_status = "hard_constraint_eligibility_only"
            reason = "hard constraints are eligibility-only and have weight zero"
            formula = None
        elif outcome.status is ConstraintStatus.UNKNOWN:
            availability_status = "unknown"
            reason = "unknown factual status contributes no DF-QuAD argument"
            formula = "importance_raw / 5 when the fact becomes known"
        else:
            availability_status = "known_but_not_selected"
            reason = "no validated scoring argument selected this known fact"
            formula = "importance_raw / 5"
        rows.append(
            {
                "scoring_unit_id": f"CONSTRAINT_INTENT::{reference}",
                "kind": "fact",
                "type": (
                    "support"
                    if outcome.status is ConstraintStatus.SATISFIED
                    else (
                        "attack"
                        if outcome.status is ConstraintStatus.VIOLATED
                        else None
                    )
                ),
                "intent_ref": reference,
                "preference_refs": [reference],
                "source_refs": [
                    str(source.get("source_ref"))
                    for source in outcome.fact_sources
                    if source.get("source_ref")
                ],
                "importance_raw": constraint.importance_raw,
                "normalized_weight": constraint.normalized_weight,
                "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                "confidence_factor": (
                    1.0
                    if outcome.status is not ConstraintStatus.UNKNOWN
                    and not constraint.hard
                    else None
                ),
                "force_formula": formula,
                "force_components": {
                    "importance_raw": constraint.importance_raw,
                    "importance_coefficient": constraint.normalized_weight,
                    "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
                    "deterministic_fact_confidence": (
                        None
                        if outcome.status is ConstraintStatus.UNKNOWN
                        else 1.0
                    ),
                },
                "final_force": None,
                "availability_status": availability_status,
                "availability_reason": reason,
                "weight_active": (
                    not constraint.hard and constraint.importance_raw > 0.0
                ),
                "attached_argument_ids": [],
                "counted_argument_id": None,
                "included_in_dfquad": False,
                "dfquad_reason": availability_status,
            }
        )
    return tuple(rows)


def evaluate_hotel_session(
    hotel: HotelProfile,
    preferences: SessionPreferences,
    *,
    argument_mode: str = "baseline",
    hybrid_generator: HybridArgumentGenerator | None = None,
    facility_ontology: FacilityOntology | None = None,
) -> HotelEvaluationResult:
    if argument_mode not in ARGUMENT_MODES:
        raise ValueError(
            f"argument_mode must be one of {sorted(ARGUMENT_MODES)}"
        )
    observed, missing, coverage, linear_score = _empirical_baseline(
        hotel,
        preferences,
    )
    constraint_outcomes = evaluate_constraints(
        hotel,
        preferences.constraints,
        ontology=facility_ontology,
    )
    facility_normalization = normalize_hotel_facilities(
        hotel,
        facility_ontology,
    )
    hybrid_payload = None
    hybrid_scoring_rows = None
    if argument_mode == "baseline":
        empirical_arguments = build_empirical_arguments(hotel, preferences)
        factual_arguments = build_structured_fact_arguments(
            hotel,
            constraint_outcomes,
        )
        arguments = tuple(empirical_arguments + factual_arguments)
    else:
        if hybrid_generator is None:
            raise ValueError(
                "hybrid argument mode requires a HybridArgumentGenerator"
            )
        prepared, hybrid_validation, generator_trace = run_hybrid_generation(
            hotel=hotel,
            preferences=preferences,
            generator=hybrid_generator,
            ontology=facility_ontology,
            constraint_outcomes=constraint_outcomes,
        )
        arguments = hybrid_validation.scoring_arguments
        facility_normalization = prepared.facility_normalization
        hybrid_payload = {
            "prepared_context": prepared.to_dict(),
            "validation": hybrid_validation.to_dict(),
            "generator_trace": generator_trace,
        }
        hybrid_scoring_rows = hybrid_validation.scoring_units

    scoring_units = _build_scoring_unit_audit(
        hotel,
        preferences,
        constraint_outcomes,
        arguments,
        hybrid_rows=hybrid_scoring_rows,
    )

    graph = build_argument_graph(
        list(arguments),
        root_base_score=ROOT_BASE_SCORE,
        root_text=ROOT_TEXT,
        target_item_name=hotel.metadata.name,
        allow_empty=True,
    )
    dfquad_result = evaluate_root_dfquad(graph)
    dfquad_payload = dfquad_result.to_dict()
    is_personalized = bool(preferences.active_aspect_preferences) or any(
        not constraint.hard and constraint.importance_raw > 0.0
        for constraint in preferences.constraints
    )
    scoring_status = (
        "no_soft_preferences"
        if not is_personalized
        else ("scored" if arguments else "no_usable_evidence")
    )
    dfquad_payload.update(
        {
            "weighting_method": ABSOLUTE_5_WEIGHTING_METHOD,
            "scoring_status": scoring_status,
            "is_personalized": is_personalized,
        }
    )
    dfquad_payload["node_scores"] = {
        node_id: {
            "initial_score": node.base_score,
            "final_score": (
                dfquad_result.final_score
                if node_id == graph.root_id
                else node.base_score
            ),
        }
        for node_id, node in graph.nodes.items()
    }

    hard_outcomes = tuple(
        outcome
        for outcome in constraint_outcomes
        if outcome.constraint.hard
    )

    if any(
        outcome.status is ConstraintStatus.VIOLATED
        for outcome in hard_outcomes
    ):
        eligibility_status = "ineligible"

    elif any(
        outcome.status is ConstraintStatus.UNKNOWN
        for outcome in hard_outcomes
    ):
        eligibility_status = "unknown"

    else:
        eligibility_status = "eligible"

    eligibility = EligibilityResult(
        status=eligibility_status,
        hard_constraints=hard_outcomes,
    )
    unknown_constraints = tuple(
        outcome
        for outcome in constraint_outcomes
        if outcome.status is ConstraintStatus.UNKNOWN
    )
    ineligibility_reasons = tuple(
        {
            "constraint_id": outcome.constraint.preference_ref,
            "reason": outcome.reason,
            "status": outcome.status.value,
        }
        for outcome in hard_outcomes
        if outcome.status is ConstraintStatus.VIOLATED
    )

    return HotelEvaluationResult(
        hotel_id=hotel.hotel_id,
        hotel_name=hotel.metadata.name,
        argument_mode=argument_mode,
        session_preferences=preferences,
        eligibility=eligibility,
        constraint_outcomes=constraint_outcomes,
        ineligibility_reasons=ineligibility_reasons,
        observed_preference_aspects=observed,
        missing_preference_aspects=missing,
        unknown_constraints=unknown_constraints,
        preference_coverage=coverage,
        linear_empirical_score=linear_score,
        dfquad_score=dfquad_result.final_score,
        weighting_method=ABSOLUTE_5_WEIGHTING_METHOD,
        scoring_status=scoring_status,
        is_personalized=is_personalized,
        arguments=arguments,
        scoring_units=scoring_units,
        graph=graph.to_dict(),
        dfquad=dfquad_payload,
        facility_normalization=facility_normalization.to_dict(),
        hybrid=hybrid_payload,
    )


def evaluate_hotel_by_id(
    dataset: HotelProfileDataset,
    hotel_id: str,
    preferences: SessionPreferences,
    *,
    argument_mode: str = "baseline",
    hybrid_generator: HybridArgumentGenerator | None = None,
    facility_ontology: FacilityOntology | None = None,
) -> HotelEvaluationResult:
    hotel = dataset.get_hotel(hotel_id)
    if hotel is None:
        raise KeyError(f"unknown hotel_id: {hotel_id}")
    return evaluate_hotel_session(
        hotel,
        preferences,
        argument_mode=argument_mode,
        hybrid_generator=hybrid_generator,
        facility_ontology=facility_ontology,
    )
