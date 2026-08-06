from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.argumentation.dfquad import evaluate_root_dfquad
from src.argumentation.graph_builder import build_argument_graph
from src.argumentation.schema import Argument

from .argument_builder import (
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
from .preferences import SessionPreferences


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
    arguments: tuple[Argument, ...]
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
            "arguments": [argument.to_dict() for argument in self.arguments],
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

    graph = build_argument_graph(
        list(arguments),
        root_base_score=ROOT_BASE_SCORE,
        root_text=ROOT_TEXT,
        target_item_name=hotel.metadata.name,
        allow_empty=True,
    )
    dfquad_result = evaluate_root_dfquad(graph)
    dfquad_payload = dfquad_result.to_dict()
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
    eligibility = EligibilityResult(
        status=(
            "ineligible"
            if any(
                outcome.status is ConstraintStatus.VIOLATED
                for outcome in hard_outcomes
            )
            else "eligible"
        ),
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
        arguments=arguments,
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
