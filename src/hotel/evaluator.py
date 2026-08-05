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
from .models import HotelProfile, HotelProfileDataset
from .preferences import SessionPreferences


ROOT_TEXT = "Recommend this hotel for the current session"
ROOT_BASE_SCORE = 0.5


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
    session_preferences: SessionPreferences
    eligibility: EligibilityResult
    observed_preference_aspects: tuple[str, ...]
    missing_preference_aspects: tuple[str, ...]
    unknown_constraints: tuple[ConstraintOutcome, ...]
    preference_coverage: float
    linear_empirical_score: float | None
    dfquad_score: float
    arguments: tuple[Argument, ...]
    graph: dict[str, Any]
    dfquad: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "hotel_name": self.hotel_name,
            "session_preferences": self.session_preferences.to_dict(),
            "eligibility": self.eligibility.to_dict(),
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
) -> HotelEvaluationResult:
    observed, missing, coverage, linear_score = _empirical_baseline(
        hotel,
        preferences,
    )
    empirical_arguments = build_empirical_arguments(hotel, preferences)
    constraint_outcomes = evaluate_constraints(hotel, preferences.constraints)
    factual_arguments = build_structured_fact_arguments(
        hotel,
        constraint_outcomes,
    )
    arguments = tuple(empirical_arguments + factual_arguments)

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

    return HotelEvaluationResult(
        hotel_id=hotel.hotel_id,
        hotel_name=hotel.metadata.name,
        session_preferences=preferences,
        eligibility=eligibility,
        observed_preference_aspects=observed,
        missing_preference_aspects=missing,
        unknown_constraints=unknown_constraints,
        preference_coverage=coverage,
        linear_empirical_score=linear_score,
        dfquad_score=dfquad_result.final_score,
        arguments=arguments,
        graph=graph.to_dict(),
        dfquad=dfquad_payload,
    )


def evaluate_hotel_by_id(
    dataset: HotelProfileDataset,
    hotel_id: str,
    preferences: SessionPreferences,
) -> HotelEvaluationResult:
    hotel = dataset.get_hotel(hotel_id)
    if hotel is None:
        raise KeyError(f"unknown hotel_id: {hotel_id}")
    return evaluate_hotel_session(hotel, preferences)
