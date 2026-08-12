from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .gemini_common import DEFAULT_GEMINI_MODEL, call_gemini_json
from .hybrid import (
    EXPLANATORY_KINDS,
    HYBRID_ARGUMENT_KINDS,
    HYBRID_ARGUMENT_TYPES,
    MAX_HYBRID_ARGUMENTS,
    MAX_HYBRID_PREFERENCE_REFS,
    MAX_HYBRID_RELATIONS,
    MAX_HYBRID_SCORING_UNIT_REFS,
    MAX_HYBRID_SOURCE_REFS,
)
from .preferences import SessionPreferences


_PROMPT_SCORING_FIELDS = frozenset(
    {
        "importance",
        "importance_raw",
        "normalized_weight",
        "weight",
        "weighting_method",
        "strength",
        "intrinsic_strength",
        "final_strength",
        "evidence_score",
        "confidence_factor",
        "wilson",
        "wilson_lower_bound",
        "force",
        "force_formula",
        "force_components",
        "final_force",
        "importance_coefficient",
        "score",
        "dfquad_score",
        "linear_empirical_score",
        "root_base_score",
        "aggregated_support",
        "aggregated_attack",
        "strength_method",
        "budget_included",
        "weight_active",
        "interpretation_trace",
        "raw_response",
    }
)


def _without_scoring_values(value: Any) -> Any:
    """Build a prompt-only copy without deterministic scoring information."""
    if isinstance(value, dict):
        return {
            key: _without_scoring_values(item)
            for key, item in value.items()
            if str(key).casefold() not in _PROMPT_SCORING_FIELDS
        }
    if isinstance(value, list):
        return [_without_scoring_values(item) for item in value]
    if isinstance(value, tuple):
        return [_without_scoring_values(item) for item in value]
    return value


def build_hybrid_argument_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": sorted(HYBRID_ARGUMENT_KINDS),
                        },
                        "type": {
                            "type": "string",
                            "enum": sorted(HYBRID_ARGUMENT_TYPES),
                        },
                        "text": {"type": "string"},
                        "preference_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "source_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "scoring_unit_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "explanatory_only": {"type": "boolean"},
                    },
                    "required": [
                        "id",
                        "kind",
                        "type",
                        "text",
                        "source_refs",
                        "scoring_unit_refs",
                        "explanatory_only",
                    ],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "source_argument_id": {"type": "string"},
                        "target_argument_id": {"type": "string"},
                        "relation_type": {
                            "type": "string",
                            "enum": [
                                "support",
                                "attack",
                                "qualifies",
                                "tradeoff",
                                "synthesis",
                            ],
                        },
                    },
                    "required": [
                        "id",
                        "source_argument_id",
                        "target_argument_id",
                        "relation_type",
                    ],
                },
            },
        },
        "required": ["arguments", "relations"],
    }


def build_hybrid_argument_prompt(
    *,
    preferences: SessionPreferences,
    hotel_context: dict[str, Any],
    authorized_sources: list[dict[str, Any]],
    scoring_units: list[dict[str, Any]],
    constraint_outcomes: list[dict[str, Any]],
) -> str:
    clean_preferences = _without_scoring_values(preferences.to_dict())
    clean_preferences.pop("interpretation_trace", None)
    clean_hotel_context = _without_scoring_values(hotel_context)
    clean_constraint_outcomes = _without_scoring_values(constraint_outcomes)
    clean_authorized_sources = _without_scoring_values(authorized_sources)
    clean_scoring_units = _without_scoring_values(scoring_units)
    return f"""Propose a small set of hotel arguments as JSON.

You are a language and selection component, never a fact checker or scorer.
Use only the exact source_id, preference reference, and scoring_unit_id values
provided below. Never invent a hotel field, review_id, facility_id, fact,
numeric scoring input, confidence coefficient, graph edge, or DF-QuAD score.

Your primary task is to verbalize the available deterministic scoring units.
For each useful scoring unit selected for the explanation:
- create one atomic opinion or fact argument;
- copy exactly one compatible scoring_unit_id into scoring_unit_refs;
- use every source required by that scoring unit;
- set explanatory_only=false.

Contextual, tradeoff, or summary arguments are optional secondary arguments
and must never replace the atomic scoring-unit arguments.

Rules:

- Atomic opinion/fact arguments may influence scoring only when they cite
  exactly one compatible scoring unit and every required source of that unit.
  The deterministic code derives preference_refs and supplies strength later.
- preference_refs is optional and non-authoritative whenever one valid
  scoring_unit_ref is supplied. If emitted, copy the unit's values exactly.
- More wording or more citations never creates more score.
- contextual, tradeoff, and summary arguments must set explanatory_only=true.
- Never attach several scoring units to one argument. Use one atomic argument
  per unit and express cross-unit synthesis through explanatory relations.
- Hard constraints are eligibility-only. They may be mentioned only in an
  explanatory-only statement and never as a scoring argument.
- Unknown information may be reported as uncertainty, never as an attack or
  proof of absence.
- Relations are suggestions only and will remain explanatory because the
  validated graph currently connects atomic arguments directly to the root.
- Return at most {MAX_HYBRID_ARGUMENTS} useful arguments. Do not force four.
- Return at most {MAX_HYBRID_RELATIONS} relations. Each argument must use
  1-{MAX_HYBRID_SOURCE_REFS} source_refs and at most
  {MAX_HYBRID_SCORING_UNIT_REFS} scoring_unit_refs.
- An argument without a scoring unit must use 1-{MAX_HYBRID_PREFERENCE_REFS}
  explicit preference_refs.

Explanatory-only kinds:
{json.dumps(sorted(EXPLANATORY_KINDS))}

Validated preferences:
{json.dumps(clean_preferences, ensure_ascii=False, sort_keys=True)}

Compact hotel context:
{json.dumps(clean_hotel_context, ensure_ascii=False, sort_keys=True)}

Constraint outcomes:
{json.dumps(clean_constraint_outcomes, ensure_ascii=False, sort_keys=True)}

Authorized sources:
{json.dumps(clean_authorized_sources, ensure_ascii=False, sort_keys=True)}

Available scoring units (identifiers and compatibility only):
{json.dumps(clean_scoring_units, ensure_ascii=False, sort_keys=True)}
"""


@dataclass(slots=True)
class GeminiHybridArgumentGenerator:
    model_name: str = DEFAULT_GEMINI_MODEL
    client: object | None = None
    api_key: str | None = None
    max_output_tokens: int = 4096
    thinking_budget: int = 1024
    last_trace: dict[str, Any] | None = field(default=None, init=False)

    def propose_arguments(
        self,
        *,
        preferences: SessionPreferences,
        hotel_context: dict[str, Any],
        authorized_sources: list[dict[str, Any]],
        scoring_units: list[dict[str, Any]],
        constraint_outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = call_gemini_json(
            prompt=build_hybrid_argument_prompt(
                preferences=preferences,
                hotel_context=hotel_context,
                authorized_sources=authorized_sources,
                scoring_units=scoring_units,
                constraint_outcomes=constraint_outcomes,
            ),
            response_schema=build_hybrid_argument_response_schema(),
            model_name=self.model_name,
            client=self.client,
            api_key=self.api_key,
            max_output_tokens=self.max_output_tokens,
            thinking_budget=self.thinking_budget,
        )
        self.last_trace = dict(result.trace)
        return result.payload
