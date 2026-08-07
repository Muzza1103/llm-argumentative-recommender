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
                        "preference_refs",
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
    clean_preferences = preferences.to_dict()
    clean_preferences.pop("interpretation_trace", None)
    return f"""Propose a small set of contextual hotel arguments as JSON.

You are a language and selection component, never a fact checker or scorer.
Use only the exact source_id, preference reference, and scoring_unit_id values
provided below. Never invent a hotel field, review_id, facility_id, fact,
numeric strength, user importance, Wilson value, graph edge, or DF-QuAD score.

Rules:
- Atomic opinion/fact arguments may influence scoring only when they cite
  exactly one compatible scoring unit and every required source/preference of
  that unit. The deterministic code supplies its strength later.
- More wording or more citations never creates more score.
- contextual, tradeoff, and summary arguments must set explanatory_only=true.
- A composite using several units is explanatory only; do not sum units.
- Hard constraints are eligibility-only. They may be mentioned only in an
  explanatory-only statement and never as a scoring argument.
- Unknown information may be reported as uncertainty, never as an attack or
  proof of absence.
- Relations are suggestions only and will remain explanatory because the
  validated graph currently connects atomic arguments directly to the root.
- Return at most {MAX_HYBRID_ARGUMENTS} useful arguments. Do not force four.
- Return at most {MAX_HYBRID_RELATIONS} relations. Each argument must use
  1-{MAX_HYBRID_PREFERENCE_REFS} preference_refs,
  1-{MAX_HYBRID_SOURCE_REFS} source_refs, and at most
  {MAX_HYBRID_SCORING_UNIT_REFS} scoring_unit_refs.

Explanatory-only kinds:
{json.dumps(sorted(EXPLANATORY_KINDS))}

Validated preferences:
{json.dumps(clean_preferences, ensure_ascii=False, sort_keys=True)}

Compact hotel context:
{json.dumps(hotel_context, ensure_ascii=False, sort_keys=True)}

Constraint outcomes:
{json.dumps(constraint_outcomes, ensure_ascii=False, sort_keys=True)}

Authorized sources:
{json.dumps(authorized_sources, ensure_ascii=False, sort_keys=True)}

Available scoring units (intentionally without numeric strengths):
{json.dumps(scoring_units, ensure_ascii=False, sort_keys=True)}
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