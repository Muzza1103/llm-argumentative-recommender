from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from .aspects import HOTEL_ASPECTS
from .errors import HotelPreferenceValidationError
from .facility_ontology import (
    FacilityOntology,
    load_default_facility_ontology,
)
from .gemini_common import DEFAULT_GEMINI_MODEL, call_gemini_json
from .intent_deduplication import (
    MAX_ASPECT_PREFERENCES,
    MAX_PREFERENCE_CONSTRAINTS,
    MAX_UNINTERPRETED_ITEMS,
    filter_structurally_valid_preferences,
)
from .preferences import SessionPreferences, session_preferences_from_dict


_UNINTERPRETED_REASONS = (
    "unsupported_facility_target",
    "unsupported_aspect",
    "unsupported_operator",
    "ambiguous_request",
    "insufficient_information",
)

def build_preference_response_schema(
    ontology: FacilityOntology,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "aspect_preferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "aspect": {
                            "type": "string",
                        },
                        "importance_raw": {
                            "type": "number",
                        },
                        "source_text": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "aspect",
                        "importance_raw",
                        "source_text",
                    ],
                },
            },
            "constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "constraint_id": {"type": "string"},
                        "target_type": {
                            "type": "string",
                            "enum": ["facility", "metadata"],
                        },
                        "target": {
                            "type": "string",
                        },
                        "operator": {
                            "type": "string",
                            "enum": list(ontology.operators),
                        },
                        "qualifiers": {
                            "type": "object",
                        },
                        "value": {
                            "type": ["string", "null"],
                        },
                        "hard": {"type": "boolean"},
                        "importance_raw": {
                            "type": "number",
                        },
                        "source_text": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "constraint_id",
                        "target_type",
                        "target",
                        "operator",
                        "qualifiers",
                        "value",
                        "hard",
                        "importance_raw",
                        "source_text",
                    ],
                },
            },
            "uninterpreted_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "reason": {
                            "type": "string",
                            "enum": list(_UNINTERPRETED_REASONS),
                        },
                    },
                    "required": ["text", "reason"],
                },
            },
        },
        "required": [
            "aspect_preferences",
            "constraints",
            "uninterpreted_items",
        ],
    }


def build_preference_prompt_contract(
    ontology: FacilityOntology,
) -> str:
    contract = ontology.compact_prompt_contract()
    lines = [
        "Allowed facility targets:",
        ", ".join(contract["capabilities"]),
        "",
        "Allowed metadata targets:",
        "city",
        "",
        "Allowed operators:",
        ", ".join(contract["operators"]),
        "",
        "Allowed qualifiers by facility:",
    ]
    for facility, specification in contract["capabilities"].items():
        qualifier_parts = []
        for qualifier, values in specification["qualifiers"].items():
            rendered_values = ", ".join(
                json.dumps(value, ensure_ascii=False) for value in values
            )
            qualifier_parts.append(f"{qualifier}=[{rendered_values}]")
        rendered_qualifiers = ", ".join(qualifier_parts) or "(none)"
        lines.append(f"- {facility}: {rendered_qualifiers}")
    return "\n".join(lines)


def build_gemini_preference_prompt(
    text: str,
    ontology: FacilityOntology,
) -> str:
    contract = build_preference_prompt_contract(ontology)
    return f"""Translate the user's hotel request into the closed JSON schema.

Rules:
- You are the only semantic interpreter of the user's request. Downstream
  Python code validates structure but will not reinterpret or correct your
  semantic decisions.
- Use only the 15 aspect names and canonical facility capabilities supplied.
- Never choose or output a raw facility_id or raw facility name.
- Decide whether every factual constraint is hard or soft from the full user
  meaning. Set hard=true only for an explicit necessity, obligation, absolute
  refusal, non-negotiable condition, or indispensable requirement. The word
  "important" alone is not sufficient.
- Explicitly negated necessities such as "pas obligatoire", "pas
  indispensable", "not mandatory", and "not required" must use hard=false.
- Use importance_raw on the fixed 0-to-5 scale: 0=irrelevant, 1=very weak,
  2=optional or nice to have, 3=ordinary preference, 4=very important but
  still soft, and 5=indispensable or hard requirement.
- Subjective quality belongs in aspect_preferences: "good/reliable/fast
  Wi-Fi" maps to wifi_internet without an additional Wi-Fi-presence
  constraint; "convenient/difficult-to-access parking" maps to
  parking_voiture only.
- Factual presence, absence, value, or qualifier belongs in constraints when
  it can be expressed by the supplied closed contract. Otherwise preserve it
  in uninterpreted_items. "Wi-Fi available/with Wi-Fi/free Wi-Fi" and
  "parking would be useful/I would like parking" must not also create their
  corresponding aspect.
- A sentence with two distinct needs may create two entries.  For example,
  "free and fast Wi-Fi" is both a price-qualified Wi-Fi fact and a qualitative
  wifi_internet preference.  Reuse the relevant source excerpt for each.
- Use target_type=facility, operator=present, and value=null for canonical
  facilities. Use target_type=metadata, target=city, operator=equals, and an
  explicit city value for city requirements.
- For each hard constraint, source_text must be one exact contiguous excerpt
  containing both the requested target and the wording that makes it
  mandatory. It may be a complete clause or sentence when necessary.
- Do not shorten source_text in a way that removes words such as mandatory,
  required, indispensable, obligatoire, impératif, not mandatory, or pas
  obligatoire. Otherwise use the smallest exact contiguous excerpt that fully
  contains the intention. Preserve the user's wording exactly.
- Represent each semantic intention exactly once. Do not split one requirement
  into a target-only constraint and a second necessity-only constraint. Do not
  return two constraints with the same target, operator, value, and qualifiers.
- Do not create both an aspect and a constraint for the same wording unless
  the text explicitly expresses two distinct intentions.
- Preserve unsupported or ambiguous requests in uninterpreted_items instead
  of approximating them.
- Return at most {MAX_ASPECT_PREFERENCES} aspect preferences,
  {MAX_PREFERENCE_CONSTRAINTS} constraints, and
  {MAX_UNINTERPRETED_ITEMS} uninterpreted items.

Allowed aspects:
{json.dumps(list(HOTEL_ASPECTS), ensure_ascii=False)}

Canonical facility contract (not raw provider facilities):
{contract}

User text:
{text}
"""


def validate_preference_structure(
    payload: object,
    *,
    original_text: str,
    ontology: FacilityOntology,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Validate and filter Gemini output without semantic reinterpretation."""
    return filter_structurally_valid_preferences(
        payload,
        original_text=original_text,
        ontology=ontology,
        uninterpreted_reasons=_UNINTERPRETED_REASONS,
    )


def _validate_interpretation_payload(
    payload: object,
    *,
    original_text: str,
    ontology: FacilityOntology,
) -> dict[str, Any]:
    """Backward-compatible validation helper returning the clean payload."""
    validated, _ = validate_preference_structure(
        payload,
        original_text=original_text,
        ontology=ontology,
    )
    return validated


@dataclass(slots=True)
class GeminiPreferenceInterpreter:
    model_name: str = DEFAULT_GEMINI_MODEL
    ontology: FacilityOntology = field(
        default_factory=load_default_facility_ontology
    )
    client: object | None = None
    api_key: str | None = None
    max_output_tokens: int = 4096
    thinking_budget: int = 1024
    last_trace: dict[str, Any] | None = field(default=None, init=False)

    def interpret(self, text: str) -> SessionPreferences:
        if not isinstance(text, str) or not text.strip():
            raise HotelPreferenceValidationError(
                "preference text must be a non-empty string"
            )
        schema = build_preference_response_schema(self.ontology)
        result = call_gemini_json(
            prompt=build_gemini_preference_prompt(text, self.ontology),
            response_schema=schema,
            model_name=self.model_name,
            client=self.client,
            api_key=self.api_key,
            max_output_tokens=self.max_output_tokens,
            thinking_budget=self.thinking_budget,
        )
        validated, structural_validation_trace = validate_preference_structure(
            result.payload,
            original_text=text,
            ontology=self.ontology,
        )
        preferences = session_preferences_from_dict(
            validated,
            path="interpreter_output",
            original_text=text,
        )
        trace = {
            **dict(result.trace),
            "structural_validation": [
                dict(decision)
                for decision in structural_validation_trace
            ],
        }
        self.last_trace = trace
        return replace(
            preferences,
            interpretation_trace=trace,
        )
