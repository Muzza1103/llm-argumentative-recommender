from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .aspects import HOTEL_ASPECTS
from .errors import HotelPreferenceValidationError
from .facility_ontology import (
    FacilityOntology,
    load_default_facility_ontology,
)
from .gemini_common import DEFAULT_GEMINI_MODEL, call_gemini_json
from .intent_deduplication import deduplicate_preference_intentions
from .preferences import SessionPreferences, session_preferences_from_dict


_UNINTERPRETED_REASONS = (
    "unsupported_facility_target",
    "unsupported_aspect",
    "unsupported_operator",
    "ambiguous_request",
    "insufficient_information",
)

MAX_ASPECT_PREFERENCES = 15
MAX_PREFERENCE_CONSTRAINTS = 12
MAX_UNINTERPRETED_ITEMS = 20


def _normalized_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _assert_excerpt(source_text: object, original_text: str, path: str) -> str:
    if not isinstance(source_text, str) or not source_text.strip():
        raise HotelPreferenceValidationError(
            "expected a non-empty source excerpt",
            path=path,
        )
    normalized_source = _normalized_excerpt(source_text)
    if normalized_source not in _normalized_excerpt(original_text):
        raise HotelPreferenceValidationError(
            "source excerpt is not present in the original user text",
            path=path,
        )
    return source_text.strip()


def _importance(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HotelPreferenceValidationError(
            "importance must be a number from 0 to 5",
            path=path,
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 5.0:
        raise HotelPreferenceValidationError(
            "importance must be a finite number from 0 to 5",
            path=path,
        )
    return result


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
- Use only the 15 aspect names and canonical facility capabilities supplied.
- Never choose or output a raw facility_id or raw facility name.
- Use a hard constraint only for an explicit necessity, obligation, absolute
  refusal, or indispensable requirement. The word "important" alone is not
  sufficient. An explicitly negated necessity such as "not mandatory",
  "not required", "pas obligatoire", or "pas indispensable" is soft.
- Subjective quality is an aspect preference only: "good/reliable/fast Wi-Fi"
  maps to wifi_internet without an additional Wi-Fi-presence constraint;
  "convenient/difficult-to-access parking" maps to parking_voiture only.
- Availability, requested presence, or a factual qualifier is a constraint
  only: "Wi-Fi available/with Wi-Fi/free Wi-Fi" and "parking would be useful/I
  would like parking" must not also create their corresponding aspect.
- A sentence with two distinct needs may create two entries.  For example,
  "free and fast Wi-Fi" is both a price-qualified Wi-Fi fact and a qualitative
  wifi_internet preference.  Reuse the relevant source excerpt for each.
- Use target_type=facility, operator=present, and value=null for canonical
  facilities. Use target_type=metadata, target=city, operator=equals, and an
  explicit city value for city requirements.
- For each entry, copy the smallest exact contiguous source_text excerpt that
  contains that intent. Keep one intention per source_text: never copy an
  unrelated sentence or proposition merely for context. Preserve the user's
  wording exactly. Preserve unsupported or ambiguous requests in
  uninterpreted_items instead of approximating them.
- Importance is calibrated from source_text: ordinary=3, optional or
  "would be useful/nice to have/de préférence"=2, explicitly very important
  but still soft=4, and an absolute necessity=5. Do not assign 5 to an
  ordinary request. Local deterministic code verifies and corrects this value.
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


def _validate_interpretation_payload(
    payload: object,
    *,
    original_text: str,
    ontology: FacilityOntology,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HotelPreferenceValidationError(
            "interpreter output must be an object"
        )
    expected_root = {
        "aspect_preferences",
        "constraints",
        "uninterpreted_items",
    }
    if set(payload) != expected_root:
        raise HotelPreferenceValidationError(
            "interpreter output has missing or unknown root fields"
        )
    aspects = payload["aspect_preferences"]
    constraints = payload["constraints"]
    uninterpreted = payload["uninterpreted_items"]
    if not isinstance(aspects, list) or not isinstance(constraints, list):
        raise HotelPreferenceValidationError(
            "aspect_preferences and constraints must be lists"
        )
    if not isinstance(uninterpreted, list):
        raise HotelPreferenceValidationError(
            "uninterpreted_items must be a list"
        )
    if len(aspects) > MAX_ASPECT_PREFERENCES:
        raise HotelPreferenceValidationError(
            "interpreter output has too many aspect preferences"
        )
    if len(constraints) > MAX_PREFERENCE_CONSTRAINTS:
        raise HotelPreferenceValidationError(
            "interpreter output has too many constraints"
        )
    if len(uninterpreted) > MAX_UNINTERPRETED_ITEMS:
        raise HotelPreferenceValidationError(
            "interpreter output has too many uninterpreted items"
        )

    aspect_mapping = {}
    for index, entry in enumerate(aspects):
        path = f"interpreter_output.aspect_preferences[{index}]"
        if not isinstance(entry, dict) or set(entry) != {
            "aspect",
            "importance_raw",
            "source_text",
        }:
            raise HotelPreferenceValidationError(
                "aspect entry has missing or unknown fields",
                path=path,
            )
        aspect = entry["aspect"]
        if aspect not in HOTEL_ASPECTS:
            raise HotelPreferenceValidationError(
                f"unknown hotel aspect {aspect!r}",
                path=f"{path}.aspect",
            )
        if aspect in aspect_mapping:
            raise HotelPreferenceValidationError(
                f"duplicate aspect {aspect!r}",
                path=f"{path}.aspect",
            )
        aspect_mapping[aspect] = {
            "importance_raw": _importance(
                entry["importance_raw"],
                f"{path}.importance_raw",
            ),
            "source_text": _assert_excerpt(
                entry["source_text"],
                original_text,
                f"{path}.source_text",
            ),
        }

    constraint_rows = []
    seen_constraint_ids = set()
    expected_constraint_fields = {
        "constraint_id",
        "target_type",
        "target",
        "operator",
        "qualifiers",
        "value",
        "hard",
        "importance_raw",
        "source_text",
    }
    for index, entry in enumerate(constraints):
        path = f"interpreter_output.constraints[{index}]"
        if not isinstance(entry, dict) or set(entry) != expected_constraint_fields:
            raise HotelPreferenceValidationError(
                "constraint has missing or unknown fields",
                path=path,
            )
        constraint_id = entry["constraint_id"]
        if not isinstance(constraint_id, str) or not constraint_id.strip():
            raise HotelPreferenceValidationError(
                "constraint_id must be a non-empty string",
                path=f"{path}.constraint_id",
            )
        if constraint_id in seen_constraint_ids:
            raise HotelPreferenceValidationError(
                f"duplicate constraint_id {constraint_id!r}",
                path=f"{path}.constraint_id",
            )
        seen_constraint_ids.add(constraint_id)
        target_type = entry["target_type"]
        target = entry["target"]
        operator = entry["operator"]
        qualifiers = entry["qualifiers"]
        value = entry["value"]
        if not isinstance(entry["hard"], bool):
            raise HotelPreferenceValidationError(
                "hard must be a boolean",
                path=f"{path}.hard",
            )
        if not isinstance(target_type, str) or target_type not in {
            "facility",
            "metadata",
        }:
            raise HotelPreferenceValidationError(
                "target_type must be facility or metadata",
                path=f"{path}.target_type",
            )
        if not isinstance(target, str) or not target.strip():
            raise HotelPreferenceValidationError(
                "target must be a non-empty string",
                path=f"{path}.target",
            )
        if operator not in ontology.operators:
            raise HotelPreferenceValidationError(
                f"unsupported operator {operator!r}",
                path=f"{path}.operator",
            )
        if target_type == "facility":
            if target not in ontology.capabilities:
                raise HotelPreferenceValidationError(
                    f"unsupported facility target {target!r}",
                    path=f"{path}.target",
                )
            if operator != "present" or value is not None:
                raise HotelPreferenceValidationError(
                    "facility constraints require present and value=null",
                    path=path,
                )
            errors = ontology.qualifier_errors(target, qualifiers)
            if errors:
                raise HotelPreferenceValidationError(
                    "; ".join(errors),
                    path=f"{path}.qualifiers",
                )
        elif target_type == "metadata":
            if target != "city" or operator != "equals":
                raise HotelPreferenceValidationError(
                    "only metadata city with equals is supported",
                    path=path,
                )
            if qualifiers != {}:
                raise HotelPreferenceValidationError(
                    "city constraints do not accept qualifiers",
                    path=f"{path}.qualifiers",
                )
            if not isinstance(value, str) or not value.strip():
                raise HotelPreferenceValidationError(
                    "city requires an explicit structured value",
                    path=f"{path}.value",
                )
        source_text = _assert_excerpt(
            entry["source_text"],
            original_text,
            f"{path}.source_text",
        )
        constraint_rows.append(
            {
                "constraint_id": constraint_id,
                "target_type": target_type,
                "target": target,
                "operator": operator,
                "qualifiers": dict(qualifiers),
                "value": value,
                "hard": entry["hard"],
                "importance_raw": _importance(
                    entry["importance_raw"],
                    f"{path}.importance_raw",
                ),
                "source_text": source_text,
                "text": source_text,
            }
        )

    clean_uninterpreted = []
    for index, entry in enumerate(uninterpreted):
        path = f"interpreter_output.uninterpreted_items[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"text", "reason"}:
            raise HotelPreferenceValidationError(
                "uninterpreted item has missing or unknown fields",
                path=path,
            )
        if entry["reason"] not in _UNINTERPRETED_REASONS:
            raise HotelPreferenceValidationError(
                "unknown uninterpreted reason",
                path=f"{path}.reason",
            )
        clean_uninterpreted.append(
            {
                "text": _assert_excerpt(
                    entry["text"],
                    original_text,
                    f"{path}.text",
                ),
                "reason": entry["reason"],
            }
        )

    return {
        "aspect_preferences": aspect_mapping,
        "constraints": constraint_rows,
        "uninterpreted_items": clean_uninterpreted,
    }


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
        validated = _validate_interpretation_payload(
            result.payload,
            original_text=text,
            ontology=self.ontology,
        )
        deduplicated, deduplication_trace = (
            deduplicate_preference_intentions(validated)
        )
        preferences = session_preferences_from_dict(
            deduplicated,
            path="interpreter_output",
            original_text=text,
        )
        trace = {
            **dict(result.trace),
            "deduplication": [
                dict(decision) for decision in deduplication_trace
            ],
        }
        self.last_trace = trace
        return replace(
            preferences,
            interpretation_trace=trace,
        )
