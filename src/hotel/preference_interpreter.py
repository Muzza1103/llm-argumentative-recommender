from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .aspects import HOTEL_ASPECTS
from .errors import HotelPreferenceValidationError
from .preferences import SessionPreferences, session_preferences_from_dict


@runtime_checkable
class PreferenceInterpreter(Protocol):
    def interpret(self, text: str) -> SessionPreferences:
        """Interpret free text as validated session preferences."""
        ...


@runtime_checkable
class TextGenerator(Protocol):
    """Structural subset shared by existing local and Gemini generators."""

    def generate(self, prompt: str) -> str:
        ...


def build_preference_interpretation_prompt(text: str) -> str:
    aspects = "\n".join(f"- {aspect}" for aspect in HOTEL_ASPECTS)
    return f"""Extract hotel session preferences as one JSON object.

Allowed aspect names (use these exact strings only):
{aspects}

Return this structure:
{{
  "aspect_preferences": {{
    "allowed_aspect_name": {{
      "importance_raw": 0,
      "source_text": "exact supporting excerpt"
    }}
  }},
  "constraints": [
    {{
      "text": "constraint excerpt",
      "importance_raw": 0,
      "mode": "hard or soft",
      "field": "verifiable metadata field",
      "value": "optional expected value"
    }}
  ],
  "uninterpreted_items": []
}}

Importance must be a finite number from 0 to 5. Preserve anything that cannot
be mapped safely in uninterpreted_items. Do not invent hotel facts or silently
rename an unknown aspect. Return JSON only.

Preference text:
{text}
"""


def _extract_json_object(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise HotelPreferenceValidationError(
                "interpreter did not return a JSON object"
            )
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise HotelPreferenceValidationError(
                f"interpreter returned invalid JSON: {exc.msg}"
            ) from exc


@dataclass(slots=True)
class GeneratorPreferenceInterpreter:
    """Adapter for existing generators exposing ``generate(prompt)``."""

    generator: TextGenerator

    def interpret(self, text: str) -> SessionPreferences:
        if not isinstance(text, str) or not text.strip():
            raise HotelPreferenceValidationError(
                "preference text must be a non-empty string"
            )
        output = self.generator.generate(
            build_preference_interpretation_prompt(text.strip())
        )
        if not isinstance(output, str):
            raise HotelPreferenceValidationError(
                "interpreter generator must return text"
            )
        return session_preferences_from_dict(
            _extract_json_object(output),
            path="interpreter_output",
            original_text=text.strip(),
        )


def interpret_session_preferences(
    text: str,
    interpreter: PreferenceInterpreter,
) -> SessionPreferences:
    result = interpreter.interpret(text)
    if not isinstance(result, SessionPreferences):
        raise HotelPreferenceValidationError(
            "preference interpreter must return SessionPreferences"
        )
    return result
