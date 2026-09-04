from __future__ import annotations

from typing import Any

from src.hotel import SessionPreferences, session_preferences_from_dict


class FakePreferenceInterpreter:
    def interpret(self, text: str) -> SessionPreferences:
        return session_preferences_from_dict(
            {
                "aspect_preferences": {
                    "localisation_transport": {
                        "importance_raw": 5,
                        "source_text": text,
                    }
                },
                "constraints": [
                    {
                        "text": text,
                        "source_text": text,
                        "importance_raw": 3,
                        "mode": "soft",
                        "field": "wifi",
                    }
                ],
            },
            original_text=text,
        )


class FakeHybridArgumentGenerator:
    last_trace = {
        "model_requested": "fake-offline-generator",
        "usage": {},
        "request_count": 0,
    }

    def propose_arguments(
        self,
        *,
        preferences: SessionPreferences,
        hotel_context: dict[str, Any],
        authorized_sources: list[dict[str, Any]],
        scoring_units: list[dict[str, Any]],
        constraint_outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del preferences, hotel_context, authorized_sources, constraint_outcomes
        arguments = []
        for index, unit in enumerate(scoring_units[:3], start=1):
            arguments.append(
                {
                    "id": f"FAKE_{index}",
                    "kind": unit["kind"],
                    "type": unit["type"],
                    "text": f"Offline grounded argument {index}.",
                    "preference_refs": unit["preference_refs"],
                    "source_refs": unit["source_refs"],
                    "scoring_unit_refs": [unit["scoring_unit_id"]],
                    "explanatory_only": False,
                }
            )
        return {"arguments": arguments, "relations": []}


class FakeHybridArgumentGeneratorFactory:
    def __new__(cls):
        return FakeHybridArgumentGenerator()
