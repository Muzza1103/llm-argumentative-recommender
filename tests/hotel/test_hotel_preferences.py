import json
import math
import unittest

from src.hotel import (
    GeneratorPreferenceInterpreter,
    HotelPreferenceValidationError,
    SessionPreferences,
    interpret_session_preferences,
    session_preferences_from_dict,
)


def preference_payload(**aspect_importances):
    return {
        "aspect_preferences": {
            aspect: {
                "importance_raw": importance,
                "source_text": f"Preference for {aspect}",
            }
            for aspect, importance in aspect_importances.items()
        },
        "constraints": [],
    }


class FakePreferenceInterpreter:
    def interpret(self, text: str) -> SessionPreferences:
        return session_preferences_from_dict(
            preference_payload(proprete_hygiene=5),
            original_text=text,
        )


class FakeGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps(self.payload)


class HotelPreferenceTests(unittest.TestCase):
    def test_rejects_unknown_aspect(self):
        payload = preference_payload(location=5)
        with self.assertRaisesRegex(
            HotelPreferenceValidationError,
            "unknown hotel aspect 'location'",
        ):
            session_preferences_from_dict(payload)

    def test_rejects_importance_outside_range(self):
        for value in (-0.01, 5.01):
            with self.subTest(value=value):
                with self.assertRaises(HotelPreferenceValidationError):
                    session_preferences_from_dict(
                        preference_payload(proprete_hygiene=value)
                    )

    def test_rejects_nan_and_infinity(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(HotelPreferenceValidationError):
                    session_preferences_from_dict(
                        preference_payload(proprete_hygiene=value)
                    )

    def test_normalizes_active_weights_and_ignores_zero(self):
        preferences = session_preferences_from_dict(
            preference_payload(
                proprete_hygiene=5,
                bruit_calme=3,
                wifi_internet=0,
            )
        )

        self.assertAlmostEqual(
            preferences.get_aspect("proprete_hygiene").normalized_weight,
            5 / 8,
        )
        self.assertAlmostEqual(
            preferences.get_aspect("bruit_calme").normalized_weight,
            3 / 8,
        )
        self.assertEqual(
            preferences.get_aspect("wifi_internet").normalized_weight,
            0.0,
        )
        self.assertEqual(
            [item.aspect for item in preferences.active_aspect_preferences],
            ["proprete_hygiene", "bruit_calme"],
        )

    def test_preserves_constraints_original_text_and_unknown_items(self):
        payload = preference_payload(proprete_hygiene=5)
        payload["constraints"] = [
            {
                "text": "Parking is required",
                "importance_raw": 5,
                "mode": "hard",
                "field": "parking",
                "unmapped_detail": "covered",
            }
        ]
        payload["unknown_items"] = ["sea-view mood"]
        payload["future_field"] = {"raw": True}
        preferences = session_preferences_from_dict(
            payload,
            original_text="Original preference",
        )

        self.assertEqual(preferences.original_text, "Original preference")
        self.assertTrue(preferences.constraints[0].hard)
        self.assertEqual(
            preferences.constraints[0].uninterpreted,
            {"unmapped_detail": "covered"},
        )
        self.assertEqual(preferences.uninterpreted_items[0], "sea-view mood")
        self.assertEqual(
            preferences.uninterpreted_items[1]["field"],
            "future_field",
        )

    def test_injectable_interpreter_needs_no_llm(self):
        preferences = interpret_session_preferences(
            "A very clean hotel",
            FakePreferenceInterpreter(),
        )
        self.assertEqual(preferences.original_text, "A very clean hotel")
        self.assertEqual(
            preferences.active_aspect_preferences[0].aspect,
            "proprete_hygiene",
        )

    def test_generator_adapter_validates_llm_output_strictly(self):
        generator = FakeGenerator(preference_payload(wifi_internet=2))
        interpreter = GeneratorPreferenceInterpreter(generator)
        preferences = interpreter.interpret("Wi-Fi would be useful")

        self.assertEqual(preferences.original_text, "Wi-Fi would be useful")
        self.assertIn("wifi_internet", generator.last_prompt)

        invalid = GeneratorPreferenceInterpreter(
            FakeGenerator(preference_payload(internet=2))
        )
        with self.assertRaises(HotelPreferenceValidationError):
            invalid.interpret("Internet")


if __name__ == "__main__":
    unittest.main()
