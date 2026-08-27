import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.hotel import (
    GeminiHybridArgumentGenerator,
    GeminiPreferenceInterpreter,
    HotelGeminiError,
    HotelHybridValidationError,
    evaluate_hotel_session,
    load_hotel_profiles,
    prepare_hybrid_context,
    session_preferences_from_dict,
)
from tests.hotel.fake_components import FakeHybridArgumentGenerator


FIXTURE = Path(__file__).parent / "fixtures" / "hotel_profiles_minimal.json"


def preferences(aspects=None, constraints=None, original_text=None):
    return session_preferences_from_dict(
        {
            "aspect_preferences": {
                aspect: {
                    "importance_raw": importance,
                    "source_text": f"Need {aspect}",
                }
                for aspect, importance in (aspects or {}).items()
            },
            "constraints": constraints or [],
        },
        original_text=original_text,
    )


class StaticGenerator:
    def __init__(self, factory):
        self.factory = factory
        self.last_trace = {"model_requested": "fake", "request_count": 0}

    def propose_arguments(self, **kwargs):
        return self.factory(**kwargs)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


class HotelHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def test_pipeline_is_deterministic_non_mutating_and_scores_soft_fact(self):
        profile = preferences(
            {"localisation_transport": 5},
            constraints=[
                {
                    "text": "Wi-Fi would be useful",
                    "importance_raw": 3,
                    "mode": "soft",
                    "field": "wifi",
                }
            ],
        )
        hotel_before = repr(self.hotel)
        preferences_before = json.dumps(profile.to_dict(), sort_keys=True)
        first = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=FakeHybridArgumentGenerator(),
        )
        second = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=FakeHybridArgumentGenerator(),
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.dfquad_score, second.dfquad_score)
        self.assertEqual(repr(self.hotel), hotel_before)
        self.assertEqual(
            json.dumps(profile.to_dict(), sort_keys=True),
            preferences_before,
        )
        self.assertTrue(
            any(
                argument.argument_family == "structured_fact"
                for argument in first.arguments
            )
        )
        self.assertEqual(first.argument_mode, "hybrid")
        json.dumps(first.to_dict())

    def test_invalid_sources_and_hallucinated_fields_are_rejected(self):
        profile = preferences({"localisation_transport": 5})
        prepared = prepare_hybrid_context(self.hotel, profile)
        unit = prepared.scoring_units[0]

        def payload(**_):
            common = {
                "kind": unit.kind,
                "type": unit.arg_type,
                "text": "A proposed argument.",
                "preference_refs": list(unit.preference_refs),
                "scoring_unit_refs": [unit.unit_id],
                "explanatory_only": False,
            }
            return {
                "arguments": [
                    {
                        **common,
                        "id": "BAD_SOURCE",
                        "source_refs": ["REVIEW::invented"],
                    },
                    {
                        **common,
                        "id": "BAD_FIELD",
                        "source_refs": list(unit.source_refs),
                        "hotel_field": "invented_pool_count",
                    },
                    {
                        **common,
                        "id": "BAD_UNIT",
                        "source_refs": list(unit.source_refs),
                        "scoring_unit_refs": ["UNIT::invented"],
                    },
                    {
                        **common,
                        "id": "BAD_POLARITY",
                        "kind": "contextual",
                        "type": (
                            "attack" if unit.arg_type == "support" else "support"
                        ),
                        "source_refs": list(unit.source_refs),
                        "scoring_unit_refs": [],
                        "explanatory_only": True,
                    },
                ],
                "relations": [],
            }

        result = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=StaticGenerator(payload),
        )
        rejected = result.hybrid["validation"]["rejected_arguments"]
        reasons = {reason for item in rejected for reason in item["reasons"]}
        self.assertIn("unknown_source_ref", reasons)
        self.assertIn("unknown_scoring_unit_ref", reasons)
        self.assertIn("hallucinated_hotel_field", reasons)
        self.assertIn("polarity_source_mismatch", reasons)
        self.assertEqual(result.arguments, ())
        self.assertEqual(result.dfquad_score, 0.5)

    def test_hard_constraints_and_unknown_information_never_score(self):
        profile = preferences(
            constraints=[
                {
                    "text": "Wi-Fi is mandatory",
                    "importance_raw": 5,
                    "mode": "hard",
                    "field": "wifi",
                },
                {
                    "text": "Parking is mandatory",
                    "importance_raw": 5,
                    "mode": "hard",
                    "field": "parking",
                },
            ]
        )

        def payload(**kwargs):
            sources = {
                source["source_id"]: source
                for source in kwargs["authorized_sources"]
            }
            self.assertIn("CONSTRAINT::hard_01", sources)
            self.assertIn("CONSTRAINT::hard_02", sources)
            return {
                "arguments": [
                    {
                        "id": "HARD",
                        "kind": "fact",
                        "type": "support",
                        "text": "The hard constraint supports the score.",
                        "preference_refs": ["hard_01"],
                        "source_refs": ["CONSTRAINT::hard_01"],
                        "scoring_unit_refs": [],
                        "explanatory_only": False,
                    },
                    {
                        "id": "UNKNOWN_ATTACK",
                        "kind": "contextual",
                        "type": "attack",
                        "text": "Missing parking is negative.",
                        "preference_refs": ["hard_02"],
                        "source_refs": ["CONSTRAINT::hard_02"],
                        "scoring_unit_refs": [],
                        "explanatory_only": True,
                    },
                ],
                "relations": [],
            }

        result = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=StaticGenerator(payload),
        )
        statuses = [
            outcome.status.value for outcome in result.constraint_outcomes
        ]
        self.assertEqual(statuses, ["satisfied", "unknown"])
        self.assertEqual(result.eligibility.status, "unknown")
        self.assertEqual(result.arguments, ())
        self.assertEqual(list(result.graph["nodes"]), ["ROOT"])
        self.assertEqual(result.dfquad_score, 0.5)
        reasons = {
            reason
            for item in result.hybrid["validation"]["rejected_arguments"]
            for reason in item["reasons"]
        }
        self.assertIn("hard_constraint_excluded", reasons)
        self.assertIn("unknown_information_used_as_negative", reasons)

    def test_duplicate_units_and_summaries_do_not_double_count(self):
        profile = preferences({"localisation_transport": 5})
        prepared = prepare_hybrid_context(self.hotel, profile)
        unit = next(
            item
            for item in prepared.scoring_units
            if item.arg_type == "support"
        )

        def payload(**_):
            atomic = {
                "kind": unit.kind,
                "type": unit.arg_type,
                "preference_refs": list(unit.preference_refs),
                "source_refs": list(unit.source_refs),
                "scoring_unit_refs": [unit.unit_id],
                "explanatory_only": False,
            }
            return {
                "arguments": [
                    {**atomic, "id": "ONE", "text": "First wording."},
                    {**atomic, "id": "TWO", "text": "Second wording."},
                    {
                        **atomic,
                        "id": "SUMMARY",
                        "kind": "summary",
                        "text": "A readable synthesis.",
                    },
                ],
                "relations": [
                    {
                        "id": "R1",
                        "source_argument_id": "ONE",
                        "target_argument_id": "SUMMARY",
                        "relation_type": "synthesis",
                    }
                ],
            }

        result = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=StaticGenerator(payload),
        )
        self.assertEqual(len(result.arguments), 1)
        self.assertEqual(result.arguments[0].id, "ONE")
        self.assertEqual(result.dfquad["support_scores"], [unit.intrinsic_strength])
        excluded = result.hybrid["validation"]["excluded_arguments"]
        reasons = {item["reason"] for item in excluded}
        self.assertIn("duplicate_scoring_unit", reasons)
        self.assertIn("composite_or_explanatory_only", reasons)
        relation = result.hybrid["validation"]["relations"][0]
        self.assertTrue(relation["accepted"])
        self.assertTrue(relation["explanatory_only"])
        self.assertEqual(len(result.graph["edges"]), 1)

    def test_hybrid_reference_limits_are_enforced_locally(self):
        profile = preferences({"localisation_transport": 5})
        prepared = prepare_hybrid_context(self.hotel, profile)
        unit = prepared.scoring_units[0]

        def payload(**_):
            common = {
                "kind": unit.kind,
                "type": unit.arg_type,
                "text": "A bounded proposal.",
                "preference_refs": list(unit.preference_refs),
                "source_refs": list(unit.source_refs),
                "scoring_unit_refs": [unit.unit_id],
                "explanatory_only": False,
            }
            return {
                "arguments": [
                    {
                        **common,
                        "id": "TOO_MANY_PREFERENCES",
                        "preference_refs": [unit.preference_refs[0]] * 6,
                    },
                    {
                        **common,
                        "id": "TOO_MANY_SOURCES",
                        "source_refs": [unit.source_refs[0]] * 9,
                    },
                    {
                        **common,
                        "id": "TOO_MANY_UNITS",
                        "scoring_unit_refs": [unit.unit_id] * 4,
                    },
                ],
                "relations": [],
            }

        result = evaluate_hotel_session(
            self.hotel,
            profile,
            argument_mode="hybrid",
            hybrid_generator=StaticGenerator(payload),
        )
        rejected = result.hybrid["validation"]["rejected_arguments"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all("invalid_argument_schema" in row["reasons"] for row in rejected)
        )
        self.assertEqual(len(result.arguments), 1)
        self.assertEqual(result.arguments[0].id, "TOO_MANY_PREFERENCES")
        self.assertEqual(
            result.arguments[0].preference_refs,
            list(unit.preference_refs),
        )
        self.assertNotIn(
            "preference_scoring_unit_mismatch",
            {
                reason
                for row in rejected
                for reason in row["reasons"]
            },
        )

    def test_hybrid_batch_limits_are_enforced_locally(self):
        profile = preferences({"localisation_transport": 5})

        for payload in (
            {"arguments": [{}] * 9, "relations": []},
            {"arguments": [], "relations": [{}] * 9},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(HotelHybridValidationError):
                    evaluate_hotel_session(
                        self.hotel,
                        profile,
                        argument_mode="hybrid",
                        hybrid_generator=StaticGenerator(lambda **_: payload),
                    )

    def test_json_and_provider_errors_are_never_silently_baselined(self):
        profile = preferences({"localisation_transport": 5})

        class MalformedGenerator:
            def propose_arguments(self, **_):
                return {"arguments": "not-a-list", "relations": []}

        with self.assertRaises(HotelHybridValidationError):
            evaluate_hotel_session(
                self.hotel,
                profile,
                argument_mode="hybrid",
                hybrid_generator=MalformedGenerator(),
            )

        bad_json_response = SimpleNamespace(
            text="not-json",
            usage_metadata=None,
            model_version="fake",
            response_id="fake-id",
        )
        generator = GeminiHybridArgumentGenerator(
            client=FakeClient(response=bad_json_response)
        )
        with self.assertRaises(HotelGeminiError):
            evaluate_hotel_session(
                self.hotel,
                profile,
                argument_mode="hybrid",
                hybrid_generator=generator,
            )

        provider_generator = GeminiHybridArgumentGenerator(
            client=FakeClient(error=RuntimeError("provider unavailable"))
        )
        with self.assertRaises(HotelGeminiError):
            evaluate_hotel_session(
                self.hotel,
                profile,
                argument_mode="hybrid",
                hybrid_generator=provider_generator,
            )

    def test_gemini_preference_interpreter_uses_closed_local_validation(self):
        text = "I need a quiet hotel and parking is mandatory."
        response_payload = {
            "aspect_preferences": [
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 4,
                    "source_text": "quiet hotel",
                }
            ],
            "constraints": [
                {
                    "constraint_id": "hard_01",
                    "target_type": "facility",
                    "target": "parking",
                    "operator": "present",
                    "qualifiers": {},
                    "value": None,
                    "hard": True,
                    "importance_raw": 5,
                    "source_text": "parking is mandatory",
                }
            ],
            "uninterpreted_items": [],
        }
        response = SimpleNamespace(
            text=json.dumps(response_payload),
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
                total_token_count=30,
                thoughts_token_count=2,
            ),
            model_version="gemini-2.5-flash-001",
            response_id="response-1",
        )
        client = FakeClient(response=response)
        interpreter = GeminiPreferenceInterpreter(client=client)
        result = interpreter.interpret(text)

        self.assertEqual(result.original_text, text)
        self.assertEqual(result.constraints[0].target, "parking")
        self.assertTrue(result.constraints[0].hard)
        self.assertEqual(
            client.models.calls[0]["model"],
            "gemini-2.5-flash",
        )
        config = client.models.calls[0]["config"]
        self.assertEqual(config["temperature"], 0.0)
        self.assertIn("response_json_schema", config)
        rendered = json.dumps(result.to_dict())
        self.assertNotIn("GEMINI_API_KEY", rendered)
        self.assertEqual(
            result.interpretation_trace["usage"]["thoughts_tokens"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
