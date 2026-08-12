import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.hotel.render_hotel_graph import (
    build_hotel_html,
    format_optional_score,
)
from src.hotel import (
    GeminiPreferenceInterpreter,
    build_gemini_preference_prompt,
    build_hybrid_argument_prompt,
    load_default_facility_ontology,
    load_hotel_profiles,
    prepare_hybrid_context,
    session_preferences_from_dict,
    validate_hybrid_proposals,
)


FIXTURE = Path(__file__).parent / "fixtures" / "hotel_profiles_minimal.json"
NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "hotel_hybrid_argumentation_colab_enterprise.ipynb"
)


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload

    def generate_content(self, **_):
        return SimpleNamespace(
            text=json.dumps(self.payload, ensure_ascii=False),
            usage_metadata=None,
            model_version="fake",
            response_id="fake-preferences",
        )


class _FakeClient:
    def __init__(self, payload):
        self.models = _FakeModels(payload)


def _constraint(
    constraint_id,
    *,
    target,
    source_text,
    hard=True,
    importance=5,
):
    return {
        "constraint_id": constraint_id,
        "target_type": "facility",
        "target": target,
        "operator": "present",
        "qualifiers": {},
        "value": None,
        "hard": hard,
        "importance_raw": importance,
        "source_text": source_text,
    }


def _interpret(text, *, constraints, aspects=None):
    payload = {
        "aspect_preferences": list(aspects or []),
        "constraints": list(constraints),
        "uninterpreted_items": [],
    }
    return GeminiPreferenceInterpreter(
        client=_FakeClient(payload)
    ).interpret(text)


class AbsoluteFiveStabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def test_negated_requirement_is_soft_optional_parking(self):
        text = "Un parking serait utile, mais il n’est pas obligatoire."
        preferences = _interpret(
            text,
            constraints=[
                _constraint(
                    "parking",
                    target="parking",
                    source_text=text,
                )
            ],
        )
        parking = preferences.constraints[0]
        self.assertFalse(parking.hard)
        self.assertEqual(parking.importance_raw, 2.0)
        self.assertEqual(parking.normalized_weight, 0.4)
        self.assertEqual(parking.weighting_method, "absolute_5")

    def test_all_listed_negated_requirements_remain_soft(self):
        negations = (
            "pas obligatoire",
            "non obligatoire",
            "n’est pas obligatoire",
            "pas indispensable",
            "non indispensable",
            "pas nécessaire",
            "non nécessaire",
            "not mandatory",
            "is not mandatory",
            "not required",
            "is not required",
            "not necessary",
            "is not necessary",
        )
        for negation in negations:
            with self.subTest(negation=negation):
                text = f"Parking {negation}."
                parking = _interpret(
                    text,
                    constraints=[
                        _constraint(
                            "parking",
                            target="parking",
                            source_text=text,
                        )
                    ],
                ).constraints[0]
                self.assertFalse(parking.hard)
                self.assertEqual(parking.importance_raw, 2.0)
                self.assertEqual(parking.normalized_weight, 0.4)

    def test_positive_requirements_stay_hard_at_five(self):
        cases = (
            ("parking obligatoire", "parking"),
            ("parking indispensable", "parking"),
            ("Wi-Fi indispensable", "wifi"),
            ("Wi-Fi nécessaire", "wifi"),
            ("must have parking", "parking"),
            ("parking is mandatory", "parking"),
            ("parking is required", "parking"),
        )
        for text, target in cases:
            with self.subTest(text=text):
                requirement = _interpret(
                    text,
                    constraints=[
                        _constraint(
                            "requirement",
                            target=target,
                            source_text=text,
                            hard=False,
                            importance=2,
                        )
                    ],
                ).constraints[0]
                self.assertTrue(requirement.hard)
                self.assertEqual(requirement.importance_raw, 5.0)
                self.assertEqual(requirement.normalized_weight, 0.0)

    def test_positive_and_negated_requirements_are_scoped_separately(self):
        text = (
            "Le Wi-Fi est indispensable, mais le parking n’est pas "
            "obligatoire."
        )
        preferences = _interpret(
            text,
            constraints=[
                _constraint("wifi", target="wifi", source_text=text),
                _constraint("parking", target="parking", source_text=text),
            ],
        )
        by_target = {item.target: item for item in preferences.constraints}
        self.assertTrue(by_target["wifi"].hard)
        self.assertEqual(by_target["wifi"].importance_raw, 5.0)
        self.assertEqual(by_target["wifi"].normalized_weight, 0.0)
        self.assertFalse(by_target["parking"].hard)
        self.assertEqual(by_target["parking"].importance_raw, 2.0)
        self.assertEqual(by_target["parking"].normalized_weight, 0.4)

    def test_multisentence_source_does_not_contaminate_aspect_importance(self):
        text = "Je veux un hôtel calme. Un parking serait utile."
        preferences = _interpret(
            text,
            aspects=[
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 5,
                    "source_text": text,
                }
            ],
            constraints=[
                _constraint(
                    "parking",
                    target="parking",
                    source_text=text,
                )
            ],
        )
        quiet = preferences.get_aspect("bruit_calme")
        parking = preferences.constraints[0]
        self.assertEqual(quiet.importance_raw, 3.0)
        self.assertEqual(quiet.normalized_weight, 0.6)
        self.assertEqual(parking.importance_raw, 2.0)
        self.assertEqual(parking.normalized_weight, 0.4)

    def test_first_prompt_requires_one_minimal_exact_excerpt_per_intent(self):
        prompt = build_gemini_preference_prompt(
            "Je veux un hôtel calme. Un parking serait utile.",
            load_default_facility_ontology(),
        )
        self.assertIn("smallest exact contiguous source_text excerpt", prompt)
        self.assertIn("Keep one intention per source_text", prompt)
        self.assertIn("explicitly negated necessity", prompt)

    def test_optional_score_formatting_handles_none_everywhere(self):
        self.assertEqual(format_optional_score(None), "non disponible")
        self.assertEqual(format_optional_score(0.123456), "0.1235")
        rendered = build_hotel_html({"linear_empirical_score": None})
        self.assertIn("non disponible", rendered)

        notebook_text = NOTEBOOK.read_text(encoding="utf-8")
        self.assertNotIn("linear_empirical_score']:.4f", notebook_text)
        self.assertNotIn('linear_empirical_score\"]:.4f', notebook_text)
        self.assertIn("format_optional_score", notebook_text)
        self.assertIn("HOTEL_PIPELINE_REF", notebook_text)
        self.assertNotIn("ad3562d", notebook_text)

    def test_second_prompt_removes_nested_scores_but_keeps_unit_ids(self):
        preferences = session_preferences_from_dict(
            {
                "aspect_preferences": {
                    "bruit_calme": {
                        "importance_raw": 3,
                        "source_text": "calme",
                    }
                },
                "constraints": [],
            }
        )
        prepared = prepare_hybrid_context(self.hotel, preferences)
        units = prepared.prompt_units()
        self.assertTrue(units)
        unit_id = units[0]["scoring_unit_id"]
        sentinel = 987654.321987
        units[0].update(
            {
                "importance": sentinel,
                "weight": sentinel,
                "wilson": sentinel,
                "force": sentinel,
                "score": sentinel,
                "nested": {"final_force": sentinel},
            }
        )
        hotel_context = {
            **prepared.hotel_context,
            "nested": {"intrinsic_strength": sentinel},
        }
        prompt = build_hybrid_argument_prompt(
            preferences=preferences,
            hotel_context=hotel_context,
            authorized_sources=prepared.prompt_sources(),
            scoring_units=units,
            constraint_outcomes=[
                {"status": "known", "nested": {"normalized_weight": sentinel}}
            ],
        )
        self.assertNotIn(str(sentinel), prompt)
        for field in (
            "importance",
            "weight",
            "wilson",
            "force",
            "score",
            "final_force",
            "intrinsic_strength",
            "normalized_weight",
        ):
            self.assertNotIn(f'"{field}":', prompt)
        self.assertIn(f'"scoring_unit_id": "{unit_id}"', prompt)
        self.assertIn("scoring_unit_refs", prompt)

    def test_authorized_scoring_unit_still_materializes_into_dfquad(self):
        preferences = session_preferences_from_dict(
            {
                "aspect_preferences": {
                    "bruit_calme": {
                        "importance_raw": 3,
                        "source_text": "calme",
                    }
                },
                "constraints": [],
            }
        )
        prepared = prepare_hybrid_context(self.hotel, preferences)
        unit = prepared.prompt_units()[0]
        raw_batch = {
            "arguments": [
                {
                    "id": "ARG_VALID",
                    "kind": unit["kind"],
                    "type": unit["type"],
                    "text": "A grounded deterministic argument.",
                    "preference_refs": ["gemini-non-authoritative"],
                    "source_refs": unit["source_refs"],
                    "scoring_unit_refs": [unit["scoring_unit_id"]],
                    "explanatory_only": False,
                }
            ],
            "relations": [],
        }
        validation = validate_hybrid_proposals(
            raw_batch,
            prepared=prepared,
            preferences=preferences,
            hotel=self.hotel,
        )
        self.assertEqual(len(validation.scoring_arguments), 1)
        self.assertEqual(
            validation.scoring_arguments[0].scoring_unit_id,
            unit["scoring_unit_id"],
        )
        self.assertEqual(
            validation.scoring_arguments[0].preference_refs,
            unit["preference_refs"],
        )
        self.assertTrue(validation.scoring_units[0]["included_in_dfquad"])


if __name__ == "__main__":
    unittest.main()
