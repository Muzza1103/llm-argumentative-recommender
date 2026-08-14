import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from src.hotel import (
    Facility,
    GeminiPreferenceInterpreter,
    evaluate_hotel_session,
    load_default_facility_ontology,
    load_hotel_profiles,
    validate_preference_structure,
)


FIXTURE = Path(__file__).parent / "fixtures" / "hotel_profiles_minimal.json"
PMR_TEXT = (
    "L'hôtel doit impérativement être situé à Londres et disposer "
    "d'installations accessibles aux personnes à mobilité réduite, notamment "
    "en fauteuil roulant. L'accès PMR est indispensable."
)


class _FakeModels:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
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
    target="parking",
    source_text="parking",
    hard=False,
    importance=2,
    qualifiers=None,
):
    return {
        "constraint_id": constraint_id,
        "target_type": "facility",
        "target": target,
        "operator": "present",
        "qualifiers": dict(qualifiers or {}),
        "value": None,
        "hard": hard,
        "importance_raw": importance,
        "source_text": source_text,
    }


def _pmr_payload():
    return {
        "aspect_preferences": [],
        "constraints": [
            {
                "constraint_id": "c1",
                "target_type": "metadata",
                "target": "city",
                "operator": "equals",
                "qualifiers": {},
                "value": "London",
                "hard": True,
                "importance_raw": 5,
                "source_text": (
                    "L'hôtel doit impérativement être situé à Londres"
                ),
            },
            {
                "constraint_id": "c2",
                "target_type": "facility",
                "target": "accessible_facilities",
                "operator": "present",
                "qualifiers": {},
                "value": None,
                "hard": True,
                "importance_raw": 5,
                "source_text": (
                    "disposer d'installations accessibles aux personnes "
                    "à mobilité réduite"
                ),
            },
            {
                "constraint_id": "c3",
                "target_type": "facility",
                "target": "accessible_facilities",
                "operator": "present",
                "qualifiers": {},
                "value": None,
                "hard": True,
                "importance_raw": 5,
                "source_text": "L'accès PMR est indispensable",
            },
        ],
        "uninterpreted_items": [],
    }


class GeminiPreferenceSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = load_default_facility_ontology()
        cls.base_hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def test_gemini_hard_and_importance_decisions_are_preserved(self):
        payload = {
            "aspect_preferences": [],
            "constraints": [
                _constraint(
                    "accessible",
                    target="accessible_facilities",
                    source_text="L'accès PMR est indispensable",
                    hard=True,
                    importance=5,
                ),
                _constraint(
                    "parking",
                    source_text="Un parking serait un plus",
                    hard=False,
                    importance=2,
                ),
            ],
            "uninterpreted_items": [],
        }
        text = (
            "L'accès PMR est indispensable. "
            "Un parking serait un plus, mais il n'est pas obligatoire."
        )
        client = _FakeClient(payload)
        preferences = GeminiPreferenceInterpreter(
            client=client,
            ontology=self.ontology,
        ).interpret(text)

        self.assertEqual(len(client.models.calls), 1)
        accessible, parking = preferences.constraints
        self.assertTrue(accessible.hard)
        self.assertEqual(accessible.importance_raw, 5.0)
        self.assertEqual(accessible.normalized_weight, 0.0)
        self.assertFalse(parking.hard)
        self.assertEqual(parking.importance_raw, 2.0)
        self.assertEqual(parking.normalized_weight, 0.4)
        self.assertEqual(
            preferences.interpretation_trace["request_count"],
            1,
        )
        self.assertIn("raw_response", preferences.interpretation_trace)
        self.assertIn("usage", preferences.interpretation_trace)

    def test_wording_never_overrides_gemini_semantics(self):
        payload = {
            "aspect_preferences": [
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 5,
                    "source_text": "calme si possible",
                }
            ],
            "constraints": [
                _constraint(
                    "parking",
                    source_text="parking pas obligatoire",
                    hard=True,
                    importance=5,
                ),
                _constraint(
                    "wifi",
                    target="wifi",
                    source_text="Wi-Fi indispensable",
                    hard=False,
                    importance=2,
                ),
            ],
            "uninterpreted_items": [],
        }
        text = (
            "Je préfère un hôtel calme si possible; parking pas obligatoire; "
            "Wi-Fi indispensable."
        )
        preferences = GeminiPreferenceInterpreter(
            client=_FakeClient(payload),
            ontology=self.ontology,
        ).interpret(text)

        quiet = preferences.get_aspect("bruit_calme")
        parking, wifi = preferences.constraints
        self.assertEqual(quiet.importance_raw, 5.0)
        self.assertTrue(parking.hard)
        self.assertEqual(parking.importance_raw, 5.0)
        self.assertFalse(wifi.hard)
        self.assertEqual(wifi.importance_raw, 2.0)
        self.assertEqual(
            preferences.interpretation_trace["structural_validation"],
            [],
        )

    def test_invalid_individual_entries_are_dropped_with_trace(self):
        payload = {
            "aspect_preferences": [
                {
                    "aspect": "unknown_aspect",
                    "importance_raw": 3,
                    "source_text": "calme",
                },
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 6,
                    "source_text": "calme",
                },
                {
                    "aspect": "wifi_internet",
                    "importance_raw": 3,
                    "source_text": "phrase absente",
                },
                {
                    "aspect": "localisation_transport",
                    "source_text": "centre",
                },
            ],
            "constraints": [
                _constraint(
                    "unknown",
                    target="raw provider facility",
                    source_text="parking",
                ),
                _constraint(
                    "bad_hard",
                    source_text="parking",
                    hard="true",
                ),
                _constraint(
                    "bad_importance",
                    source_text="parking",
                    importance=-1,
                ),
                _constraint(
                    "bad_source",
                    source_text="phrase absente",
                ),
                _constraint(
                    "bad_qualifier",
                    source_text="parking",
                    qualifiers={"price": "gratis"},
                ),
                {
                    "constraint_id": "missing",
                    "target_type": "facility",
                    "target": "parking",
                },
            ],
            "uninterpreted_items": [],
        }
        validated, trace = validate_preference_structure(
            payload,
            original_text="Je cherche un hôtel calme au centre avec parking.",
            ontology=self.ontology,
        )

        self.assertEqual(validated["aspect_preferences"], {})
        self.assertEqual(validated["constraints"], [])
        reason_sets = [set(row["reasons"]) for row in trace]
        for expected in (
            "unknown_aspect",
            "invalid_importance_raw",
            "source_text_not_in_original",
            "missing_required_field",
            "unknown_canonical_target",
            "invalid_hard",
            "invalid_qualifier",
        ):
            self.assertTrue(
                any(expected in reasons for reasons in reason_sets),
                expected,
            )
        self.assertTrue(
            all(row["action"] == "drop_invalid_entry" for row in trace)
        )

    def test_first_structurally_valid_duplicate_is_kept_unchanged(self):
        payload = {
            "aspect_preferences": [
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 7,
                    "source_text": "calme",
                },
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 2,
                    "source_text": "calme",
                },
                {
                    "aspect": "bruit_calme",
                    "importance_raw": 5,
                    "source_text": "très calme",
                },
            ],
            "constraints": [
                _constraint(
                    "c1",
                    source_text="parking",
                    hard=False,
                    importance=2,
                ),
                _constraint(
                    "c2",
                    source_text="parking pas obligatoire",
                    hard=True,
                    importance=5,
                ),
            ],
            "uninterpreted_items": [],
        }
        validated, trace = validate_preference_structure(
            payload,
            original_text="Je veux un hôtel très calme avec parking pas obligatoire.",
            ontology=self.ontology,
        )

        quiet = validated["aspect_preferences"]["bruit_calme"]
        self.assertEqual(quiet["importance_raw"], 2)
        self.assertEqual(quiet["source_text"], "calme")
        self.assertEqual(len(validated["constraints"]), 1)
        self.assertFalse(validated["constraints"][0]["hard"])
        self.assertEqual(validated["constraints"][0]["importance_raw"], 2)
        reasons = [reason for row in trace for reason in row["reasons"]]
        self.assertIn("duplicate_aspect", reasons)
        self.assertIn("duplicate_constraint", reasons)

    def test_complete_pmr_regression_has_one_hard_fact_unit(self):
        client = _FakeClient(_pmr_payload())
        preferences = GeminiPreferenceInterpreter(
            client=client,
            ontology=self.ontology,
        ).interpret(PMR_TEXT)

        self.assertEqual(len(client.models.calls), 1)
        self.assertEqual(
            [constraint.preference_ref for constraint in preferences.constraints],
            ["c1", "c2"],
        )
        accessible = preferences.constraints[1]
        self.assertTrue(accessible.hard)
        self.assertEqual(accessible.importance_raw, 5.0)
        self.assertEqual(accessible.normalized_weight, 0.0)

        trace = preferences.interpretation_trace["structural_validation"]
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["constraint_id"], "c3")
        self.assertEqual(trace[0]["reasons"], ["duplicate_constraint"])
        forbidden_actions = {
            "correct_constraint_mode",
            "calibrate_importance",
            "create_aspect",
            "create_constraint",
            "drop_aspect",
        }
        self.assertTrue(
            forbidden_actions.isdisjoint(
                row["action"] for row in trace
            )
        )

        metadata = replace(
            self.base_hotel.metadata,
            city="London",
            source_facility_ids=(
                *self.base_hotel.metadata.source_facility_ids,
                185,
            ),
            facilities=(
                *self.base_hotel.metadata.facilities,
                Facility(
                    name="Wheelchair accessible",
                    facility_ids=(185,),
                ),
            ),
        )
        hotel = replace(self.base_hotel, metadata=metadata)
        result = evaluate_hotel_session(
            hotel,
            preferences,
            facility_ontology=self.ontology,
        )
        accessible_units = [
            row
            for row in result.scoring_units
            if row.get("intent_ref") == "c2"
        ]
        self.assertEqual(len(accessible_units), 1)
        self.assertFalse(accessible_units[0]["included_in_dfquad"])
        self.assertEqual(accessible_units[0]["normalized_weight"], 0.0)
        self.assertEqual(result.eligibility.status, "eligible")
        self.assertEqual(result.dfquad_score, 0.5)

    def test_unknown_hard_fact_warns_without_becoming_a_violation(self):
        preferences = GeminiPreferenceInterpreter(
            client=_FakeClient(_pmr_payload()),
            ontology=self.ontology,
        ).interpret(PMR_TEXT)
        hotel = replace(
            self.base_hotel,
            metadata=replace(self.base_hotel.metadata, city="London"),
        )
        result = evaluate_hotel_session(
            hotel,
            preferences,
            facility_ontology=self.ontology,
        )
        outcome = next(
            row
            for row in result.constraint_outcomes
            if row.constraint.preference_ref == "c2"
        )
        self.assertEqual(outcome.status.value, "unknown")
        self.assertEqual(result.eligibility.status, "eligible")
        self.assertEqual(result.ineligibility_reasons, ())
        self.assertEqual(len(result.unknown_constraints), 1)


if __name__ == "__main__":
    unittest.main()
