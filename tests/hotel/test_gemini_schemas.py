import copy
import unittest

from src.hotel import (
    HOTEL_ASPECTS,
    FacilityOntology,
    HotelPreferenceValidationError,
    build_gemini_preference_prompt,
    build_hybrid_argument_response_schema,
    build_preference_prompt_contract,
    build_preference_response_schema,
    load_default_facility_ontology,
    session_preferences_from_dict,
)
from src.hotel.gemini_preference_interpreter import (
    MAX_ASPECT_PREFERENCES,
    MAX_PREFERENCE_CONSTRAINTS,
    MAX_UNINTERPRETED_ITEMS,
    _validate_interpretation_payload,
)


USER_TEXT = (
    "I need a quiet hotel. Parking is mandatory and it must be free. "
    "Leave this unclear."
)


def base_payload():
    return {
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
                "qualifiers": {"price": "free"},
                "value": None,
                "hard": True,
                "importance_raw": 5,
                "source_text": "Parking is mandatory",
            }
        ],
        "uninterpreted_items": [],
    }


def assert_absent_recursively(test_case, value, forbidden_keys):
    if isinstance(value, dict):
        test_case.assertTrue(forbidden_keys.isdisjoint(value))
        for child in value.values():
            assert_absent_recursively(test_case, child, forbidden_keys)
    elif isinstance(value, list):
        for child in value:
            assert_absent_recursively(test_case, child, forbidden_keys)


class GeminiSchemaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = load_default_facility_ontology()

    def validate(self, payload):
        return _validate_interpretation_payload(
            payload,
            original_text=USER_TEXT,
            ontology=self.ontology,
        )

    def test_preference_schema_keeps_structure_without_state_heavy_rules(self):
        schema = build_preference_response_schema(self.ontology)

        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            schema["required"],
            ["aspect_preferences", "constraints", "uninterpreted_items"],
        )
        properties = schema["properties"]
        self.assertEqual(
            set(properties),
            {"aspect_preferences", "constraints", "uninterpreted_items"},
        )
        constraint = properties["constraints"]["items"]
        self.assertEqual(
            constraint["properties"]["qualifiers"],
            {"type": "object"},
        )
        self.assertEqual(
            constraint["properties"]["target_type"]["enum"],
            ["facility", "metadata"],
        )
        self.assertNotIn(
            "enum",
            properties["aspect_preferences"]["items"]["properties"][
                "aspect"
            ],
        )
        self.assertNotIn("enum", constraint["properties"]["target"])
        assert_absent_recursively(
            self,
            schema,
            {"minItems", "maxItems", "minimum", "maximum", "minLength"},
        )

    def test_hybrid_schema_keeps_structure_without_array_bounds(self):
        schema = build_hybrid_argument_response_schema()

        self.assertEqual(schema["required"], ["arguments", "relations"])
        argument = schema["properties"]["arguments"]["items"]
        self.assertEqual(
            set(argument["required"]),
            {
                "id",
                "kind",
                "type",
                "text",
                "source_refs",
                "scoring_unit_refs",
                "explanatory_only",
            },
        )
        self.assertIn("preference_refs", argument["properties"])
        self.assertNotIn("preference_refs", argument["required"])
        self.assertIn("enum", argument["properties"]["kind"])
        self.assertIn("enum", argument["properties"]["type"])
        assert_absent_recursively(self, schema, {"minItems", "maxItems"})

    def test_prompt_lists_closed_values_compactly(self):
        prompt = build_gemini_preference_prompt(USER_TEXT, self.ontology)

        for aspect in HOTEL_ASPECTS:
            self.assertIn(aspect, prompt)
        for facility in self.ontology.capability_names:
            self.assertIn(f"- {facility}:", prompt)
        self.assertIn("Allowed facility targets:\nparking, wifi", prompt)
        self.assertIn("Allowed metadata targets:\ncity", prompt)
        self.assertIn("Allowed operators:\npresent, equals", prompt)
        self.assertIn(
            '- parking: price=["free", "paid"], '
            'location=["on_site", "nearby"]',
            prompt,
        )
        self.assertIn(
            '- wifi: price=["free", "paid"], '
            'coverage=["rooms", "common_areas", "all_areas"]',
            prompt,
        )
        self.assertIn("heated=[true, false]", prompt)

    def test_prompt_contract_is_generated_from_the_ontology(self):
        custom = FacilityOntology.from_dict(
            {
                "schema_version": "test",
                "ontology_version": "test",
                "operators": ["present"],
                "capabilities": {
                    "custom_facility": {
                        "qualifiers": {
                            "mode": ["alpha", "beta", "unknown"]
                        }
                    }
                },
                "facility_mappings": [],
            }
        )

        section = build_preference_prompt_contract(custom)

        self.assertIn("Allowed facility targets:\ncustom_facility", section)
        self.assertIn('- custom_facility: mode=["alpha", "beta"]', section)
        self.assertNotIn("unknown", section)

    def test_local_validation_rejects_unknown_aspect_target_and_operator(self):
        mutations = (
            ("aspect", lambda row: row["aspect_preferences"][0].update(
                aspect="unknown_aspect"
            )),
            ("target", lambda row: row["constraints"][0].update(
                target="unknown_facility"
            )),
            ("operator", lambda row: row["constraints"][0].update(
                operator="contains"
            )),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                payload = base_payload()
                mutate(payload)
                with self.assertRaises(HotelPreferenceValidationError):
                    self.validate(payload)

    def test_local_validation_rejects_importance_outside_range(self):
        for collection, value in (
            ("aspect_preferences", -0.01),
            ("aspect_preferences", 5.01),
            ("constraints", -0.01),
            ("constraints", 5.01),
        ):
            with self.subTest(collection=collection, value=value):
                payload = base_payload()
                payload[collection][0]["importance_raw"] = value
                with self.assertRaises(HotelPreferenceValidationError):
                    self.validate(payload)

    def test_local_validation_rejects_invalid_facility_qualifiers(self):
        invalid_qualifiers = (
            {"unknown_key": "free"},
            {"price": "gratis"},
            {"style": "buffet"},
        )
        for qualifiers in invalid_qualifiers:
            with self.subTest(qualifiers=qualifiers):
                payload = base_payload()
                payload["constraints"][0]["qualifiers"] = qualifiers
                with self.assertRaises(HotelPreferenceValidationError):
                    self.validate(payload)

    def test_local_validation_enforces_removed_size_and_text_rules(self):
        oversized = (
            (
                "aspect_preferences",
                [base_payload()["aspect_preferences"][0]]
                * (MAX_ASPECT_PREFERENCES + 1),
            ),
            (
                "constraints",
                [base_payload()["constraints"][0]]
                * (MAX_PREFERENCE_CONSTRAINTS + 1),
            ),
            (
                "uninterpreted_items",
                [
                    {
                        "text": "Leave this unclear",
                        "reason": "ambiguous_request",
                    }
                ]
                * (MAX_UNINTERPRETED_ITEMS + 1),
            ),
        )
        for field, value in oversized:
            with self.subTest(field=field):
                payload = base_payload()
                payload[field] = value
                with self.assertRaises(HotelPreferenceValidationError):
                    self.validate(payload)

        empty_text_mutations = (
            lambda row: row["aspect_preferences"][0].update(source_text=""),
            lambda row: row["constraints"][0].update(constraint_id=""),
            lambda row: row.update(
                uninterpreted_items=[
                    {"text": "", "reason": "ambiguous_request"}
                ]
            ),
        )
        for mutate in empty_text_mutations:
            payload = base_payload()
            mutate(payload)
            with self.assertRaises(HotelPreferenceValidationError):
                self.validate(payload)

    def test_parking_free_is_accepted_and_public_format_is_preserved(self):
        validated = self.validate(copy.deepcopy(base_payload()))
        preferences = session_preferences_from_dict(
            validated,
            path="interpreter_output",
            original_text=USER_TEXT,
        )
        public = preferences.to_dict()

        self.assertEqual(
            validated["constraints"][0]["qualifiers"],
            {"price": "free"},
        )
        self.assertEqual(
            set(public),
            {
                "original_text",
                "aspect_preferences",
                "constraints",
                "uninterpreted_items",
            },
        )
        self.assertEqual(public["constraints"][0]["target"], "parking")
        self.assertEqual(
            public["constraints"][0]["qualifiers"],
            {"price": "free"},
        )


if __name__ == "__main__":
    unittest.main()
