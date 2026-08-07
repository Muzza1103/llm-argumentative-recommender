import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from scripts.hotel.render_hotel_graph import (
    build_hotel_html,
    render_hotel_graph,
)
from src.hotel import (
    ConstraintOutcome,
    ConstraintStatus,
    Facility,
    GeminiPreferenceInterpreter,
    build_structured_fact_arguments,
    canonicalize_city_name,
    deduplicate_preference_intentions,
    evaluate_hotel_session,
    load_hotel_profiles,
    session_preferences_from_dict,
)


FIXTURE = Path(__file__).parent / "fixtures" / "hotel_profiles_minimal.json"
NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "hotel_hybrid_argumentation_colab_enterprise.ipynb"
)
USER_TEXT = (
    "Je cherche un hôtel à Londres, proche du centre, calme, avec un bon "
    "Wi-Fi. Un parking serait utile."
)


class FakeModels:
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


class FakeClient:
    def __init__(self, payload):
        self.models = FakeModels(payload)


class RegistryGenerator:
    """Use valid units/sources while deliberately copying bad preferences."""

    last_trace = {"model_requested": "fake", "request_count": 0}

    def propose_arguments(self, *, scoring_units, **_):
        return {
            "arguments": [
                {
                    "id": f"ARG_{index}",
                    "kind": unit["kind"],
                    "type": unit["type"],
                    "text": f"Grounded wording {index}.",
                    "preference_refs": [
                        "gemini-invented-ref",
                        *unit["preference_refs"],
                    ],
                    "source_refs": unit["source_refs"],
                    "scoring_unit_refs": [unit["scoring_unit_id"]],
                    "explanatory_only": False,
                }
                for index, unit in enumerate(scoring_units, start=1)
            ],
            "relations": [],
        }


def example_interpreter_payload():
    return {
        "aspect_preferences": [
            {
                "aspect": "localisation_transport",
                "importance_raw": 3,
                "source_text": "proche du centre",
            },
            {
                "aspect": "bruit_calme",
                "importance_raw": 3,
                "source_text": "calme",
            },
            {
                "aspect": "wifi_internet",
                "importance_raw": 3,
                "source_text": "bon Wi-Fi",
            },
            {
                "aspect": "parking_voiture",
                "importance_raw": 2,
                "source_text": "Un parking serait utile",
            },
        ],
        "constraints": [
            {
                "constraint_id": "c1",
                "target_type": "metadata",
                "target": "city",
                "operator": "equals",
                "qualifiers": {},
                "value": "Londres",
                "hard": True,
                "importance_raw": 5,
                "source_text": "Londres",
            },
            {
                "constraint_id": "c2",
                "target_type": "facility",
                "target": "wifi",
                "operator": "present",
                "qualifiers": {},
                "value": None,
                "hard": False,
                "importance_raw": 3,
                "source_text": "bon Wi-Fi",
            },
            {
                "constraint_id": "c3",
                "target_type": "facility",
                "target": "parking",
                "operator": "present",
                "qualifiers": {},
                "value": None,
                "hard": False,
                "importance_raw": 2,
                "source_text": "Un parking serait utile",
            },
        ],
        "uninterpreted_items": [],
    }


def raw_cross_type_payload(
    source_text,
    *,
    capability,
    aspect,
    qualifiers=None,
):
    return {
        "aspect_preferences": {
            aspect: {
                "importance_raw": 3,
                "source_text": source_text,
            }
        },
        "constraints": [
            {
                "constraint_id": "soft_01",
                "target": capability,
                "operator": "present",
                "qualifiers": qualifiers or {},
                "hard": False,
                "source_text": source_text,
            }
        ],
        "uninterpreted_items": [],
    }


class HotelIntentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def preferences_for_city(self, requested_city):
        return session_preferences_from_dict(
            {
                "aspect_preferences": {},
                "constraints": [
                    {
                        "constraint_id": "city",
                        "text": f"Ville {requested_city}",
                        "source_text": requested_city,
                        "importance_raw": 5,
                        "mode": "hard",
                        "target_type": "metadata",
                        "target": "city",
                        "operator": "equals",
                        "value": requested_city,
                    }
                ],
            }
        )

    def test_city_aliases_and_real_differences_are_conservative(self):
        cases = (
            ("Londres", "London", "satisfied"),
            ("London", "London", "satisfied"),
            ("Paris", "Paris", "satisfied"),
            ("Londres", "Paris", "violated"),
        )
        for requested, metadata, expected in cases:
            with self.subTest(requested=requested, metadata=metadata):
                hotel = replace(
                    self.base_hotel,
                    metadata=replace(self.base_hotel.metadata, city=metadata),
                )
                result = evaluate_hotel_session(
                    hotel,
                    self.preferences_for_city(requested),
                )
                outcome = result.constraint_outcomes[0].to_dict()
                self.assertEqual(outcome["status"], expected)
                self.assertEqual(
                    outcome["comparison"]["requested_value"],
                    requested,
                )
                self.assertEqual(
                    outcome["comparison"]["metadata_value"],
                    metadata,
                )
        self.assertEqual(canonicalize_city_name("  LÓNDRES  "), "london")
        self.assertEqual(canonicalize_city_name("Londrez"), "londrez")
        self.assertNotEqual(
            canonicalize_city_name("Londrez"),
            canonicalize_city_name("London"),
        )

    def test_qualitative_and_factual_requests_are_deduplicated(self):
        cases = (
            ("bon Wi-Fi", "wifi", "wifi_internet", "aspect"),
            ("Wi-Fi fiable", "wifi", "wifi_internet", "aspect"),
            ("connexion rapide", "wifi", "wifi_internet", "aspect"),
            ("Wi-Fi disponible", "wifi", "wifi_internet", "constraint"),
            ("avec Wi-Fi", "wifi", "wifi_internet", "constraint"),
            ("Wi-Fi gratuit", "wifi", "wifi_internet", "constraint"),
            (
                "un parking serait utile",
                "parking",
                "parking_voiture",
                "constraint",
            ),
            (
                "je voudrais un parking",
                "parking",
                "parking_voiture",
                "constraint",
            ),
            (
                "parking pratique",
                "parking",
                "parking_voiture",
                "aspect",
            ),
            (
                "parking difficile d'accès",
                "parking",
                "parking_voiture",
                "aspect",
            ),
        )
        for source, capability, aspect, expected in cases:
            with self.subTest(source=source):
                qualifiers = (
                    {"price": "free"}
                    if source == "Wi-Fi gratuit"
                    else {}
                )
                cleaned, _ = deduplicate_preference_intentions(
                    raw_cross_type_payload(
                        source,
                        capability=capability,
                        aspect=aspect,
                        qualifiers=qualifiers,
                    )
                )
                self.assertEqual(
                    aspect in cleaned["aspect_preferences"],
                    expected == "aspect",
                )
                self.assertEqual(
                    bool(cleaned["constraints"]),
                    expected == "constraint",
                )

        both, trace = deduplicate_preference_intentions(
            raw_cross_type_payload(
                "Wi-Fi gratuit et rapide",
                capability="wifi",
                aspect="wifi_internet",
                qualifiers={"price": "free"},
            )
        )
        self.assertIn("wifi_internet", both["aspect_preferences"])
        self.assertEqual(len(both["constraints"]), 1)
        self.assertEqual(trace[0]["action"], "keep_both")

    def build_example(self, *, with_parking=True):
        interpreter = GeminiPreferenceInterpreter(
            client=FakeClient(example_interpreter_payload())
        )
        preferences = interpreter.interpret(USER_TEXT)
        metadata = replace(self.base_hotel.metadata, city="London")
        if with_parking:
            metadata = replace(
                metadata,
                source_facility_ids=metadata.source_facility_ids + (46,),
                facilities=metadata.facilities
                + (Facility(name="Free Parking", facility_ids=(46,)),),
            )
        hotel = replace(self.base_hotel, metadata=metadata)
        result = evaluate_hotel_session(
            hotel,
            preferences,
            argument_mode="hybrid",
            hybrid_generator=RegistryGenerator(),
        )
        return preferences, result

    def test_example_uses_one_global_budget_and_authoritative_units(self):
        preferences, result = self.build_example()
        self.assertEqual(result.eligibility.status, "eligible")
        self.assertEqual(result.constraint_outcomes[0].status.value, "satisfied")
        self.assertEqual(
            {item.aspect for item in preferences.aspect_preferences},
            {"localisation_transport", "bruit_calme", "wifi_internet"},
        )
        self.assertEqual(
            [item.preference_ref for item in preferences.constraints],
            ["c1", "c3"],
        )
        expected_weights = {
            "localisation_transport": 3 / 11,
            "bruit_calme": 3 / 11,
            "wifi_internet": 3 / 11,
        }
        for aspect, expected in expected_weights.items():
            self.assertAlmostEqual(
                preferences.get_aspect(aspect).normalized_weight,
                expected,
            )
        hard, parking = preferences.constraints
        self.assertEqual(hard.normalized_weight, 0.0)
        self.assertAlmostEqual(parking.normalized_weight, 2 / 11)
        self.assertAlmostEqual(
            sum(
                item.normalized_weight
                for item in preferences.aspect_preferences
            )
            + sum(
                item.normalized_weight
                for item in preferences.constraints
                if not item.hard
            ),
            1.0,
        )

        parking_argument = next(
            argument
            for argument in result.arguments
            if argument.argument_family == "structured_fact"
        )
        self.assertEqual(parking_argument.arg_type, "support")
        self.assertAlmostEqual(parking_argument.intrinsic_strength, 2 / 11)
        self.assertEqual(parking_argument.preference_refs, ["c3"])
        self.assertTrue(
            all(
                "gemini-invented-ref" not in argument.preference_refs
                for argument in result.arguments
            )
        )
        rejected_reasons = {
            reason
            for row in result.hybrid["validation"]["rejected_arguments"]
            for reason in row["reasons"]
        }
        self.assertNotIn("preference_scoring_unit_mismatch", rejected_reasons)
        self.assertEqual(rejected_reasons, set())

        for argument in result.arguments:
            if argument.argument_family == "empirical_aspect":
                self.assertAlmostEqual(
                    argument.intrinsic_strength,
                    argument.normalized_weight * argument.evidence_score,
                )
        public = result.to_dict()
        self.assertTrue(public["arguments"])
        for argument in public["arguments"]:
            self.assertIn("arg_type", argument)
            self.assertNotIn("type", argument)

    def test_unknown_fact_has_weight_but_no_dfquad_contribution(self):
        preferences, unknown = self.build_example(with_parking=False)
        _, known = self.build_example(with_parking=True)
        parking = next(
            item for item in preferences.constraints
            if item.preference_ref == "c3"
        )
        self.assertAlmostEqual(parking.normalized_weight, 2 / 11)
        outcome = next(
            item for item in unknown.constraint_outcomes
            if item.constraint.preference_ref == "c3"
        )
        self.assertEqual(outcome.status.value, "unknown")
        self.assertFalse(
            any(
                argument.argument_family == "structured_fact"
                for argument in unknown.arguments
            )
        )
        audit = next(
            row for row in unknown.scoring_units
            if row.get("intent_ref") == "c3"
        )
        self.assertFalse(audit["included_in_dfquad"])
        self.assertEqual(audit["availability_status"], "unknown")
        self.assertIsNone(audit["final_force"])
        self.assertGreater(known.dfquad_score, unknown.dfquad_score)

    def test_violated_soft_fact_uses_the_same_global_weight_as_an_attack(self):
        preferences = session_preferences_from_dict(
            {
                "aspect_preferences": {
                    "bruit_calme": {
                        "importance_raw": 3,
                        "source_text": "calme",
                    }
                },
                "constraints": [
                    {
                        "constraint_id": "parking",
                        "text": "parking utile",
                        "source_text": "parking utile",
                        "importance_raw": 2,
                        "mode": "soft",
                        "target_type": "facility",
                        "target": "parking",
                        "operator": "present",
                    }
                ],
            }
        )
        parking = preferences.constraints[0]
        outcome = ConstraintOutcome(
            constraint=parking,
            status=ConstraintStatus.VIOLATED,
            reason="explicitly absent in the synthetic outcome",
            evidence=("No parking",),
        )
        argument = build_structured_fact_arguments(
            self.base_hotel,
            (outcome,),
        )[0]
        self.assertEqual(argument.arg_type, "attack")
        self.assertAlmostEqual(parking.normalized_weight, 2 / 5)
        self.assertAlmostEqual(argument.intrinsic_strength, 2 / 5)
        self.assertEqual(
            argument.metadata["force_formula"],
            "normalized_weight",
        )

    def test_renderer_is_autonomous_complete_and_escapes_text(self):
        _, result = self.build_example()
        payload = copy.deepcopy(result.to_dict())
        payload["hotel_name"] = '<img src=x onerror="alert(1)">'
        payload["arguments"][0]["text"] = "<script>alert(1)</script>"
        rendered = build_hotel_html(payload)
        self.assertIn("Contraintes factuelles souples", rendered)
        self.assertIn("Registre auditable des unités de score", rendered)
        self.assertIn("Support agrégé", rendered)
        self.assertIn("--green:#217a3c", rendered)
        self.assertIn("--red:#b42318", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn('<img src=x onerror="alert(1)">', rendered)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hotel.html"
            returned = render_hotel_graph(payload, output)
            self.assertEqual(output.read_text(encoding="utf-8"), returned)
            self.assertTrue(returned.startswith("<!doctype html>"))

    def test_notebook_reads_public_arguments_with_arg_type_and_compiles(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn('argument["type"]', code)
        self.assertNotIn('a["type"]', code)
        self.assertNotIn('a.get("type")', code)
        self.assertIn('argument["arg_type"]', code)
        self.assertIn("render_hotel_graph(payload, HOTEL_HTML_PATH)", code)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            source = "\n".join(
                line for line in source.splitlines()
                if not line.lstrip().startswith("%")
            )
            compile(source, f"notebook-cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
