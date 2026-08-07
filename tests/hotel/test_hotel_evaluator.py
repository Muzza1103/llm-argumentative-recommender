import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.hotel import (
    evaluate_hotel_session,
    load_hotel_profiles,
    session_preferences_from_dict,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "hotel"
    / "fixtures"
    / "hotel_profiles_minimal.json"
)
EXAMPLE_PREFERENCES = REPOSITORY_ROOT / "configs" / "hotel_session_example.json"


def preferences(aspects=None, constraints=None):
    aspects = aspects or {}
    return session_preferences_from_dict(
        {
            "aspect_preferences": {
                aspect: {
                    "importance_raw": importance,
                    "source_text": f"Need {aspect}",
                }
                for aspect, importance in aspects.items()
            },
            "constraints": constraints or [],
        }
    )


class HotelEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def test_coverage_and_linear_baseline_use_observed_aspects_only(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences(
                {
                    "localisation_transport": 5,
                    "bruit_calme": 5,
                    "wifi_internet": 5,
                }
            ),
        )
        self.assertEqual(
            result.observed_preference_aspects,
            ("localisation_transport", "bruit_calme"),
        )
        self.assertEqual(result.missing_preference_aspects, ("wifi_internet",))
        self.assertAlmostEqual(result.preference_coverage, 2 / 3)
        self.assertAlmostEqual(result.linear_empirical_score, 0.5)

    def test_root_is_fixed_at_half_and_uses_existing_dfquad(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences(
                {
                    "localisation_transport": 5,
                    "bruit_calme": 5,
                }
            ),
        )
        root = result.graph["nodes"]["ROOT"]
        self.assertEqual(root["base_score"], 0.5)
        self.assertEqual(
            root["text"],
            "Recommend this hotel for the current session",
        )
        self.assertEqual(result.dfquad["root_base_score"], 0.5)
        self.assertAlmostEqual(result.dfquad_score, 0.5)
        self.assertEqual(
            result.dfquad["node_scores"]["ROOT"]["final_score"],
            result.dfquad_score,
        )

    def test_hard_constraint_violation_makes_hotel_ineligible(self):
        london_hotel = replace(
            self.hotel,
            metadata=replace(self.hotel.metadata, city="London"),
        )
        result = evaluate_hotel_session(
            london_hotel,
            preferences(
                constraints=[
                    {
                        "text": "The hotel must be in Paris",
                        "importance_raw": 5,
                        "mode": "hard",
                        "field": "city",
                        "value": "Paris",
                    }
                ]
            ),
        )
        self.assertEqual(result.eligibility.status, "ineligible")
        self.assertEqual(
            result.eligibility.hard_constraints[0].status.value,
            "violated",
        )
        self.assertEqual(result.arguments, ())
        self.assertEqual(list(result.graph["nodes"]), ["ROOT"])

    def test_satisfied_hard_constraint_does_not_saturate_dfquad(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences(
                constraints=[
                    {
                        "text": "Wi-Fi is required",
                        "importance_raw": 5,
                        "mode": "hard",
                        "field": "wifi",
                    }
                ]
            ),
        )
        self.assertEqual(result.eligibility.status, "eligible")
        self.assertEqual(
            result.eligibility.hard_constraints[0].status.value,
            "satisfied",
        )
        self.assertEqual(result.dfquad_score, 0.5)
        self.assertEqual(result.arguments, ())
        self.assertEqual(list(result.graph["nodes"]), ["ROOT"])
        self.assertEqual(result.graph["edges"], [])
        self.assertEqual(result.scoring_status, "no_soft_preferences")
        self.assertFalse(result.is_personalized)

    def test_unknown_hard_constraint_is_reported_but_not_ineligible(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences(
                constraints=[
                    {
                        "text": "Parking is required",
                        "importance_raw": 5,
                        "mode": "hard",
                        "field": "parking",
                    }
                ]
            ),
        )
        self.assertEqual(result.eligibility.status, "eligible")
        self.assertEqual(len(result.unknown_constraints), 1)
        self.assertEqual(result.arguments, ())

    def test_city_without_expected_value_is_unknown(self):
        london_hotel = replace(
            self.hotel,
            metadata=replace(self.hotel.metadata, city="London"),
        )
        missing_value_result = evaluate_hotel_session(
            london_hotel,
            preferences(
                constraints=[
                    {
                        "text": "Must be in Paris",
                        "importance_raw": 5,
                        "mode": "hard",
                        "field": "city",
                    }
                ]
            ),
        )
        self.assertEqual(
            missing_value_result.eligibility.hard_constraints[0].status.value,
            "unknown",
        )
        self.assertEqual(missing_value_result.eligibility.status, "eligible")

        expected_statuses = {
            "London": "satisfied",
            "Paris": "violated",
        }
        for expected_city, expected_status in expected_statuses.items():
            with self.subTest(expected_city=expected_city):
                result = evaluate_hotel_session(
                    london_hotel,
                    preferences(
                        constraints=[
                            {
                                "text": "A structured city constraint",
                                "importance_raw": 5,
                                "mode": "hard",
                                "field": "city",
                                "value": expected_city,
                            }
                        ]
                    ),
                )
                self.assertEqual(
                    result.eligibility.hard_constraints[0].status.value,
                    expected_status,
                )

    def test_hotel_without_usable_information_stays_at_half(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences({"proprete_hygiene": 5}),
        )
        self.assertEqual(result.preference_coverage, 0.0)
        self.assertIsNone(result.linear_empirical_score)
        self.assertEqual(result.dfquad_score, 0.5)
        self.assertEqual(result.arguments, ())
        self.assertEqual(list(result.graph["nodes"]), ["ROOT"])
        self.assertEqual(result.scoring_status, "no_usable_evidence")
        self.assertTrue(result.is_personalized)

    def test_full_result_is_json_serializable_and_traceable(self):
        result = evaluate_hotel_session(
            self.hotel,
            preferences({"localisation_transport": 5}),
        )
        payload = result.to_dict()
        rendered = json.dumps(payload)
        self.assertIn("review_sources", rendered)
        self.assertEqual(payload["hotel_name"], "Fixture Hotel")
        self.assertEqual(payload["weighting_method"], "absolute_5")
        self.assertEqual(payload["scoring_status"], "scored")
        self.assertTrue(payload["is_personalized"])
        required = {
            "id",
            "arg_type",
            "argument_family",
            "text",
            "aspect",
            "intrinsic_strength",
            "importance_raw",
            "normalized_weight",
            "evidence_score",
            "n_support",
            "n_attack",
            "n_neutral",
            "review_sources",
        }
        self.assertTrue(required.issubset(payload["arguments"][0]))

    def test_argumentation_core_imports_without_torch_or_transformers(self):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import src.hotel; import src.argumentation; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules"
                ),
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cli_runs_end_to_end_with_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "evaluation.json"
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.hotel.evaluate_hotel_session",
                    "--profiles",
                    str(FIXTURE),
                    "--hotel-id",
                    "hotel-1",
                    "--preferences",
                    str(EXAMPLE_PREFERENCES),
                    "--output",
                    str(output_path),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["hotel_id"], "hotel-1")
            self.assertEqual(payload["graph"]["nodes"]["ROOT"]["base_score"], 0.5)

    def test_cli_text_mode_fails_clearly_without_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.hotel.evaluate_hotel_session",
                    "--profiles",
                    str(FIXTURE),
                    "--hotel-id",
                    "hotel-1",
                    "--preference-text",
                    "A quiet hotel",
                    "--output",
                    str(Path(temporary_directory) / "evaluation.json"),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("requires an explicitly configured", process.stderr)


if __name__ == "__main__":
    unittest.main()
