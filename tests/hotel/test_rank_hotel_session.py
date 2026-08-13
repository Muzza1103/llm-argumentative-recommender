from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import scripts.hotel.rank_hotel_session as ranking_module
from scripts.hotel.rank_hotel_session import (
    CandidateEvaluation,
    CandidateFailure,
    CandidateSelection,
    build_ranking_rows,
    build_vertex_client,
    evaluate_candidates,
    load_or_interpret_preferences,
    select_candidate_hotel_ids,
    write_ranking_outputs,
)
from src.hotel import (
    FacilityOntology,
    load_hotel_profiles,
    session_preferences_from_dict,
)
from tests.hotel.fake_components import FakeHybridArgumentGenerator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPOSITORY_ROOT
    / "tests"
    / "hotel"
    / "fixtures"
    / "hotel_profiles_minimal.json"
)
EXAMPLE_PREFERENCES = (
    REPOSITORY_ROOT / "configs" / "hotel_session_example.json"
)


def _dataset_with_ids(*hotel_ids: str):
    dataset = load_hotel_profiles(FIXTURE)
    template = dataset.hotels[0]
    hotels = []
    for hotel_id in hotel_ids:
        reviews = tuple(
            replace(review, hotel_id=hotel_id)
            for review in template.reviews
        )
        hotels.append(
            replace(
                template,
                hotel_id=hotel_id,
                reviews=reviews,
                metadata=replace(
                    template.metadata,
                    liteapi_id=hotel_id,
                    name=f"Hôtel {hotel_id}",
                ),
            )
        )
    return replace(
        dataset,
        n_hotels=len(hotels),
        n_reviews=dataset.n_reviews * len(hotels),
        hotels=tuple(hotels),
    )


PREFERENCES = session_preferences_from_dict(
    {
        "weighting_method": "absolute_5",
        "aspect_preferences": {
            "bruit_calme": {
                "importance_raw": 3,
                "source_text": "Je cherche un hôtel calme.",
            }
        },
        "constraints": [],
    },
    original_text="Je cherche un hôtel calme.",
)


def _payload(
    hotel_id: str,
    score: float | None,
    *,
    eligibility_status: str = "eligible",
    linear_score: float | None = None,
) -> dict[str, object]:
    reasons = (
        [
            {
                "constraint_id": "city_1",
                "reason": "city mismatch",
                "status": "violated",
            }
        ]
        if eligibility_status == "ineligible"
        else []
    )
    return {
        "hotel_id": hotel_id,
        "hotel_name": f"Hôtel {hotel_id}",
        "argument_mode": "baseline",
        "weighting_method": "absolute_5",
        "eligibility": {
            "status": eligibility_status,
            "hard_constraints": [],
        },
        "ineligibility_reasons": reasons,
        "unknown_constraints": [],
        "dfquad_score": score,
        "linear_empirical_score": linear_score,
        "scoring_status": "scored",
        "is_personalized": True,
        "arguments": [
            {"id": "S1", "arg_type": "support", "text": "Support"},
            {"id": "A1", "arg_type": "attack", "text": "Attack"},
        ],
        "scoring_units": [
            {"scoring_unit_id": "U1", "included_in_dfquad": True},
            {"scoring_unit_id": "U2", "included_in_dfquad": False},
        ],
    }


class _Result:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


class _CountingInterpreter:
    def __init__(self):
        self.calls = 0
        self.texts: list[str] = []

    def interpret(self, text: str):
        self.calls += 1
        self.texts.append(text)
        return PREFERENCES


class RankHotelSessionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset_with_ids("h1", "h2", "h3", "h4", "h5")

    def test_selection_with_only_explicit_ids(self):
        selection = select_candidate_hotel_ids(
            self.dataset,
            ["h3", "h1"],
            candidate_count=2,
            seed=42,
        )
        self.assertEqual(selection.explicit_hotel_ids, ("h3", "h1"))
        self.assertEqual(selection.randomly_selected_hotel_ids, ())
        self.assertEqual(selection.final_hotel_ids, ("h3", "h1"))

    def test_random_completion_is_reproducible_and_unique(self):
        first = select_candidate_hotel_ids(
            self.dataset,
            ["h2"],
            candidate_count=4,
            seed=17,
        )
        second = select_candidate_hotel_ids(
            self.dataset,
            ["h2"],
            candidate_count=4,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.final_hotel_ids[0], "h2")
        self.assertEqual(len(set(first.final_hotel_ids)), 4)

    def test_random_selection_without_explicit_ids(self):
        selection = select_candidate_hotel_ids(
            self.dataset,
            candidate_count=3,
            seed=8,
        )
        self.assertEqual(selection.explicit_hotel_ids, ())
        self.assertEqual(
            selection.final_hotel_ids,
            selection.randomly_selected_hotel_ids,
        )
        self.assertEqual(len(set(selection.final_hotel_ids)), 3)

    def test_explicit_duplicates_are_removed_in_first_seen_order(self):
        selection = select_candidate_hotel_ids(
            self.dataset,
            ["h2", "h1", "h2", "h1"],
            candidate_count=2,
        )
        self.assertEqual(selection.explicit_hotel_ids, ("h2", "h1"))
        self.assertEqual(selection.final_hotel_ids, ("h2", "h1"))

    def test_unknown_explicit_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown hotel_id.*missing"):
            select_candidate_hotel_ids(
                self.dataset,
                ["h1", "missing"],
                candidate_count=2,
            )

    def test_candidate_count_smaller_than_explicit_set_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "candidate-count is smaller",
        ):
            select_candidate_hotel_ids(
                self.dataset,
                ["h1", "h2"],
                candidate_count=1,
            )

    def test_candidate_count_larger_than_dataset_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "exceeds the number of available hotels",
        ):
            select_candidate_hotel_ids(
                self.dataset,
                candidate_count=6,
            )

    def test_interpretation_runs_once_and_preferences_are_reused_by_identity(self):
        interpreter = _CountingInterpreter()
        text = "Je cherche un hôtel calme."
        preferences, source = load_or_interpret_preferences(
            preference_text=text,
            interpreter=interpreter,
        )
        generator = object()
        calls = []

        def evaluator(
            dataset,
            hotel_id,
            received_preferences,
            **kwargs,
        ):
            calls.append(
                (
                    dataset,
                    hotel_id,
                    received_preferences,
                    kwargs["hybrid_generator"],
                )
            )
            return _Result(_payload(hotel_id, 0.5))

        evaluations, failures = evaluate_candidates(
            self.dataset,
            ["h1", "h2", "h3"],
            preferences,
            argument_mode="hybrid",
            facility_ontology=object(),
            hybrid_generator=generator,
            evaluator=evaluator,
        )

        self.assertEqual(interpreter.calls, 1)
        self.assertEqual(interpreter.texts, [text])
        self.assertIs(preferences, PREFERENCES)
        self.assertEqual(source, {"type": "text", "original_text": text})
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(failures, [])
        self.assertTrue(all(call[0] is self.dataset for call in calls))
        self.assertTrue(all(call[2] is preferences for call in calls))
        self.assertTrue(all(call[3] is generator for call in calls))

    def test_one_vertex_client_is_shared_by_both_gemini_components(self):
        client = object()
        ontology = object()
        interpreter = _CountingInterpreter()
        generator = object()
        evaluation = CandidateEvaluation("h1", _payload("h1", 0.6))

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    ranking_module,
                    "load_hotel_profiles",
                    return_value=self.dataset,
                ),
                patch.object(
                    ranking_module.FacilityOntology,
                    "load",
                    return_value=ontology,
                ),
                patch.object(
                    ranking_module,
                    "build_vertex_client",
                    return_value=client,
                ) as build_client,
                patch.object(
                    ranking_module,
                    "GeminiPreferenceInterpreter",
                    return_value=interpreter,
                ) as interpreter_class,
                patch.object(
                    ranking_module,
                    "GeminiHybridArgumentGenerator",
                    return_value=generator,
                ) as generator_class,
                patch.object(
                    ranking_module,
                    "evaluate_candidates",
                    return_value=([evaluation], []),
                ) as evaluate,
                patch.object(
                    ranking_module,
                    "write_ranking_outputs",
                    return_value={"summary": {"n_evaluated": 1}},
                ),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = ranking_module.main(
                    [
                        "--profiles",
                        "unused.json",
                        "--preference-text",
                        "Je cherche un hôtel calme.",
                        "--hotel-ids",
                        "h1",
                        "--candidate-count",
                        "1",
                        "--argument-mode",
                        "hybrid",
                        "--gcp-project",
                        "test-project",
                        "--gcp-location",
                        "europe-west1",
                        "--output-dir",
                        temporary_directory,
                    ]
                )

        self.assertEqual(exit_code, 0)
        build_client.assert_called_once_with(
            "test-project",
            "europe-west1",
        )
        self.assertIs(
            interpreter_class.call_args.kwargs["client"],
            client,
        )
        self.assertIs(generator_class.call_args.kwargs["client"], client)
        self.assertEqual(interpreter.calls, 1)
        self.assertIs(evaluate.call_args.args[2], PREFERENCES)
        self.assertIs(
            evaluate.call_args.kwargs["hybrid_generator"],
            generator,
        )

    def test_vertex_client_uses_vertex_ai_and_adc_compatible_arguments(self):
        calls = []

        def factory(**kwargs):
            calls.append(kwargs)
            return object()

        client = build_vertex_client(
            "project-id",
            "us-central1",
            client_factory=factory,
        )
        self.assertIsNotNone(client)
        self.assertEqual(
            calls,
            [
                {
                    "vertexai": True,
                    "project": "project-id",
                    "location": "us-central1",
                }
            ],
        )

    def test_dfquad_sorting_ties_and_unranked_statuses(self):
        evaluations = [
            CandidateEvaluation("h-b", _payload("h-b", 0.8)),
            CandidateEvaluation("h-a", _payload("h-a", 0.8)),
            CandidateEvaluation("h-c", _payload("h-c", 0.7)),
            CandidateEvaluation(
                "h-ineligible",
                _payload(
                    "h-ineligible",
                    0.99,
                    eligibility_status="ineligible",
                ),
            ),
            CandidateEvaluation(
                "h-no-score",
                _payload("h-no-score", None, linear_score=None),
            ),
        ]
        ranked, not_ranked = build_ranking_rows(evaluations)

        self.assertEqual(
            [row["hotel_id"] for row in ranked],
            ["h-a", "h-b", "h-c"],
        )
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])
        statuses = {
            row["hotel_id"]: row["ranking_status"]
            for row in not_ranked
        }
        self.assertEqual(statuses["h-ineligible"], "ineligible")
        self.assertEqual(statuses["h-no-score"], "not_ranked")
        no_score = next(
            row for row in not_ranked if row["hotel_id"] == "h-no-score"
        )
        self.assertIsNone(no_score["rank"])
        self.assertIsNone(no_score["linear_empirical_score"])
        self.assertEqual(ranked[0]["n_supports"], 1)
        self.assertEqual(ranked[0]["n_attacks"], 1)
        self.assertEqual(ranked[0]["n_scoring_units"], 2)
        self.assertEqual(ranked[0]["n_counted_units"], 1)

    def test_one_hotel_failure_does_not_interrupt_following_evaluations(self):
        calls = []

        def evaluator(dataset, hotel_id, preferences, **kwargs):
            del dataset, preferences, kwargs
            calls.append(hotel_id)
            if hotel_id == "h2":
                raise RuntimeError("simulated hotel failure")
            return _Result(_payload(hotel_id, 0.5))

        evaluations, failures = evaluate_candidates(
            self.dataset,
            ["h1", "h2", "h3"],
            PREFERENCES,
            argument_mode="baseline",
            facility_ontology=object(),
            evaluator=evaluator,
        )
        self.assertEqual(calls, ["h1", "h2", "h3"])
        self.assertEqual(
            [evaluation.hotel_id for evaluation in evaluations],
            ["h1", "h3"],
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].hotel_id, "h2")
        self.assertEqual(failures[0].error_type, "RuntimeError")
        self.assertEqual(failures[0].message, "simulated hotel failure")

    def test_one_hybrid_generator_instance_serves_every_hotel(self):
        class CountingGenerator(FakeHybridArgumentGenerator):
            def __init__(self):
                self.calls = 0

            def propose_arguments(self, **kwargs):
                self.calls += 1
                return super().propose_arguments(**kwargs)

        generator = CountingGenerator()
        ontology = FacilityOntology.load(
            REPOSITORY_ROOT / "configs" / "hotel_facility_ontology.json"
        )
        evaluations, failures = evaluate_candidates(
            self.dataset,
            ["h1", "h2", "h3"],
            PREFERENCES,
            argument_mode="hybrid",
            facility_ontology=ontology,
            hybrid_generator=generator,
        )

        self.assertEqual(generator.calls, 3)
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(failures, [])

    def test_outputs_include_ranking_csv_preferences_jsons_and_html(self):
        evaluations = [
            CandidateEvaluation("h2", _payload("h2", 0.4)),
            CandidateEvaluation(
                "h1",
                _payload("h1", 0.7, linear_score=None),
            ),
        ]
        selection = CandidateSelection(
            requested_count=3,
            explicit_hotel_ids=("h2", "h1", "h3"),
            randomly_selected_hotel_ids=(),
            seed=42,
            final_hotel_ids=("h2", "h1", "h3"),
        )
        failures = [
            CandidateFailure(
                hotel_id="h3",
                hotel_name="Hôtel h3",
                error_type="RuntimeError",
                message="simulated failure",
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            payload = write_ranking_outputs(
                output_dir,
                preferences=PREFERENCES,
                preference_source={
                    "type": "text",
                    "original_text": PREFERENCES.original_text,
                },
                selection=selection,
                argument_mode="baseline",
                evaluations=evaluations,
                failures=failures,
            )

            expected_paths = [
                output_dir / "ranking.json",
                output_dir / "ranking.csv",
                output_dir / "session_preferences.json",
                output_dir / "evaluations" / "h1.json",
                output_dir / "evaluations" / "h2.json",
                output_dir / "reports" / "h1.html",
                output_dir / "reports" / "h2.html",
            ]
            self.assertTrue(all(path.is_file() for path in expected_paths))

            saved_ranking = json.loads(
                (output_dir / "ranking.json").read_text(encoding="utf-8")
            )
            saved_preferences = json.loads(
                (output_dir / "session_preferences.json").read_text(
                    encoding="utf-8"
                )
            )
            with (output_dir / "ranking.csv").open(
                encoding="utf-8",
                newline="",
            ) as stream:
                csv_rows = list(csv.DictReader(stream))

            self.assertEqual(payload, saved_ranking)
            self.assertEqual(saved_preferences, PREFERENCES.to_dict())
            self.assertEqual(saved_ranking["schema_version"], "1.0")
            self.assertEqual(
                saved_ranking["weighting_method"],
                "absolute_5",
            )
            self.assertEqual(saved_ranking["summary"]["n_ranked"], 2)
            self.assertEqual(saved_ranking["summary"]["n_failed"], 1)
            self.assertEqual(
                [row["hotel_id"] for row in saved_ranking["ranking"]],
                ["h1", "h2"],
            )
            self.assertEqual(
                [row["hotel_id"] for row in csv_rows],
                ["h1", "h2", "h3"],
            )
            self.assertEqual(csv_rows[0]["linear_empirical_score"], "")
            self.assertEqual(csv_rows[2]["ranking_status"], "failed")
            self.assertEqual(csv_rows[2]["error_type"], "RuntimeError")
            self.assertIn(
                "absolute_5",
                (output_dir / "reports" / "h1.html").read_text(
                    encoding="utf-8"
                ),
            )

    def test_direct_cli_runs_offline_with_structured_preferences(self):
        environment = os.environ.copy()
        environment.pop("GOOGLE_CLOUD_PROJECT", None)
        environment.pop("GOOGLE_CLOUD_LOCATION", None)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "ranking"
            process = subprocess.run(
                [
                    sys.executable,
                    "scripts/hotel/rank_hotel_session.py",
                    "--profiles",
                    str(FIXTURE),
                    "--preferences",
                    str(EXAMPLE_PREFERENCES),
                    "--hotel-ids",
                    "hotel-1",
                    "--candidate-count",
                    "1",
                    "--seed",
                    "42",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue((output_dir / "ranking.json").is_file())
            self.assertTrue(
                (output_dir / "evaluations" / "hotel-1.json").is_file()
            )
            self.assertTrue(
                (output_dir / "reports" / "hotel-1.html").is_file()
            )
            ranking = json.loads(
                (output_dir / "ranking.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ranking["summary"]["n_evaluated"], 1)
            self.assertEqual(
                ranking["candidate_selection"]["final_hotel_ids"],
                ["hotel-1"],
            )


if __name__ == "__main__":
    unittest.main()
