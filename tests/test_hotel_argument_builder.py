import unittest
from dataclasses import replace
from pathlib import Path

from src.hotel import (
    AnnotationStatus,
    Facility,
    ReviewSignal,
    Stance,
    build_empirical_arguments,
    build_structured_fact_arguments,
    evaluate_constraints,
    load_hotel_profiles,
    select_review_sources,
    session_preferences_from_dict,
    wilson_lower_bound,
)


FIXTURE = Path(__file__).parent / "fixtures" / "hotel_profiles_minimal.json"


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


class HotelArgumentBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hotel = load_hotel_profiles(FIXTURE).hotels[0]

    def _contradictory_hotel(self):
        base_review = self.hotel.reviews[0]
        aspect = replace(
            self.hotel.get_aspect("localisation_transport"),
            score=0.6,
            confidence=0.4,
            wilson_lower=None,
            wilson_upper=None,
            wilson_width=None,
            n_mentions=4,
            support_count=3,
            attack_count=1,
            neutral_count=0,
            n_raw_signals_total=4,
        )
        review_specs = (
            ("review-b", "2026-02-01T00:00:00", Stance.SUPPORT, "new B"),
            ("review-a", "2026-02-01T00:00:00", Stance.SUPPORT, "new A"),
            ("review-old", "2026-01-01T00:00:00", Stance.SUPPORT, "old"),
            ("review-attack", "2026-01-15T00:00:00", Stance.ATTACK, "far"),
        )
        reviews = tuple(
            replace(
                base_review,
                dataset_index=index,
                review_id=review_id,
                review_date=review_date,
                annotation_status=AnnotationStatus.ANNOTATED,
                signals=(
                    ReviewSignal(
                        aspect="localisation_transport",
                        stance=stance,
                        evidence=evidence,
                    ),
                ),
            )
            for index, (review_id, review_date, stance, evidence)
            in enumerate(review_specs)
        )
        return replace(self.hotel, aspects=(aspect,), reviews=reviews)

    def test_known_wilson_lower_bound(self):
        self.assertAlmostEqual(
            wilson_lower_bound(1, 1),
            0.20654329147389294,
        )
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_generates_empirical_support(self):
        arguments = build_empirical_arguments(
            self.hotel,
            preferences({"localisation_transport": 5}),
        )
        self.assertEqual(len(arguments), 1)
        self.assertEqual(arguments[0].arg_type, "support")
        self.assertAlmostEqual(
            arguments[0].intrinsic_strength,
            wilson_lower_bound(1, 1),
        )

    def test_generates_empirical_attack_without_using_one_minus_support(self):
        arguments = build_empirical_arguments(
            self.hotel,
            preferences({"bruit_calme": 5}),
        )
        self.assertEqual(len(arguments), 1)
        self.assertEqual(arguments[0].arg_type, "attack")
        self.assertAlmostEqual(
            arguments[0].evidence_score,
            wilson_lower_bound(1, 1),
        )

    def test_keeps_support_and_attack_for_contradictory_aspect(self):
        arguments = build_empirical_arguments(
            self._contradictory_hotel(),
            preferences({"localisation_transport": 5}),
        )
        self.assertEqual(
            [argument.arg_type for argument in arguments],
            ["support", "attack"],
        )
        self.assertAlmostEqual(
            arguments[0].evidence_score,
            wilson_lower_bound(3, 4),
        )
        self.assertAlmostEqual(
            arguments[1].evidence_score,
            wilson_lower_bound(1, 4),
        )

    def test_does_not_build_unrequested_or_missing_aspects(self):
        noise_arguments = build_empirical_arguments(
            self.hotel,
            preferences({"bruit_calme": 5}),
        )
        self.assertTrue(
            all(argument.aspect == "bruit_calme" for argument in noise_arguments)
        )
        self.assertEqual(
            build_empirical_arguments(
                self.hotel,
                preferences({"proprete_hygiene": 5}),
            ),
            [],
        )

    def test_lack_of_negative_data_never_becomes_an_attack(self):
        arguments = build_empirical_arguments(
            self.hotel,
            preferences({"localisation_transport": 5}),
        )
        self.assertEqual([argument.arg_type for argument in arguments], ["support"])

    def test_sources_match_hotel_aspect_and_stance(self):
        argument = build_empirical_arguments(
            self.hotel,
            preferences({"bruit_calme": 5}),
        )[0]
        for source in argument.review_sources:
            self.assertEqual(source["review_id"], "hotel-1:review-1")
            self.assertEqual(source["aspect"], "bruit_calme")
            self.assertEqual(source["stance"], "attack")

    def test_sources_are_limited_to_two_and_ordered_deterministically(self):
        hotel = self._contradictory_hotel()
        sources = select_review_sources(
            hotel,
            aspect="localisation_transport",
            stance=Stance.SUPPORT,
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(
            [source["review_id"] for source in sources],
            ["review-a", "review-b"],
        )

    def test_duplicate_evidence_is_removed(self):
        hotel = self._contradictory_hotel()
        duplicate_review = replace(
            hotel.reviews[0],
            signals=(
                ReviewSignal(
                    aspect="localisation_transport",
                    stance=Stance.SUPPORT,
                    evidence="new A",
                ),
            ),
        )
        hotel = replace(
            hotel,
            reviews=(duplicate_review,) + hotel.reviews[1:],
        )
        sources = select_review_sources(
            hotel,
            aspect="localisation_transport",
            stance=Stance.SUPPORT,
        )
        normalized = [source["evidence"].casefold() for source in sources]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_explicit_facility_builds_factual_support(self):
        prefs = preferences(
            constraints=[
                {
                    "text": "Wi-Fi is required",
                    "importance_raw": 5,
                    "mode": "soft",
                    "field": "wifi",
                }
            ]
        )
        outcomes = evaluate_constraints(self.hotel, prefs.constraints)
        arguments = build_structured_fact_arguments(self.hotel, outcomes)
        self.assertEqual(outcomes[0].status.value, "satisfied")
        self.assertEqual(arguments[0].arg_type, "support")
        self.assertEqual(arguments[0].review_sources, [])

    def test_facility_matching_uses_word_boundaries(self):
        prefs = preferences(
            constraints=[
                {
                    "text": "A spa would be useful",
                    "importance_raw": 3,
                    "mode": "soft",
                    "field": "spa",
                }
            ]
        )
        spacious_hotel = replace(
            self.hotel,
            metadata=replace(
                self.hotel.metadata,
                facilities=(
                    Facility(
                        name="Spacious family rooms",
                        facility_ids=(901,),
                    ),
                ),
            ),
        )
        spacious_outcomes = evaluate_constraints(
            spacious_hotel,
            prefs.constraints,
        )
        self.assertEqual(spacious_outcomes[0].status.value, "unknown")
        self.assertEqual(
            build_structured_fact_arguments(
                spacious_hotel,
                spacious_outcomes,
            ),
            [],
        )

        spa_hotel = replace(
            spacious_hotel,
            metadata=replace(
                spacious_hotel.metadata,
                facilities=(
                    Facility(
                        name="Spa and wellness centre",
                        facility_ids=(902,),
                    ),
                ),
            ),
        )
        spa_outcomes = evaluate_constraints(spa_hotel, prefs.constraints)
        spa_arguments = build_structured_fact_arguments(
            spa_hotel,
            spa_outcomes,
        )
        self.assertEqual(spa_outcomes[0].status.value, "satisfied")
        self.assertEqual(len(spa_arguments), 1)
        self.assertEqual(spa_arguments[0].arg_type, "support")

    def test_absent_facility_is_unknown_not_attack(self):
        prefs = preferences(
            constraints=[
                {
                    "text": "Parking is required",
                    "importance_raw": 5,
                    "mode": "hard",
                    "field": "parking",
                }
            ]
        )
        outcomes = evaluate_constraints(self.hotel, prefs.constraints)
        self.assertEqual(outcomes[0].status.value, "unknown")
        self.assertEqual(
            build_structured_fact_arguments(self.hotel, outcomes),
            [],
        )


if __name__ == "__main__":
    unittest.main()
