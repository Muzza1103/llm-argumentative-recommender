import json
import unittest
from pathlib import Path

from scripts.hotel.validate_hotel_data import build_validation_summary
from src.hotel import (
    HOTEL_ASPECTS,
    AnnotationStatus,
    HotelDataValidationError,
    HotelPolicy,
    Stance,
    hotel_profile_dataset_from_dict,
    hotel_review_from_dict,
    load_hotel_profiles,
    load_review_annotations,
    validate_reviews_match_profiles,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PROFILE_FIXTURE = FIXTURE_DIR / "hotel_profiles_minimal.json"
REVIEW_FIXTURE = FIXTURE_DIR / "hotel_review_annotations_minimal.jsonl"


class HotelLoaderTests(unittest.TestCase):
    def _profile_dict(self):
        return json.loads(PROFILE_FIXTURE.read_text(encoding="utf-8"))

    def _review_dict(self, index: int = 0):
        lines = REVIEW_FIXTURE.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[index])

    def test_official_vocabulary_contains_exactly_fifteen_aspects(self):
        self.assertEqual(len(HOTEL_ASPECTS), 15)
        self.assertEqual(len(set(HOTEL_ASPECTS)), 15)

    def test_loads_profiles_and_keeps_unknown_distinct_from_negative(self):
        dataset = load_hotel_profiles(PROFILE_FIXTURE)
        hotel = dataset.hotels[0]

        self.assertEqual(dataset.n_hotels, 1)
        self.assertEqual(dataset.n_reviews, 3)
        self.assertIsNone(hotel.metadata.city)
        self.assertEqual(hotel.metadata.policies, ())
        self.assertIsNone(hotel.get_aspect("wifi_internet"))
        self.assertEqual(hotel.get_aspect("bruit_calme").attack_count, 1)

    def test_deduplicates_facilities_by_normalized_name(self):
        hotel = load_hotel_profiles(PROFILE_FIXTURE).hotels[0]

        self.assertEqual(len(hotel.metadata.facilities), 2)
        self.assertEqual(hotel.metadata.facilities[0].name, "Free WiFi")
        self.assertEqual(hotel.metadata.facilities[0].facility_ids, (107, 492))

    def test_loads_all_annotation_statuses_without_dropping_reviews(self):
        reviews = load_review_annotations(REVIEW_FIXTURE)

        self.assertEqual(len(reviews), 3)
        self.assertEqual(
            [review.annotation_status for review in reviews],
            [
                AnnotationStatus.ANNOTATED,
                AnnotationStatus.ANNOTATED_NO_ASPECT,
                AnnotationStatus.MISSING_OR_ERROR,
            ],
        )
        self.assertEqual(reviews[0].signals[0].stance, Stance.SUPPORT)
        self.assertEqual(reviews[1].signals, ())
        self.assertTrue(reviews[2].annotation_failed)

    def test_standalone_and_embedded_reviews_match(self):
        dataset = load_hotel_profiles(PROFILE_FIXTURE)
        reviews = load_review_annotations(REVIEW_FIXTURE)
        validate_reviews_match_profiles(dataset, reviews)

        summary = build_validation_summary(dataset, reviews)
        self.assertTrue(summary["cross_file_match"])
        self.assertEqual(summary["n_hotels"], 1)
        self.assertEqual(summary["n_review_annotations"], 3)
        self.assertEqual(summary["n_raw_facilities"], 3)
        self.assertEqual(summary["n_deduplicated_facilities"], 2)

    def test_rejects_an_unknown_aspect_instead_of_normalizing_it(self):
        review = self._review_dict()
        review["signals"][0]["aspect"] = "location"

        with self.assertRaisesRegex(
            HotelDataValidationError,
            "unknown hotel aspect 'location'",
        ):
            hotel_review_from_dict(review)

    def test_rejects_an_unknown_annotation_status(self):
        review = self._review_dict()
        review["annotation_status"] = "failed"

        with self.assertRaisesRegex(HotelDataValidationError, "expected one of"):
            hotel_review_from_dict(review)

    def test_rejects_signals_for_no_aspect_status(self):
        review = self._review_dict()
        review["annotation_status"] = "annotated_no_aspect"

        with self.assertRaisesRegex(
            HotelDataValidationError,
            "must not contain signals",
        ):
            hotel_review_from_dict(review)

    def test_rejects_catalog_with_an_extra_aspect(self):
        dataset = self._profile_dict()
        dataset["aspect_catalog"]["spa"] = "Unknown alias."

        with self.assertRaisesRegex(HotelDataValidationError, "catalog mismatch"):
            hotel_profile_dataset_from_dict(dataset)

    def test_rejects_declared_review_count_mismatch(self):
        dataset = self._profile_dict()
        dataset["n_reviews"] = 4

        with self.assertRaisesRegex(
            HotelDataValidationError,
            "declared n_reviews",
        ):
            hotel_profile_dataset_from_dict(dataset)

    def test_policy_semantic_payload_excludes_id_and_empty_fields(self):
        policy = HotelPolicy(
            source_id=42,
            name="Pets",
            policy_type="POLICY_HOTEL_PETS",
            description="Pets allowed",
            parking="",
        )

        self.assertEqual(
            policy.semantic_payload(),
            {
                "name": "Pets",
                "policy_type": "POLICY_HOTEL_PETS",
                "description": "Pets allowed",
            },
        )


if __name__ == "__main__":
    unittest.main()
