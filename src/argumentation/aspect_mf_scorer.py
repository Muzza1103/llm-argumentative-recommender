import json
from pathlib import Path

from .scoring import BaseMFScorer
from .schema import Argument


ASPECT_ALIASES = {
    "outdoor_seating": [
        "outdoor_seating",
        "ambience",
    ],
    "attire": [
        "attire",
        "attire_casual",
        "attire_dressy",
    ],
    "drinks": [
        "drinks",
        "alcohol_full_bar",
        "alcohol_beer_and_wine",
        "alcohol_none",
    ],
    "noise": [
        "noise",
        "noise_quiet",
        "noise_average",
        "noise_loud",
        "noise_very_loud",
    ],
    "price": [
        "price",
        "price_1",
        "price_2",
        "price_3",
        "price_4",
    ],
    "ambience": [
        "ambience",
        "attire_casual",
        "attire_dressy",
        "outdoor_seating",
    ],
    "takeout": [
        "takeout",
    ],
    "delivery": [
        "delivery",
    ],
    "reservations": [
        "reservations",
    ],
    "good_for_groups": [
        "good_for_groups",
    ],
    "good_for_kids": [
        "good_for_kids",
    ],
    "food": [
        "food",
        "quality",
        "freshness",
        "portions",
    ],
}


class AspectMFScorer(BaseMFScorer):
    """
    Empirical scorer based on aspect-level MF predictions.

    It scores an argument by averaging the predicted MF scores of the aspects
    mentioned by the argument for the current user.

    The scorer also supports aliases, so general argument aspects such as
    "drinks" can be mapped to more specific MF aspects such as "alcohol_full_bar".
    """

    def __init__(
        self,
        predictions_path: str | Path,
        user_id: str | None = None,
        default_score: float = 0.5,
    ):
        self.predictions_path = Path(predictions_path)
        self.user_id = user_id
        self.default_score = default_score
        self.predictions = self._load_predictions(self.predictions_path)

    def _load_predictions(self, path: Path) -> dict[str, dict[str, float]]:
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        predictions: dict[str, dict[str, float]] = {}

        for record in records:
            user_id = record.get("user_id")
            aspect = record.get("aspect")
            score = record.get("score")

            if user_id is None or aspect is None or score is None:
                continue

            user_id = str(user_id)
            aspect = str(aspect).strip().lower()

            predictions.setdefault(user_id, {})[aspect] = float(score)

        return predictions

    def set_user(self, user_id: str | None):
        self.user_id = user_id

    def score(self, argument: Argument) -> float:
        if not self.user_id:
            return self.default_score

        aspects = self._get_argument_aspects(argument)

        if not aspects:
            return self.default_score

        user_predictions = self.predictions.get(str(self.user_id), {})

        scores = []

        for aspect in aspects:
            candidate_aspects = self._expand_aspect_aliases(aspect)

            candidate_scores = [
                user_predictions[candidate]
                for candidate in candidate_aspects
                if candidate in user_predictions
            ]

            if candidate_scores:
                scores.append(sum(candidate_scores) / len(candidate_scores))

        if not scores:
            return self.default_score

        return sum(scores) / len(scores)

    def _expand_aspect_aliases(self, aspect: str) -> list[str]:
        aspect = aspect.strip().lower()

        aliases = ASPECT_ALIASES.get(aspect, [aspect])

        cleaned = []
        seen = set()

        for alias in aliases:
            alias = alias.strip().lower()
            if not alias:
                continue

            if alias in seen:
                continue

            seen.add(alias)
            cleaned.append(alias)

        return cleaned

    def _get_argument_aspects(self, argument: Argument) -> list[str]:
        aspects = []

        for field_name in [
            "used_aspects",
            "aspects",
            "used_categories",
            "used_attributes",
            "used_review_aspects",
        ]:
            value = getattr(argument, field_name, None)

            if isinstance(value, list):
                aspects.extend(value)

        cleaned = []
        seen = set()

        for aspect in aspects:
            if not isinstance(aspect, str):
                continue

            aspect = aspect.strip().lower()
            if not aspect:
                continue

            if aspect in seen:
                continue

            seen.add(aspect)
            cleaned.append(aspect)

        return cleaned