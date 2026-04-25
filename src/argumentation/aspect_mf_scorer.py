import json
from pathlib import Path

from .scoring import BaseMFScorer
from .schema import Argument


class AspectMFScorer(BaseMFScorer):
    """
    Empirical scorer based on aspect-level MF predictions.

    It scores an argument by averaging the predicted MF scores of the aspects
    mentioned by the argument for the current user.
    """

    def __init__(
        self,
        predictions_path: str | Path,
        user_id: str,
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
            aspect = str(aspect)

            predictions.setdefault(user_id, {})[aspect] = float(score)

        return predictions

    def score(self, argument: Argument) -> float:
        aspects = self._get_argument_aspects(argument)

        if not aspects:
            return self.default_score

        user_predictions = self.predictions.get(self.user_id, {})

        scores = []
        for aspect in aspects:
            if aspect in user_predictions:
                scores.append(user_predictions[aspect])

        if not scores:
            return self.default_score

        return sum(scores) / len(scores)

    def _get_argument_aspects(self, argument: Argument) -> list[str]:
        """
        Supports several possible field names to stay compatible
        with current/future argument schemas.
        """
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