from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .scoring import BaseMFScorer
from .schema import Argument


@dataclass
class MFScorerConfig:
    """
    Configuration for empirical scoring based on precomputed MF predictions.
    """
    default_score: float = 0.5


class GlobalRatingFallbackMFScorer(BaseMFScorer):
    """
    Temporary fallback empirical scorer.
    It simply maps the target item's global rating to [0, 1].

    Useful while the MF pipeline is not available yet.
    """

    def __init__(self, default_score: float = 0.5):
        self.default_score = default_score

    def score(self, argument: Argument) -> float:
        target_item = argument.target_item or {}
        global_stars = target_item.get("global_stars")

        if isinstance(global_stars, (int, float)):
            return max(0.0, min(1.0, (float(global_stars) - 1.0) / 4.0))

        return self.default_score


class PrecomputedMFScorer(BaseMFScorer):
    """
    Empirical scorer based on precomputed user-item MF predictions.

    Expected key:
        (user_id, business_id) -> score in [0, 1]

    The same user-item score is assigned to all arguments
    attached to the same target item.
    """

    def __init__(
        self,
        predictions: dict[tuple[str, str], float],
        config: MFScorerConfig | None = None,
    ):
        self.predictions = predictions
        self.config = config or MFScorerConfig()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        config: MFScorerConfig | None = None,
    ) -> "PrecomputedMFScorer":
        """
        Load MF predictions from a JSON file.

        Supported formats:

        1. List of records:
        [
          {"user_id": "...", "business_id": "...", "score": 0.73},
          ...
        ]

        2. Dict with composite keys:
        {
          "user_id|||business_id": 0.73,
          ...
        }
        """
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        predictions: dict[tuple[str, str], float] = {}

        if isinstance(data, list):
            for row in data:
                user_id = row.get("user_id")
                business_id = row.get("business_id")
                score = row.get("score")

                if (
                    isinstance(user_id, str)
                    and isinstance(business_id, str)
                    and isinstance(score, (int, float))
                ):
                    predictions[(user_id, business_id)] = float(score)

        elif isinstance(data, dict):
            for key, score in data.items():
                if not isinstance(key, str):
                    continue
                if not isinstance(score, (int, float)):
                    continue
                if "|||" not in key:
                    continue

                user_id, business_id = key.split("|||", 1)
                predictions[(user_id, business_id)] = float(score)

        return cls(predictions=predictions, config=config)

    def score(self, argument: Argument) -> float:
        user_id = argument.user_id
        target_item = argument.target_item or {}
        business_id = target_item.get("business_id")

        if not isinstance(user_id, str) or not isinstance(business_id, str):
            return self.config.default_score

        score = self.predictions.get((user_id, business_id), self.config.default_score)

        return max(0.0, min(1.0, float(score)))