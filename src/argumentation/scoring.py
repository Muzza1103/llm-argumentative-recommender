from __future__ import annotations
from dataclasses import dataclass
from .schema import Argument


@dataclass
class ScoreConfig:
    """
    Configuration of the weight factor assigned to each element, semantic scoring (llm) and empiric scoring (mf).
    """
    llm_weight: float = 0.5
    mf_weight: float = 0.5

    def validate(self):
        total = self.llm_weight + self.mf_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Invalid score weights: llm_weight + mf_weight must equal 1.0, got {total}."
            )


class BaseLLMScorer:
    """
    Base interface for semantic scoring with an LLM.
    Will be updated later
    """

    def score(self, argument: Argument) -> float:
        raise NotImplementedError


class BaseMFScorer:
    """
    Base interface for empirical / collaborative scoring.
    Will be updated later
    """

    def score(self, argument: Argument) -> float:
        raise NotImplementedError


class DummyMFScorer(BaseMFScorer):
    """
    Temporary placeholder collaborative scorer to test argument structure.
    """

    def score(self, argument: Argument) -> float:
        # if target has a reasonably good global rating, give a moderate score
        target_item = argument.target_item or {}
        global_stars = target_item.get("global_stars")

        if isinstance(global_stars, (int, float)):
            # normalize from [1, 5] to [0, 1]
            return max(0.0, min(1.0, (global_stars - 1.0) / 4.0))

        return 0.5


def combine_scores(
    llm_score: float,
    mf_score: float,
    config: ScoreConfig,
) -> float:
    config.validate()

    return (
        config.llm_weight * llm_score
        + config.mf_weight * mf_score
    )


def score_argument(
    argument: Argument,
    llm_scorer: BaseLLMScorer,
    mf_scorer: BaseMFScorer,
    config: ScoreConfig,
) -> Argument:
    """
    Score one Argument and return it.
    """
    llm_score = llm_scorer.score(argument)
    mf_score = mf_scorer.score(argument)
    combined_score = combine_scores(llm_score, mf_score, config)

    argument.llm_score = llm_score
    argument.mf_score = mf_score
    argument.combined_score = combined_score

    return argument


def score_arguments(
    arguments: list[Argument],
    llm_scorer: BaseLLMScorer,
    mf_scorer: BaseMFScorer,
    config: ScoreConfig,
) -> list[Argument]:
    """
    Score a list of arguments.
    """
    return [
        score_argument(argument, llm_scorer, mf_scorer, config)
        for argument in arguments
    ]