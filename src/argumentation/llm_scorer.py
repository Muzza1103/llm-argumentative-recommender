from __future__ import annotations

from dataclasses import dataclass

from src.argumentation.schema import Argument
from src.llm.generator import LocalLLMGenerator
from src.llm.utils import extract_first_json_object
from src.prompting.scoring_prompt import build_scoring_prompt


@dataclass
class LLMScorerConfig:
    """
    Configuration for semantic argument scoring with a local LLM.
    """
    min_score: float = 0.0
    max_score: float = 1.0
    default_score: float = 0.5
    default_reason: str = "No valid scoring explanation returned by the LLM."


class LocalLLMScorer:
    """
    Semantic scorer based on a local Hugging Face LLM.

    The scorer evaluates how coherent, grounded, and relevant
    a generated argument is with respect to:
    - the user history
    - the target item
    """

    def __init__(
        self,
        generator: LocalLLMGenerator,
        config: LLMScorerConfig | None = None,
    ):
        self.generator = generator
        self.config = config or LLMScorerConfig()

    def _normalize_score(self, score: float) -> float:
        if score < self.config.min_score:
            return self.config.min_score
        if score > self.config.max_score:
            return self.config.max_score
        return score

    def score(self, argument: Argument) -> float:
        prompt = build_scoring_prompt(argument)
        output_text = self.generator.generate(prompt)
        parsed_json = extract_first_json_object(output_text)

        argument.llm_scoring_prompt = prompt
        argument.llm_scoring_raw_output = output_text

        if not isinstance(parsed_json, dict):
            argument.llm_score_reason = self.config.default_reason
            return self.config.default_score

        raw_score = parsed_json.get("score")
        raw_reason = parsed_json.get("reason")

        if not isinstance(raw_reason, str) or raw_reason.strip() == "":
            raw_reason = self.config.default_reason

        argument.llm_score_reason = raw_reason.strip()

        if not isinstance(raw_score, (int, float)):
            return self.config.default_score

        return self._normalize_score(float(raw_score))