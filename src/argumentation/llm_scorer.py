from __future__ import annotations

from dataclasses import dataclass

from src.argumentation.schema import Argument
from src.llm.generator import LocalLLMGenerator
from src.llm.utils import extract_first_json_object
from src.prompting.scoring_prompt import build_joint_scoring_prompt
from src.prompting.gemini_scoring_prompt import build_gemini_joint_scoring_prompt


@dataclass
class LLMScorerConfig:
    """
    Configuration for semantic argument scoring with an LLM.
    """
    min_score: float = 0.0
    max_score: float = 1.0
    default_score: float = 0.5
    default_reason: str = "No valid scoring explanation returned by the LLM."


class LocalLLMScorer:
    """
    Semantic scorer based on a generator.

    It can score arguments jointly so the model can calibrate
    their relative importance within the same recommendation context.
    """

    def __init__(
        self,
        generator: LocalLLMGenerator,
        config: LLMScorerConfig | None = None,
        use_gemini_prompt: bool = False,
    ):
        self.generator = generator
        self.config = config or LLMScorerConfig()
        self.use_gemini_prompt = use_gemini_prompt

    def _normalize_score(self, score: float) -> float:
        if score < self.config.min_score:
            return self.config.min_score
        if score > self.config.max_score:
            return self.config.max_score
        return score

    def _build_prompt(self, arguments: list[Argument]) -> str:
        if self.use_gemini_prompt:
            return build_gemini_joint_scoring_prompt(arguments)

        return build_joint_scoring_prompt(arguments)

    def score(self, argument: Argument) -> float:
        """
        Backward-compatible single-argument scoring.

        Internally, it uses the joint scoring format with one argument.
        """
        scores = self.score_many([argument])

        if not scores:
            return self.config.default_score

        return scores[0]

    def score_many(self, arguments: list[Argument]) -> list[float]:
        if not arguments:
            return []

        prompt = self._build_prompt(arguments)
        output_text = self.generator.generate(prompt)
        parsed_json = extract_first_json_object(output_text)

        scores_by_id: dict[str, float] = {}
        reasons_by_id: dict[str, str] = {}

        if isinstance(parsed_json, dict):
            argument_scores = parsed_json.get("argument_scores", [])

            if isinstance(argument_scores, list):
                for item in argument_scores:
                    if not isinstance(item, dict):
                        continue

                    arg_id = item.get("id")
                    raw_score = item.get("score")
                    raw_reason = item.get("reason")

                    if isinstance(arg_id, str) and isinstance(raw_score, (int, float)):
                        scores_by_id[arg_id] = self._normalize_score(float(raw_score))

                    if isinstance(arg_id, str) and isinstance(raw_reason, str):
                        reasons_by_id[arg_id] = raw_reason.strip()

        output_scores = []

        for argument in arguments:
            argument.llm_scoring_prompt = prompt
            argument.llm_scoring_raw_output = output_text

            argument.llm_score_reason = reasons_by_id.get(
                argument.id,
                self.config.default_reason,
            )

            score = scores_by_id.get(argument.id, self.config.default_score)
            output_scores.append(score)

        return output_scores