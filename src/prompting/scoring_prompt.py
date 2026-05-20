from __future__ import annotations

import json

from src.argumentation.schema import Argument
from src.prompting.formatters import format_history, format_target_item


SCORING_PROMPT_TEMPLATE = """
You are an evaluator for recommendation arguments.

TASK:
Given:
- a user history
- a target item
- several generated arguments

Evaluate each argument jointly and assign one score per argument.

SCORING CRITERIA:
- Coherence: is the argument logically consistent?
- Grounding: is it supported by the provided input?
- Relevance: is it useful for deciding whether to recommend the target item?
- Relative importance: how important is this argument compared with the others?

Return only valid JSON in the following format:
{{
  "argument_scores": [
    {{
      "id": "A1",
      "score": 0.0,
      "reason": "short explanation"
    }}
  ]
}}

RULES:
- Each score must be a float between 0.0 and 1.0.
- 0.0 means very weak, irrelevant, or poorly grounded.
- 1.0 means very strong, well-grounded, and decision-relevant.
- The reason must be short and explicit.
- Score arguments relative to each other.
- Do not give a high score only because an argument is factually correct.
- Minor missing features should usually receive lower scores than strong preference matches or conflicts.
- Do not return any text outside the JSON.

USER_HISTORY:
{history}

TARGET_ITEM:
{target}

ARGUMENTS:
{arguments}
""".strip()


def build_scoring_prompt(argument: Argument) -> str:
    """
    Backward-compatible single-argument scoring prompt.
    """
    return build_joint_scoring_prompt([argument])


def build_joint_scoring_prompt(arguments: list[Argument]) -> str:
    if not arguments:
        raise ValueError("Cannot build scoring prompt from empty arguments.")

    first = arguments[0]
    history_str = format_history(first.history)
    target_str = format_target_item(first.target_item or {})

    argument_payload = [
        {
            "id": argument.id,
            "type": argument.arg_type,
            "text": argument.text,
            "used_aspects": argument.used_aspects,
            "aspect_effect": argument.aspect_effect,
            "evidence": argument.evidence,
        }
        for argument in arguments
    ]

    return SCORING_PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        arguments=json.dumps(argument_payload, ensure_ascii=False, indent=2),
    )