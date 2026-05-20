from __future__ import annotations

import json

from src.argumentation.schema import Argument
from src.prompting.formatters import format_history, format_target_item


GEMINI_JOINT_SCORING_PROMPT_TEMPLATE = """
Evaluate all recommendation arguments jointly.

TASK:
Given:
- a user history
- a target item
- several generated arguments

Score each argument relative to the others.

SCORING CRITERIA:
- grounding in the input
- relevance to the recommendation decision
- importance relative to the other arguments
- distinguish minor attacks from strong preference conflicts

RULES:
- score must be between 0.0 and 1.0
- 0.0 means very weak or irrelevant argument
- 1.0 means very strong and decision-relevant argument
- reason must be short and explicit
- do not give high scores only because the argument is factually correct
- minor missing features should usually receive lower scores than strong preference matches or conflicts

USER_HISTORY:
{history}

TARGET_ITEM:
{target}

ARGUMENTS:
{arguments}
""".strip()


def build_gemini_joint_scoring_prompt(arguments: list[Argument]) -> str:
    if not arguments:
        raise ValueError("Cannot build joint scoring prompt from empty arguments.")

    first = arguments[0]
    history_str = format_history(first.history)
    target_str = format_target_item(first.target_item or {})

    argument_payload = [
        {
            "id": arg.id,
            "type": arg.arg_type,
            "text": arg.text,
            "used_aspects": arg.used_aspects,
            "aspect_effect": arg.aspect_effect,
            "evidence": arg.evidence,
        }
        for arg in arguments
    ]

    return GEMINI_JOINT_SCORING_PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        arguments=json.dumps(argument_payload, ensure_ascii=False, indent=2),
    )