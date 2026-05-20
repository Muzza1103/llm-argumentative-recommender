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
- arguments based on the presence of an important aspect should generally receive higher scores than arguments based only on the absence of an aspect
- do not assign very high scores to absence-based arguments unless the missing or absent aspect is clearly central to the user's preferences

ASPECT_EFFECT SCORING GUIDANCE:
- "present_preferred":
  the target contains an aspect the user seems to value.
  This can receive a high score if the preference appears important.

- "present_disliked":
  the target contains an aspect the user seems to dislike.
  This can receive a high score if the negative aspect is clearly important.

- "missing_preferred":
  the target lacks an aspect the user usually likes.
  This should usually receive a medium score.
  Only assign a very high score if the missing aspect appears critical or repeatedly central in the user's history.

- "missing_disliked":
  the target lacks an aspect the user dislikes.
  This should usually receive a low or medium score unless the absence strongly improves the recommendation.

- "neutral_or_unclear":
  use moderate scores unless the argument is especially convincing.

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