from __future__ import annotations

import json

from src.argumentation.schema import Argument
from src.prompting.formatters import format_history, format_target_item


GEMINI_SCORING_PROMPT_TEMPLATE = """
Evaluate the recommendation argument.

TASK:
Given:
- a user history
- a target item
- one generated argument

Evaluate the argument according to:
- coherence
- grounding in the provided input
- relevance for deciding whether to recommend the target item

SCORING RULES:
- score must be between 0.0 and 1.0
- 0.0 means very poor argument
- 1.0 means very strong argument
- reason must be short and explicit

USER_HISTORY:
{history}

TARGET_ITEM:
{target}

ARGUMENT:
id: {argument_id}
type: {argument_type}
text: {argument_text}
evidence: {argument_evidence}
""".strip()


def build_gemini_scoring_prompt(argument: Argument) -> str:
    history_str = format_history(argument.history)
    target_str = format_target_item(argument.target_item or {})

    return GEMINI_SCORING_PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        argument_id=argument.id,
        argument_type=argument.arg_type,
        argument_text=argument.text,
        argument_evidence=json.dumps(argument.evidence, ensure_ascii=False),
    )