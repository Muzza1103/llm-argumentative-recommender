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
- one generated argument

Evaluate how good the argument is.

SCORING CRITERIA:
- Coherence: is the argument logically consistent?
- Grounding: is it supported by the provided input?
- Relevance: is it useful for deciding whether to recommend the target item?

Return only valid JSON in the following format:
{{
  "score": 0.0,
  "reason": "short explanation"
}}

RULES:
- The score must be a float between 0.0 and 1.0
- 0.0 means very poor argument
- 1.0 means very strong argument
- Do not return any text outside the JSON

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


def build_scoring_prompt(argument: Argument) -> str:
    history_str = format_history(argument.history)
    target_str = format_target_item(argument.target_item or {})

    return SCORING_PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        argument_id=argument.id,
        argument_type=argument.arg_type,
        argument_text=argument.text,
        argument_evidence=json.dumps(argument.evidence, ensure_ascii=False),
    )