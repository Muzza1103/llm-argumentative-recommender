from src.prompting.argument_prompt import ALLOWED_ARGUMENT_ASPECTS


GEMINI_ARGUMENT_PROMPT_TEMPLATE = """
Generate recommendation arguments from the user history and target item.

TASK:
Generate exactly 4 arguments:
- 2 support arguments
- 2 attack arguments

CONTENT RULES:
- Every argument must be grounded in the input.
- Do not invent or contradict facts.
- If evidence says an attribute is True, do not claim it is False, and vice versa.
- Attack arguments must be based on real differences.
- Do not claim a disadvantage if the target and the compared item share the same attribute value.
- Evidence must be short, copied from the input, and start with the item name.
- Each argument must include 1 to 3 evidence strings, never more.
- Each argument must include 1 to 3 used_aspects selected from the allowed list.
- used_aspects must describe the main aspect(s) actually used by the argument.
- Do not select an aspect only because it appears in the input.
- Keep argument text short.
- Each argument must include exactly one "aspect_effect".

ASPECT_EFFECT VALUES:
- "present_preferred": the target has an aspect that seems positive or preferred by the user.
- "missing_preferred": the target lacks an aspect that seems positive or preferred by the user.
- "present_disliked": the target has an aspect that seems negative or disliked by the user.
- "missing_disliked": the target lacks an aspect that seems negative or disliked by the user.
- "neutral_or_unclear": the aspect relation is unclear.

ASPECT_EFFECT RULES:
- For support arguments, use "present_preferred" or "missing_disliked" when possible.
- For attack arguments, use "missing_preferred" or "present_disliked" when possible.
- Use "neutral_or_unclear" only if the relation cannot be determined from the input.

ALLOWED ASPECTS:
[{allowed_aspects}]

USER_HISTORY:
{history}

TARGET_ITEM:
{target}
""".strip()


def build_gemini_prompt(history_str: str, target_str: str) -> str:
    return GEMINI_ARGUMENT_PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        allowed_aspects=", ".join(ALLOWED_ARGUMENT_ASPECTS),
    )