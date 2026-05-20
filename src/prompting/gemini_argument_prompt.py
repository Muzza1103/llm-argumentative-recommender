from src.prompting.argument_prompt import ALLOWED_ARGUMENT_ASPECTS


BALANCED_TASK = """
Generate exactly 4 arguments:
- 2 support arguments
- 2 attack arguments
""".strip()


UNBALANCED_TASK = """
Generate exactly 4 arguments in total.

The number of support and attack arguments does not need to be balanced:
- Generate more support arguments if the target item strongly matches the user preferences.
- Generate more attack arguments if the target item clearly conflicts with the user preferences.
- Do not invent weak attacks just to balance the output.
- Do not invent weak supports just to balance the output.
- The split between support and attack must reflect the available evidence.
- Only generate arguments that are grounded, relevant, and decision-important.
""".strip()


GEMINI_ARGUMENT_PROMPT_TEMPLATE = """
Generate recommendation arguments from the user history and target item.

Return ONLY a valid JSON object.
Do not add explanations, markdown, or code fences.

TASK:
{task}

RULES:
- Every argument must be grounded in the input.
- Do not invent or contradict facts.
- If evidence says an attribute is True, do not claim it is False, and vice versa.
- Attack arguments must be based on real differences or clear preference conflicts.
- Do not claim a disadvantage if the target and the compared item share the same attribute value.
- Evidence must be short, copied from the input, and start with the item name.
- Each argument must include 1 to 3 "used_aspects" selected from the allowed list.
- The selected "used_aspects" must describe the main aspect(s) discussed in the argument text and evidence.
- Do not select an aspect only because it appears in the input; select it only if the argument actually relies on it.
- If no aspect is clearly relevant, use [].
- Keep argument text short.
- Each argument must include exactly one "aspect_effect".
- Argument ids must be sequential: A1, A2, A3, A4.

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
- Prefer presence-based arguments over absence-based arguments when both are available.
- Do not generate absence-based attacks unless the missing aspect is clearly relevant to the user's preferences.

ALLOWED ASPECTS:
[{allowed_aspects}]

USER_HISTORY:
{history}

TARGET_ITEM:
{target}
""".strip()


def build_gemini_prompt(
    history_str: str,
    target_str: str,
    argument_mode: str = "balanced",
) -> str:
    if argument_mode == "balanced":
        task = BALANCED_TASK
    elif argument_mode == "unbalanced":
        task = UNBALANCED_TASK
    else:
        raise ValueError(f"Unknown argument mode: {argument_mode}")

    return GEMINI_ARGUMENT_PROMPT_TEMPLATE.format(
        task=task,
        history=history_str,
        target=target_str,
        allowed_aspects=", ".join(ALLOWED_ARGUMENT_ASPECTS),
    )