ALLOWED_ARGUMENT_ASPECTS = [
    "food",
    "service",
    "ambience",
    "price",
    "portions",
    "drinks",
    "speed",
    "takeout",
    "delivery",
    "reservations",
    "good_for_groups",
    "good_for_kids",
    "noise",
    "freshness",
    "quality",
    "location",
    "spice_level",
    "crowdedness",
    "selection",
]


PROMPT_TEMPLATE = """
You are an assistant that generates structured arguments for a recommendation task.

TASK:
Given a user history and a target item, generate arguments FOR and AGAINST recommending the target item to the user.

INSTRUCTIONS:
- Infer user preferences from the history.
- Compare the target item with positively and negatively rated history items.
- Generate exactly 4 arguments: 2 support and 2 attack.
- Each argument must be specific and grounded in the input.
- History items contain name, categories, rating, sometimes filtered attributes, and sometimes extracted review aspects.
- The target item contains categories, filtered attributes, and sometimes extracted review aspects.
- Do not invent missing facts or attributes.
- Do not contradict the evidence.
- Keep each argument text short.
- Use at most 2 short evidence snippets per argument.
- Each argument MUST include "used_aspects".
- "used_aspects" must contain 1 to 3 aspects selected only from the allowed aspects list.
- The selected aspects must be directly related to the argument text and evidence.
- If no aspect is clearly relevant, use an empty list.
- Return valid JSON only.

ALLOWED ASPECTS:
[{allowed_aspects}]

OUTPUT FORMAT:
{{
  "arguments": [
    {{
      "id": "A1",
      "type": "support",
      "text": "...",
      "used_aspects": ["food", "service"],
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A2",
      "type": "support",
      "text": "...",
      "used_aspects": ["price"],
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A3",
      "type": "attack",
      "text": "...",
      "used_aspects": ["noise"],
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A4",
      "type": "attack",
      "text": "...",
      "used_aspects": ["takeout", "delivery"],
      "evidence": ["...", "..."]
    }}
  ]
}}

USER_HISTORY:
{history}

TARGET_ITEM:
{target}
""".strip()


def build_prompt(history_str: str, target_str: str) -> str:
    return PROMPT_TEMPLATE.format(
        history=history_str,
        target=target_str,
        allowed_aspects=", ".join(ALLOWED_ARGUMENT_ASPECTS),
    )