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
    "outdoor_seating",
    "attire",
    "alcohol",
]


PROMPT_TEMPLATE = """
Generate recommendation arguments from the user history and target item.

Return ONLY a valid JSON object.
Do not add explanations, markdown, or code fences.

TASK:
Generate exactly 4 arguments:
- 2 support arguments
- 2 attack arguments

RULES:
- Every argument must be grounded in the input.
- Do not invent or contradict facts.
- If evidence says an attribute is True, do not claim it is False, and vice versa.
- Attack arguments must be based on real differences.
- Do not claim a disadvantage if the target and the compared item share the same attribute value.
- Evidence must be short, copied from the input, and start with the item name.
- Each argument must include 1 to 3 "used_aspects" selected from the allowed list.
- The selected "used_aspects" must describe the main aspect(s) discussed in the argument text and evidence.
- Do not select an aspect only because it appears in the input; select it only if the argument actually relies on it.
- If no aspect is clearly relevant, use [].
- Keep argument text short.

ALLOWED ASPECTS:
[{allowed_aspects}]

OUTPUT FORMAT:
{{
  "arguments": [
    {{
      "id": "A1",
      "type": "support",
      "text": "...",
      "used_aspects": ["food"],
      "evidence": ["Item name | short copied evidence"]
    }},
    {{
      "id": "A2",
      "type": "support",
      "text": "...",
      "used_aspects": ["service"],
      "evidence": ["Item name | short copied evidence"]
    }},
    {{
      "id": "A3",
      "type": "attack",
      "text": "...",
      "used_aspects": ["price"],
      "evidence": ["Item name | short copied evidence"]
    }},
    {{
      "id": "A4",
      "type": "attack",
      "text": "...",
      "used_aspects": ["noise"],
      "evidence": ["Item name | short copied evidence"]
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