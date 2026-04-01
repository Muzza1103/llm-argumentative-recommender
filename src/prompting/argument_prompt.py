PROMPT_TEMPLATE = """
You are an assistant that generates structured arguments for a recommendation task.

TASK:
Given a user history and a target item, generate arguments FOR and AGAINST recommending the target item to the user.

INSTRUCTIONS:
- Infer user preferences from the history.
- Compare the target item with positively and negatively rated history items.
- Generate exactly 4 arguments: 2 support and 2 attack.
- Each argument must be specific and grounded in the input.
- History items only contain name, categories, and rating.
- Only the target item has attributes.
- Do not invent missing facts or attributes.
- Keep each argument text short.
- Use at most 2 short evidence snippets per argument.
- Return valid JSON only.

OUTPUT FORMAT:
{{
  "arguments": [
    {{
      "id": "A1",
      "type": "support",
      "text": "...",
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A2",
      "type": "support",
      "text": "...",
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A3",
      "type": "attack",
      "text": "...",
      "evidence": ["...", "..."]
    }},
    {{
      "id": "A4",
      "type": "attack",
      "text": "...",
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
    )