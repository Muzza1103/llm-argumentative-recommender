PROMPT_TEMPLATE = """
You are an assistant that generates structured arguments for a recommendation task.

TASK:
Given a user history and a target item, generate arguments FOR and AGAINST recommending the target item to the user.

INSTRUCTIONS:
- Infer preferences from the user history
- Compare the target item with past liked and disliked items
- Generate specific arguments only
- Do not generate generic statements
- Each argument must be grounded in the provided input
- Return only valid JSON
- Do not include any explanation outside the JSON

OUTPUT FORMAT:
{
  "arguments": [
    {
      "id": "A1",
      "type": "support",
      "text": "...",
      "evidence": ["..."]
    },
    {
      "id": "A2",
      "type": "attack",
      "text": "...",
      "evidence": ["..."]
    }
  ]
}

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