from __future__ import annotations


def build_review_aspect_exploration_prompt(
    item_name: str,
    rating: float | int | None,
    review_text: str,
    source: str,
) -> str:
    return f"""
You are helping design a controlled vocabulary of restaurant review aspects.

TASK:
Given the restaurant review below, extract the most salient review aspects mentioned by the user.

IMPORTANT:
- Return only short aspect labels.
- Do NOT return full sentences.
- Use concise normalized aspect names when possible.
- Focus on restaurant-related aspects such as food, service, ambience, speed, price, freshness, noise, portions, family/group suitability, takeout, delivery, reservations, drinks, etc.
- Return at most 5 aspects.
- Return valid JSON only.

OUTPUT FORMAT:
{{
  "aspects": ["...", "..."]
}}

REVIEW CONTEXT:
Item name: {item_name}
User rating: {rating}
Source: {source}

REVIEW TEXT:
{review_text}
""".strip()


def build_review_aspect_extraction_prompt(
    item_name: str,
    rating: float | int | None,
    review_text: str,
    source: str,
    allowed_aspects: list[str],
) -> str:
    allowed_str = ", ".join(allowed_aspects)

    return f"""
You are extracting normalized restaurant review aspects.

TASK:
Given the review below, extract the relevant aspects mentioned by the user.

IMPORTANT:
- You MUST use only aspects from the allowed list below.
- If no aspect applies, return an empty list.
- Return at most 5 aspects.
- Return valid JSON only.

ALLOWED ASPECTS:
[{allowed_str}]

OUTPUT FORMAT:
{{
  "aspects": [
    {{"name": "...", "polarity": "positive|negative|neutral"}}
  ]
}}

REVIEW CONTEXT:
Item name: {item_name}
User rating: {rating}
Source: {source}

REVIEW TEXT:
{review_text}
""".strip()