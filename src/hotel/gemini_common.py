from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .errors import HotelGeminiError


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True, slots=True)
class GeminiJSONResult:
    payload: dict[str, Any]
    trace: dict[str, Any]


def _usage_payload(response: object) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return {}
    mapping = {
        "prompt_tokens": "prompt_token_count",
        "output_tokens": "candidates_token_count",
        "total_tokens": "total_token_count",
        "thoughts_tokens": "thoughts_token_count",
    }
    usage = {}
    for public_name, attribute_name in mapping.items():
        value = getattr(metadata, attribute_name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            usage[public_name] = value
    return usage


def _create_client(api_key: str | None = None):
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise HotelGeminiError(
            "GEMINI_API_KEY is required for a real Gemini call"
        )
    try:
        from google import genai
    except ModuleNotFoundError as exc:
        raise HotelGeminiError(
            "google-genai is required for a real Gemini call"
        ) from exc
    return genai.Client(api_key=resolved_key)


def call_gemini_json(
    *,
    prompt: str,
    response_schema: dict[str, Any],
    model_name: str = DEFAULT_GEMINI_MODEL,
    client: object | None = None,
    api_key: str | None = None,
    max_output_tokens: int = 4096,
    thinking_budget: int = 1024,
) -> GeminiJSONResult:
    resolved_client = client or _create_client(api_key)
    config = {
        "temperature": 0.0,
        "max_output_tokens": int(max_output_tokens),
        "response_mime_type": "application/json",
        "response_json_schema": response_schema,
        "thinking_config": {
            "thinking_budget": int(thinking_budget),
            "include_thoughts": False,
        },
    }
    try:
        response = resolved_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        raise HotelGeminiError(
            f"Gemini provider error ({type(exc).__name__})"
        ) from exc

    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise HotelGeminiError("Gemini returned an empty JSON response")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HotelGeminiError(
            f"Gemini returned invalid JSON at character {exc.pos}"
        ) from exc
    if not isinstance(payload, dict):
        raise HotelGeminiError("Gemini JSON response must be an object")

    trace = {
        "model_requested": model_name,
        "model_version": getattr(response, "model_version", None),
        "response_id": getattr(response, "response_id", None),
        "raw_response": raw_text.strip(),
        "usage": _usage_payload(response),
        "request_count": 1,
    }
    return GeminiJSONResult(payload=payload, trace=trace)
