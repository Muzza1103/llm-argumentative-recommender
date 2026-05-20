import os
import asyncio
from typing import Any

from google import genai
from google.genai import types


ARGUMENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "arguments": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["support", "attack"],
                    },
                    "text": {"type": "string"},
                    "used_aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "aspect_effect": {
                        "type": "string",
                        "enum": [
                            "present_preferred",
                            "missing_preferred",
                            "present_disliked",
                            "missing_disliked",
                            "neutral_or_unclear",
                        ],
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "type",
                    "text",
                    "used_aspects",
                    "aspect_effect",
                    "evidence",
                ],
            },
        },
    },
    "required": ["arguments"],
}


SCORING_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "argument_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
            },
        },
    },
    "required": ["argument_scores"],
}


class GeminiGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "global",
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
        response_schema: dict[str, Any] | None = None,
        debug: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = int(max_output_tokens)
        self.response_schema = response_schema
        self.debug = debug

        self.client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location,
        )

    def generate(self, prompt: str) -> str:
        config_kwargs = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }

        if self.response_schema is not None:
            config_kwargs["response_schema"] = self.response_schema

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        if self.debug and response.candidates:
            print("Gemini finish reason:", response.candidates[0].finish_reason)

        return response.text.strip() if response.text else ""

    async def _async_generate(self, prompt: str) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    async def generate_batch_async(
        self,
        prompts: list[str],
        batch_size: int = 5,
    ) -> list[str]:
        outputs = []

        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]

            print(
                f"Gemini async batch "
                f"{start + 1}-{start + len(batch)}/{len(prompts)}"
            )

            tasks = [
                self._async_generate(prompt)
                for prompt in batch
            ]

            batch_outputs = await asyncio.gather(*tasks)
            outputs.extend(batch_outputs)

        return outputs

    def generate_batch(
        self,
        prompts: list[str],
        batch_size: int = 5,
    ) -> list[str]:
        return asyncio.run(
            self.generate_batch_async(
                prompts=prompts,
                batch_size=batch_size,
            )
        )