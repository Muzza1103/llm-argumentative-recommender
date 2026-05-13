import os

from google import genai
from google.genai import types


class GeminiGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "global",
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self.client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        return response.text.strip() if response.text else ""

    def generate_batch(self, prompts: list[str], batch_size: int = 1) -> list[str]:
        outputs = []

        for i, prompt in enumerate(prompts, start=1):
            print(f"Gemini request {i}/{len(prompts)}")
            outputs.append(self.generate(prompt))

        return outputs