from transformers import GenerationConfig

from .config import LLMConfig


class LocalLLMGenerator:
    def __init__(self, model, tokenizer, config: LLMConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate structured recommendation arguments. "
                    "You must answer with valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(
            model_inputs,
            generation_config=generation_config,
        )

        generated_ids = outputs[0][model_inputs.shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip()