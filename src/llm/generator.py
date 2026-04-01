import torch
from transformers import GenerationConfig

from .config import LLMConfig


class LocalLLMGenerator:
    def __init__(self, model, tokenizer, config: LLMConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def _get_input_device(self):
        for param in self.model.parameters():
            if param.device.type != "meta":
                return param.device
        return torch.device("cpu")

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
            return_dict=True,
        )

        input_device = self._get_input_device()
        model_inputs = {
            key: value.to(input_device)
            for key, value in model_inputs.items()
        }

        if self.config.do_sample:
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        outputs = self.model.generate(
            **model_inputs,
            generation_config=generation_config,
        )

        input_length = model_inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip()