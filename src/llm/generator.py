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
    
    def generate_batch(self, prompts: list[str], batch_size: int = 4) -> list[str]:
        outputs: list[str] = []

        if not prompts:
            return outputs

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            input_lengths = inputs["input_ids"].shape[1]

            generated_only = generated_ids[:, input_lengths:]

            decoded_outputs = self.tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=True,
            )

            outputs.extend(output.strip() for output in decoded_outputs)

        return outputs