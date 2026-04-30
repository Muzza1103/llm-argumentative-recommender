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

        self.tokenizer.padding_side = "left"

        input_device = self._get_input_device()
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for start in range(0, len(prompts), batch_size):
            batch_number = start // batch_size + 1
            batch_prompts = prompts[start:start + batch_size]

            print(f"Generating batch {batch_number}/{total_batches}")

            messages_batch = [
                [
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
                for prompt in batch_prompts
            ]

            texts = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages_batch
            ]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            inputs = {
                key: value.to(input_device)
                for key, value in inputs.items()
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

            generated_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

            input_lengths = inputs["attention_mask"].sum(dim=1)

            for output_ids, input_length in zip(generated_ids, input_lengths):
                generated_only = output_ids[int(input_length):]
                text = self.tokenizer.decode(
                    generated_only,
                    skip_special_tokens=True,
                )
                outputs.append(text.strip())

        return outputs