from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LLMConfig


def load_model_and_tokenizer(config: LLMConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        trust_remote_code=True,
    )

    return tokenizer, model