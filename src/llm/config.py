from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 650
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    device_map: str = "auto"
    torch_dtype: str = "auto"