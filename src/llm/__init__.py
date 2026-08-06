from .config import LLMConfig
from .utils import extract_first_json_object

__all__ = [
    "LLMConfig",
    "load_model_and_tokenizer",
    "LocalLLMGenerator",
    "extract_first_json_object",
]


def __getattr__(name):
    """Keep optional model dependencies lazy for lightweight CLI imports."""
    if name == "load_model_and_tokenizer":
        from .loader import load_model_and_tokenizer

        return load_model_and_tokenizer
    if name == "LocalLLMGenerator":
        from .generator import LocalLLMGenerator

        return LocalLLMGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
