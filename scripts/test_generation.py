import argparse
import json
import random
from pathlib import Path

from src.llm.config import LLMConfig
from src.llm.loader import load_model_and_tokenizer
from src.llm.generator import LocalLLMGenerator
from src.llm.utils import extract_first_json_object

from src.prompting.argument_prompt import build_prompt
from src.prompting.formatters import format_history, format_target_item


def load_example(jsonl_path: Path, index: int):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line), i

    raise IndexError(f"Index {index} out of range.")


def load_random_example(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("Input JSONL file is empty.")

    index = random.randrange(len(lines))
    return json.loads(lines[index]), index


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured arguments with a local Hugging Face LLM."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the example to load from the JSONL file.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random example.",
    )
    args = parser.parse_args()

    if args.random:
        example, used_index = load_random_example(Path(args.input))
    else:
        example, used_index = load_example(Path(args.input), args.index)

    history_str = format_history(example["history"])
    target_str = format_target_item(example["target_item"])
    prompt = build_prompt(history_str, target_str)

    config = LLMConfig(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    tokenizer, model = load_model_and_tokenizer(config)
    generator = LocalLLMGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    output_text = generator.generate(prompt)
    parsed_json = extract_first_json_object(output_text)

    print(f"\nUsing example index: {used_index}")
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("RAW OUTPUT")
    print("=" * 80)
    print(output_text)

    print("\n" + "=" * 80)
    print("PARSED JSON")
    print("=" * 80)
    if parsed_json is None:
        print("Could not parse JSON.")
    else:
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()