import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.llm.config import LLMConfig
from src.llm.loader import load_model_and_tokenizer
from src.llm.generator import LocalLLMGenerator
from src.llm.utils import extract_first_json_object
from src.llm.validation import validate_generated_arguments

from src.prompting.argument_prompt import build_prompt
from src.prompting.formatters import format_history, format_target_item


def load_all_examples(jsonl_path: Path) -> list[dict]:
    examples = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    return examples


def select_indices(
    total_examples: int,
    start_index: int,
    num_examples: int,
    random_mode: bool,
    seed: int | None,
) -> list[int]:
    if random_mode:
        if seed is not None:
            random.seed(seed)

        if num_examples > total_examples:
            raise ValueError(
                f"Requested {num_examples} random examples, but dataset only has {total_examples}."
            )

        return random.sample(range(total_examples), num_examples)

    end_index = min(start_index + num_examples, total_examples)
    return list(range(start_index, end_index))


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_output_paths(output_path: Path) -> tuple[Path, Path, Path, Path]:
    stem = output_path.stem
    suffix = output_path.suffix or ".jsonl"
    parent = output_path.parent

    all_path = output_path
    valid_path = parent / f"{stem}_valid{suffix}"
    invalid_path = parent / f"{stem}_invalid{suffix}"
    summary_path = parent / f"{stem}_summary.json"

    return all_path, valid_path, invalid_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate arguments on multiple JSONL examples with validation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the main output JSONL file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for sequential generation.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to process.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample examples randomly instead of using a sequential range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when --random is enabled.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=650,
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
        "--save-prompt",
        action="store_true",
        help="Include the full prompt in the output files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    all_output_path, valid_output_path, invalid_output_path, summary_path = build_output_paths(output_path)

    examples = load_all_examples(input_path)
    if not examples:
        raise ValueError("Input JSONL file is empty.")

    indices = select_indices(
        total_examples=len(examples),
        start_index=args.start_index,
        num_examples=args.num_examples,
        random_mode=args.random,
        seed=args.seed,
    )

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

    all_records = []
    valid_records = []
    invalid_records = []
    global_error_counter = Counter()

    print(f"Loaded {len(examples)} examples.")
    print(f"Processing {len(indices)} examples...")

    for position, idx in enumerate(indices, start=1):
        example = examples[idx]

        history_str = format_history(example["history"])
        target_str = format_target_item(example["target_item"])
        prompt = build_prompt(history_str, target_str)

        output_text = generator.generate(prompt)
        parsed_json = extract_first_json_object(output_text)
        validation = validate_generated_arguments(example, parsed_json)

        for error in validation["errors"]:
            global_error_counter[error["code"]] += 1

        record = {
            "index": idx,
            "user_id": example.get("user_id"),
            "target_name": example.get("target_item", {}).get("name"),
            "raw_output": output_text,
            "parsed_json": parsed_json,
            "validation": validation,
        }

        if args.save_prompt:
            record["prompt"] = prompt

        all_records.append(record)

        if validation["is_valid"]:
            valid_records.append(record)
            status = "VALID"
        else:
            invalid_records.append(record)
            status = "INVALID"

        print(
            f"[{position}/{len(indices)}] "
            f"index={idx} "
            f"target={record['target_name']} "
            f"status={status}"
        )

    save_jsonl(all_records, all_output_path)
    save_jsonl(valid_records, valid_output_path)
    save_jsonl(invalid_records, invalid_output_path)

    summary = {
        "input_file": str(input_path),
        "output_file_all": str(all_output_path),
        "output_file_valid": str(valid_output_path),
        "output_file_invalid": str(invalid_output_path),
        "num_examples_requested": len(indices),
        "num_examples_processed": len(all_records),
        "num_valid": len(valid_records),
        "num_invalid": len(invalid_records),
        "error_counts": dict(global_error_counter),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"All results:     {all_output_path}")
    print(f"Valid results:   {valid_output_path}")
    print(f"Invalid results: {invalid_output_path}")
    print(f"Summary:         {summary_path}")
    print(f"Valid outputs:   {len(valid_records)}")
    print(f"Invalid outputs: {len(invalid_records)}")

    if global_error_counter:
        print("\nError counts:")
        for error_code, count in global_error_counter.most_common():
            print(f"- {error_code}: {count}")


if __name__ == "__main__":
    main()