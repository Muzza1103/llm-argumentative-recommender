import argparse
import json
import random
import time

from collections import Counter, deque
from pathlib import Path

from src.prompting.gemini_argument_prompt import build_gemini_prompt
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



def load_existing_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_partial_outputs(
    all_records: list[dict],
    valid_records: list[dict],
    invalid_records: list[dict],
    all_output_path: Path,
    valid_output_path: Path,
    invalid_output_path: Path,
):
    save_jsonl(all_records, all_output_path)
    save_jsonl(valid_records, valid_output_path)
    save_jsonl(invalid_records, invalid_output_path)


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "rate" in text
    )


def wait_for_rate_limit(
    request_timestamps: deque,
    requests_to_send: int,
    requests_per_minute: int,
):
    if requests_per_minute <= 0:
        return

    now = time.monotonic()

    while request_timestamps and now - request_timestamps[0] >= 60:
        request_timestamps.popleft()

    while len(request_timestamps) + requests_to_send > requests_per_minute:
        sleep_for = 60 - (now - request_timestamps[0]) + 0.5
        sleep_for = max(sleep_for, 1.0)
        print(f"Rate limit reached. Sleeping {sleep_for:.1f}s...")
        time.sleep(sleep_for)

        now = time.monotonic()
        while request_timestamps and now - request_timestamps[0] >= 60:
            request_timestamps.popleft()


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
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts generated together on GPU.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "gemini"], 
        default="local")
    parser.add_argument(
        "--gemini-model", 
        type=str, 
        default="gemini-2.5-flash")
    parser.add_argument(
        "--gcp-project", 
        type=str, 
        default=None)
    parser.add_argument(
        "--gcp-location", 
        type=str, 
        default="global")
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Include the full prompt in the output files.",
    )
    parser.add_argument(
        "--argument-mode",
        choices=["balanced", "unbalanced"],
        default="balanced",
        help="Argument generation mode: balanced = 2 supports/2 attacks, unbalanced = 4 arguments with free support/attack split.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output files and skip already processed example indices.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=0,
        help="Maximum number of Gemini requests per minute. Use 0 to disable throttling.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum number of retries after a rate-limit error.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=60.0,
        help="Initial waiting time after a rate-limit error.",
    )
    args = parser.parse_args()

    from src.llm.config import LLMConfig
    from src.llm.utils import extract_first_json_object
    from src.llm.validation import validate_generated_arguments

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

    if args.backend == "local":
        from src.llm.generator import LocalLLMGenerator
        from src.llm.loader import load_model_and_tokenizer

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

    else:
        from src.llm.gemini_generator import (
            ARGUMENT_RESPONSE_SCHEMA,
            GeminiGenerator,
        )

        generator = GeminiGenerator(
            model_name=args.gemini_model,
            project=args.gcp_project,
            location=args.gcp_location,
            temperature=args.temperature if args.do_sample else 0.0,
            max_output_tokens=args.max_new_tokens,
            response_schema=ARGUMENT_RESPONSE_SCHEMA,
            debug=True,
        )

    all_records = []
    valid_records = []
    invalid_records = []
    global_error_counter = Counter()

    if args.resume:
        all_records = load_existing_jsonl(all_output_path)
        valid_records = load_existing_jsonl(valid_output_path)
        invalid_records = load_existing_jsonl(invalid_output_path)
        print(f"Resume enabled: loaded {len(all_records)} existing records.")

    print(f"Loaded {len(examples)} examples.")
    print(f"Processing {len(indices)} examples...")
    print(f"Batch size: {args.batch_size}")
    print(f"Backend: {args.backend}")
    if args.backend == "local":
        print(f"Model: {args.model}")
    else:
        print(f"Gemini model: {args.gemini_model}")

    prompts = []
    selected_examples = []

    start_time = time.perf_counter()

    for idx in indices:
        example = examples[idx]

        history_str = format_history(example["history"])
        target_str = format_target_item(example["target_item"])

        if args.backend == "gemini":
            prompt = build_gemini_prompt(
                history_str,
                target_str,
                argument_mode=args.argument_mode,
            )
        else:
            prompt = build_prompt(
                history_str,
                target_str,
                argument_mode=args.argument_mode,
            )

        prompts.append(prompt)
        selected_examples.append(
            {
                "index": idx,
                "example": example,
                "prompt": prompt,
            }
        )

    processed_indices = {
        record.get("index")
        for record in all_records
        if isinstance(record.get("index"), int)
    }

    pending_jobs = [
        job
        for job in selected_examples
        if job["index"] not in processed_indices
    ]

    print(f"Already processed: {len(processed_indices)}")
    print(f"Pending examples:  {len(pending_jobs)}")

    request_timestamps = deque()

    for batch_start in range(0, len(pending_jobs), args.batch_size):
        batch_jobs = pending_jobs[batch_start: batch_start + args.batch_size]
        batch_prompts = [job["prompt"] for job in batch_jobs]

        attempt = 0

        while True:
            wait_for_rate_limit(
                request_timestamps=request_timestamps,
                requests_to_send=len(batch_prompts),
                requests_per_minute=args.requests_per_minute,
            )

            try:
                output_texts = generator.generate_batch(
                    batch_prompts,
                    batch_size=args.batch_size,
                )

                now = time.monotonic()
                for _ in batch_prompts:
                    request_timestamps.append(now)

                break

            except Exception as error:
                attempt += 1

                if not is_rate_limit_error(error) or attempt > args.max_retries:
                    save_partial_outputs(
                        all_records=all_records,
                        valid_records=valid_records,
                        invalid_records=invalid_records,
                        all_output_path=all_output_path,
                        valid_output_path=valid_output_path,
                        invalid_output_path=invalid_output_path,
                    )
                    raise

                wait_time = args.retry_wait_seconds * attempt
                print(
                    f"Rate-limit error on batch starting at {batch_start}. "
                    f"Retry {attempt}/{args.max_retries} after {wait_time:.1f}s."
                )
                time.sleep(wait_time)

        for job, output_text in zip(batch_jobs, output_texts):
            idx = job["index"]
            example = job["example"]
            prompt = job["prompt"]

            parsed_json = extract_first_json_object(output_text)
            validation = validate_generated_arguments(
                example,
                parsed_json,
                argument_mode=args.argument_mode,
            )

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
                f"[{len(all_records)}/{len(indices)}] "
                f"index={idx} "
                f"target={record['target_name']} "
                f"status={status}"
            )

        save_partial_outputs(
            all_records=all_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            all_output_path=all_output_path,
            valid_output_path=valid_output_path,
            invalid_output_path=invalid_output_path,
        )

    save_jsonl(all_records, all_output_path)
    save_jsonl(valid_records, valid_output_path)
    save_jsonl(invalid_records, invalid_output_path)

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60

    summary = {
        "input_file": str(input_path),
        "output_file_all": str(all_output_path),
        "output_file_valid": str(valid_output_path),
        "output_file_invalid": str(invalid_output_path),

        "backend": args.backend,

        "model_name": (
            args.model
            if args.backend == "local"
            else args.gemini_model
        ),

        "num_examples_requested": len(indices),
        "num_examples_processed": len(all_records),
        "batch_size": args.batch_size,
        "num_valid": len(valid_records),
        "num_invalid": len(invalid_records),
        "error_counts": dict(global_error_counter),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_minutes,
        "resume": args.resume,
        "requests_per_minute": args.requests_per_minute,
        "max_retries": args.max_retries,
        "retry_wait_seconds": args.retry_wait_seconds,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll results:     {all_output_path}")
    print(f"Valid results:   {valid_output_path}")
    print(f"Invalid results: {invalid_output_path}")
    print(f"Elapsed time:    {elapsed_seconds:.2f}s ({elapsed_minutes:.2f} min)")
    print(f"Summary:         {summary_path}")
    print(f"Valid outputs:   {len(valid_records)}")
    print(f"Invalid outputs: {len(invalid_records)}")

    if global_error_counter:
        print("\nError counts:")
        for error_code, count in global_error_counter.most_common():
            print(f"- {error_code}: {count}")


if __name__ == "__main__":
    main()
