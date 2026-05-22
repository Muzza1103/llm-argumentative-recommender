import argparse
import json
from pathlib import Path

from src.llm.gemini_generator import GeminiGenerator


DIRECT_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["score", "reason"],
}


PROMPT_TEMPLATE = """
Predict how much the user would like the target restaurant.

Return ONLY a valid JSON object.

TASK:
Given:
- the user's restaurant history
- the target restaurant

Predict a score between 0.0 and 1.0:
- 0.0 means the user would strongly dislike the target
- 0.5 means uncertain or neutral preference
- 1.0 means the user would strongly like the target

RULES:
- Base the score only on the provided user history and target item.
- Do not generate arguments.
- Do not use the true target rating.
- The reason must be short and explicit.

OUTPUT FORMAT:
{{
  "score": 0.0,
  "reason": "short explanation"
}}

USER_HISTORY:
{history}

TARGET_ITEM:
{target}
""".strip()


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_history(history: list[dict]) -> str:
    lines = []

    for item in history:
        lines.append(
            json.dumps(
                {
                    "name": item.get("name"),
                    "categories": item.get("categories"),
                    "attributes": item.get("attributes"),
                    "user_stars": item.get("user_stars"),
                    "review_text": item.get("review_text"),
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines)


def compact_target(target: dict) -> str:
    safe_target = {
        "name": target.get("name"),
        "categories": target.get("categories"),
        "attributes": target.get("attributes"),
        "global_stars": target.get("global_stars"),
        "review_count": target.get("review_count"),
    }

    return json.dumps(safe_target, ensure_ascii=False)


def build_prompt(example: dict) -> str:
    return PROMPT_TEMPLATE.format(
        history=compact_history(example.get("history", [])),
        target=compact_target(example.get("target_item", {})),
    )


def parse_score(output_text: str) -> tuple[float | None, str | None]:
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return None, None

    score = parsed.get("score")
    reason = parsed.get("reason")

    if not isinstance(score, (int, float)):
        return None, reason if isinstance(reason, str) else None

    score = max(0.0, min(1.0, float(score)))

    return score, reason if isinstance(reason, str) else None


def main():
    parser = argparse.ArgumentParser(
        description="LLM-only direct rating baseline."
    )

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--gcp-project", type=str, default=None)
    parser.add_argument("--gcp-location", type=str, default="global")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-examples", type=int, default=None)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    examples = load_jsonl(input_path)

    if args.num_examples is not None:
        examples = examples[:args.num_examples]

    generator = GeminiGenerator(
        model_name=args.gemini_model,
        project=args.gcp_project,
        location=args.gcp_location,
        temperature=0.0,
        max_output_tokens=args.max_new_tokens,
        response_schema=DIRECT_SCORE_SCHEMA,
        debug=True,
    )

    prompts = [build_prompt(example) for example in examples]

    output_texts = generator.generate_batch(
        prompts,
        batch_size=args.batch_size,
    )

    records = []
    valid = 0
    invalid = 0

    for index, (example, output_text) in enumerate(zip(examples, output_texts)):
        score, reason = parse_score(output_text)

        if score is None:
            invalid += 1
        else:
            valid += 1

        target_item = example.get("target_item", {})

        records.append(
            {
                "index": index,
                "user_id": example.get("user_id"),
                "target_name": target_item.get("name"),
                "business_id": target_item.get("business_id"),
                "score": score,
                "reason": reason,
                "raw_output": output_text,
                "is_valid": score is not None,
            }
        )

    save_jsonl(records, output_path)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "model": args.gemini_model,
        "num_examples": len(examples),
        "valid_outputs": valid,
        "invalid_outputs": invalid,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Output:  {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Valid:   {valid}")
    print(f"Invalid: {invalid}")


if __name__ == "__main__":
    main()