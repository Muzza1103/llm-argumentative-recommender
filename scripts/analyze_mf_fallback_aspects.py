import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze argument aspects that fall back to MF score 0.5."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to scored arguments JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON analysis file.",
    )
    parser.add_argument(
        "--fallback-score",
        type=float,
        default=0.5,
        help="MF fallback score value.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for comparing float scores.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_jsonl(input_path)

    aspect_counter = Counter()
    argument_counter = 0
    fallback_argument_counter = 0
    fallback_examples = []

    for record in records:
        for argument in record.get("scored_arguments", []):
            argument_counter += 1

            mf_score = argument.get("mf_score")
            if not isinstance(mf_score, (int, float)):
                continue

            if abs(float(mf_score) - args.fallback_score) > args.tolerance:
                continue

            fallback_argument_counter += 1

            used_aspects = argument.get("used_aspects", [])
            if not isinstance(used_aspects, list):
                used_aspects = []

            cleaned_aspects = []
            for aspect in used_aspects:
                if not isinstance(aspect, str):
                    continue

                aspect = aspect.strip().lower()
                if not aspect:
                    continue

                cleaned_aspects.append(aspect)
                aspect_counter[aspect] += 1

            fallback_examples.append(
                {
                    "dataset_index": record.get("index"),
                    "user_id": record.get("user_id"),
                    "target_name": record.get("target_name"),
                    "argument_id": argument.get("id"),
                    "arg_type": argument.get("arg_type"),
                    "text": argument.get("text"),
                    "used_aspects": cleaned_aspects,
                    "mf_score": mf_score,
                    "llm_score": argument.get("llm_score"),
                    "combined_score": argument.get("combined_score"),
                }
            )

    output = {
        "input_file": str(input_path),
        "fallback_score": args.fallback_score,
        "num_records": len(records),
        "num_arguments": argument_counter,
        "num_fallback_arguments": fallback_argument_counter,
        "fallback_ratio": (
            fallback_argument_counter / argument_counter
            if argument_counter
            else 0.0
        ),
        "fallback_aspect_counts": dict(aspect_counter.most_common()),
        "fallback_examples": fallback_examples,
    }

    save_json(output, output_path)

    print("Done.")
    print(f"Arguments:          {argument_counter}")
    print(f"Fallback arguments: {fallback_argument_counter}")
    print(f"Fallback ratio:     {output['fallback_ratio']:.3f}")
    print(f"Output:             {output_path}")

    if aspect_counter:
        print("\nMost common fallback aspects:")
        for aspect, count in aspect_counter.most_common(20):
            print(f"- {aspect}: {count}")


if __name__ == "__main__":
    main()