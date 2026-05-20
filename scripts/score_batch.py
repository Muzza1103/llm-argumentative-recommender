import argparse
import json
import time
from pathlib import Path

from src.argumentation.schema import build_arguments_from_parsed_json
from src.argumentation.scoring import (
    ScoreConfig,
    score_arguments,
    combine_scores,
)
from src.argumentation.aspect_mf_scorer import AspectMFScorer
from src.argumentation.mf_scorer import GlobalRatingFallbackMFScorer
from src.argumentation.llm_scorer import LocalLLMScorer, LLMScorerConfig
from src.llm.config import LLMConfig
from src.llm.loader import load_model_and_tokenizer
from src.llm.generator import LocalLLMGenerator
from src.llm.gemini_generator import GeminiGenerator, SCORING_RESPONSE_SCHEMA


class DisabledLLMScorer:
    """
    LLM scorer used when llm_weight=0.

    It avoids loading/calling any LLM while keeping the same scoring pipeline.
    """

    def score(self, argument):
        return 0.0


def load_jsonl(jsonl_path: Path) -> list[dict]:
    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def save_jsonl(records: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dataset_index(dataset_path: Path) -> dict[int, dict]:
    dataset_by_index = {}

    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            dataset_by_index[i] = json.loads(line)

    return dataset_by_index


def build_summary(
    scored_records: list[dict],
    input_file: str,
    output_file: str,
    dataset_file: str,
    llm_model: str,
    llm_backend: str,
    llm_weight: float,
    mf_weight: float,
    mf_source: str,
    skipped_records: int,
    only_valid: bool,
) -> dict:
    total_records = len(scored_records)

    total_arguments = 0
    llm_scores = []
    mf_scores = []
    combined_scores = []
    support_combined_scores = []
    attack_combined_scores = []

    for record in scored_records:
        for argument in record.get("scored_arguments", []):
            total_arguments += 1

            llm_score = argument.get("llm_score")
            mf_score = argument.get("mf_score")
            combined_score = argument.get("combined_score")
            arg_type = argument.get("arg_type")

            if isinstance(llm_score, (int, float)):
                llm_scores.append(float(llm_score))

            if isinstance(mf_score, (int, float)):
                mf_scores.append(float(mf_score))

            if isinstance(combined_score, (int, float)):
                combined_scores.append(float(combined_score))

                if arg_type == "support":
                    support_combined_scores.append(float(combined_score))
                elif arg_type == "attack":
                    attack_combined_scores.append(float(combined_score))

    def safe_mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "dataset_file": dataset_file,
        "input_file": input_file,
        "output_file": output_file,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
        "model_name": llm_model,
        "llm_weight": llm_weight,
        "mf_weight": mf_weight,
        "mf_source": mf_source,
        "only_valid": only_valid,
        "num_records_scored": total_records,
        "num_records_skipped": skipped_records,
        "num_arguments_scored": total_arguments,
        "mean_llm_score": safe_mean(llm_scores),
        "mean_mf_score": safe_mean(mf_scores),
        "mean_combined_score": safe_mean(combined_scores),
        "mean_support_combined_score": safe_mean(support_combined_scores),
        "mean_attack_combined_score": safe_mean(attack_combined_scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score generated arguments in batch using LLM + aspect-based MF scoring."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--backend", choices=["local", "gemini"], default="local")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--gcp-project", type=str, default=None)
    parser.add_argument("--gcp-location", type=str, default="global")

    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--llm-weight", type=float, default=0.5)
    parser.add_argument("--mf-weight", type=float, default=0.5)
    parser.add_argument(
        "--mf-predictions",
        type=str,
        default=None,
        help="Optional path to aspect-based MF predictions JSON.",
    )
    parser.add_argument("--only-valid", action="store_true")
    parser.add_argument("--save-llm-prompt", action="store_true")
    parser.add_argument("--save-llm-raw", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of Gemini scoring prompts sent in parallel.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")

    dataset_by_index = load_dataset_index(dataset_path)
    records = load_jsonl(input_path)

    if args.llm_weight > 0.0:
        if args.backend == "local":
            llm_config = LLMConfig(
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=False,
            )

            tokenizer, model = load_model_and_tokenizer(llm_config)

            generator = LocalLLMGenerator(
                model=model,
                tokenizer=tokenizer,
                config=llm_config,
            )

            llm_model_name = args.model
            llm_backend = "local"

        else:
            generator = GeminiGenerator(
                model_name=args.gemini_model,
                project=args.gcp_project,
                location=args.gcp_location,
                temperature=0.0,
                max_output_tokens=args.max_new_tokens,
                response_schema=SCORING_RESPONSE_SCHEMA,
                debug=True,
            )

            llm_model_name = args.gemini_model
            llm_backend = "gemini"

        llm_scorer = LocalLLMScorer(
            generator=generator,
            config=LLMScorerConfig(default_score=0.5),
            use_gemini_prompt=(llm_backend == "gemini"),
        )

    else:
        llm_scorer = DisabledLLMScorer()
        llm_model_name = "disabled"
        llm_backend = "disabled"

    mf_source = (
        f"aspect_mf:{args.mf_predictions}"
        if args.mf_predictions is not None
        else "global_rating_fallback"
    )

    if args.mf_predictions is not None:
        mf_scorer = AspectMFScorer(
            predictions_path=args.mf_predictions,
            user_id="",
            default_score=0.5,
        )
    else:
        mf_scorer = GlobalRatingFallbackMFScorer()

    score_config = ScoreConfig(
        llm_weight=args.llm_weight,
        mf_weight=args.mf_weight,
    )

    scored_records = []
    skipped_records = 0

    print(f"Loaded {len(records)} generated records from {input_path}")
    print(f"MF source: {mf_source}")
    print(f"LLM backend: {llm_backend}")
    print(f"LLM model: {llm_model_name}")

    start_time = time.perf_counter()

    if (
        args.llm_weight > 0.0
        and llm_backend == "gemini"
        and hasattr(llm_scorer, "score_batches")
    ):
        jobs = []

        for i, record in enumerate(records, start=1):
            validation = record.get("validation", {})
            is_valid = validation.get("is_valid", False)

            if args.only_valid and not is_valid:
                skipped_records += 1
                continue

            parsed_json = record.get("parsed_json")
            if parsed_json is None:
                skipped_records += 1
                continue

            dataset_index = record.get("index")
            if dataset_index not in dataset_by_index:
                skipped_records += 1
                continue

            example = dataset_by_index[dataset_index]
            arguments = build_arguments_from_parsed_json(parsed_json, example)

            jobs.append(
                {
                    "position": i,
                    "record": record,
                    "example": example,
                    "arguments": arguments,
                }
            )

        argument_batches = [
            job["arguments"]
            for job in jobs
        ]

        llm_score_batches = llm_scorer.score_batches(
            argument_batches,
            batch_size=args.batch_size,
        )

        for job, llm_scores in zip(jobs, llm_score_batches):
            record = job["record"]
            example = job["example"]
            arguments = job["arguments"]

            if args.mf_predictions is not None:
                mf_scorer.set_user(example.get("user_id"))

            scored_arguments_dicts = []

            for argument, llm_score in zip(arguments, llm_scores):
                mf_score = mf_scorer.score(argument)
                combined_score = combine_scores(
                    llm_score=llm_score,
                    mf_score=mf_score,
                    config=score_config,
                )

                argument.llm_score = llm_score
                argument.mf_score = mf_score
                argument.combined_score = combined_score

                argument_dict = argument.to_dict()

                if not args.save_llm_prompt:
                    argument_dict.pop("llm_scoring_prompt", None)

                if not args.save_llm_raw:
                    argument_dict.pop("llm_scoring_raw_output", None)

                scored_arguments_dicts.append(argument_dict)

            enriched_record = dict(record)
            enriched_record["scoring"] = {
                "llm_backend": llm_backend,
                "llm_model": llm_model_name,
                "llm_weight": args.llm_weight,
                "mf_weight": args.mf_weight,
                "mf_source": mf_source,
            }
            enriched_record["scored_arguments"] = scored_arguments_dicts

            scored_records.append(enriched_record)

            print(
                f"[{job['position']}/{len(records)}] "
                f"index={record.get('index')} "
                f"target={record.get('target_name')} "
                f"scored_arguments={len(scored_arguments_dicts)}"
            )

    else:
        for i, record in enumerate(records, start=1):
            validation = record.get("validation", {})
            is_valid = validation.get("is_valid", False)

            if args.only_valid and not is_valid:
                skipped_records += 1
                continue

            parsed_json = record.get("parsed_json")
            if parsed_json is None:
                skipped_records += 1
                continue

            dataset_index = record.get("index")
            if dataset_index not in dataset_by_index:
                skipped_records += 1
                continue

            example = dataset_by_index[dataset_index]

            if args.mf_predictions is not None:
                mf_scorer.set_user(example.get("user_id"))

            arguments = build_arguments_from_parsed_json(parsed_json, example)

            scored_arguments = score_arguments(
                arguments=arguments,
                llm_scorer=llm_scorer,
                mf_scorer=mf_scorer,
                config=score_config,
            )

            scored_arguments_dicts = []

            for argument in scored_arguments:
                argument_dict = argument.to_dict()

                if args.llm_weight == 0.0:
                    argument_dict["llm_score_reason"] = (
                        "LLM scoring disabled because llm_weight=0."
                    )

                if not args.save_llm_prompt:
                    argument_dict.pop("llm_scoring_prompt", None)

                if not args.save_llm_raw:
                    argument_dict.pop("llm_scoring_raw_output", None)

                scored_arguments_dicts.append(argument_dict)

            enriched_record = dict(record)
            enriched_record["scoring"] = {
                "llm_backend": llm_backend,
                "llm_model": llm_model_name,
                "llm_weight": args.llm_weight,
                "mf_weight": args.mf_weight,
                "mf_source": mf_source,
            }
            enriched_record["scored_arguments"] = scored_arguments_dicts

            scored_records.append(enriched_record)

            print(
                f"[{i}/{len(records)}] "
                f"index={record.get('index')} "
                f"target={record.get('target_name')} "
                f"scored_arguments={len(scored_arguments_dicts)}"
            )

    elapsed_seconds = time.perf_counter() - start_time
    elapsed_minutes = elapsed_seconds / 60

    save_jsonl(scored_records, output_path)

    summary = build_summary(
        scored_records=scored_records,
        input_file=str(input_path),
        output_file=str(output_path),
        dataset_file=str(dataset_path),
        llm_model=llm_model_name,
        llm_backend=llm_backend,
        llm_weight=args.llm_weight,
        mf_weight=args.mf_weight,
        mf_source=mf_source,
        skipped_records=skipped_records,
        only_valid=args.only_valid,
    )

    summary["elapsed_seconds"] = elapsed_seconds
    summary["elapsed_minutes"] = elapsed_minutes
    summary["batch_size"] = args.batch_size

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nScored output:  {output_path}")
    print(f"Summary:        {summary_path}")
    print(f"Scored records: {len(scored_records)}")
    print(f"Skipped:        {skipped_records}")
    print(f"Elapsed time:   {elapsed_seconds:.2f}s ({elapsed_minutes:.2f} min)")


if __name__ == "__main__":
    main()