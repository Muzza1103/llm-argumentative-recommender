from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.hotel import (
    ARGUMENT_MODES,
    DEFAULT_FACILITY_ONTOLOGY_PATH,
    FacilityOntology,
    GeminiHybridArgumentGenerator,
    GeminiPreferenceInterpreter,
    HotelDataValidationError,
    HotelGeminiError,
    HotelHybridValidationError,
    HotelPreferenceValidationError,
    HybridArgumentGenerator,
    PreferenceInterpreter,
    evaluate_hotel_by_id,
    interpret_session_preferences,
    load_hotel_profiles,
    load_session_preferences,
)


def _load_component(specification: str, protocol: type, label: str):
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"{label} factory must use the form 'module:attribute'"
        )
    module = importlib.import_module(module_name)
    target: Any = getattr(module, attribute_name)

    if isinstance(target, type):
        candidate = target()
    elif isinstance(target, PreferenceInterpreter):
        candidate = target
    elif callable(target):
        candidate = target()
    else:
        candidate = target

    if not isinstance(candidate, protocol):
        raise TypeError(
            f"configured {label} factory returned an incompatible object"
        )
    return candidate


def _load_interpreter(specification: str) -> PreferenceInterpreter:
    return _load_component(
        specification,
        PreferenceInterpreter,
        "interpreter",
    )


def _load_hybrid_generator(
    specification: str,
) -> HybridArgumentGenerator:
    return _load_component(
        specification,
        HybridArgumentGenerator,
        "hybrid generator",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one hotel against explicit session preferences with "
            "traceable arguments and DF-QuAD."
        )
    )
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--hotel-id", required=True)
    parser.add_argument(
        "--argument-mode",
        choices=sorted(ARGUMENT_MODES),
        default="baseline",
        help="Keep baseline as the default; hybrid validates Gemini proposals.",
    )
    preference_group = parser.add_mutually_exclusive_group(required=True)
    preference_group.add_argument("--preferences")
    preference_group.add_argument("--preference-text")
    parser.add_argument(
        "--interpreter-factory",
        help=(
            "Explicit dependency-injection hook for --preference-text, using "
            "'module:attribute'. The attribute must be an interpreter or a "
            "zero-argument factory."
        ),
    )
    parser.add_argument(
        "--hybrid-generator-factory",
        help=(
            "Offline injection hook for hybrid mode, using "
            "'module:attribute'. Without it, hybrid mode uses Gemini."
        ),
    )
    parser.add_argument(
        "--facility-ontology",
        default=str(DEFAULT_FACILITY_ONTOLOGY_PATH),
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")

    parser.add_argument(
        "--gcp-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help=(
            "Google Cloud project used by Vertex AI. Defaults to "
            "GOOGLE_CLOUD_PROJECT."
        ),
    )

    parser.add_argument(
        "--gcp-location",
        default=os.environ.get(
            "GOOGLE_CLOUD_LOCATION",
            "us-central1",
        ),
        help=(
            "Vertex AI location. Defaults to GOOGLE_CLOUD_LOCATION or "
            "us-central1."
        ),
    )

    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if (
        args.preference_text is not None
        and not args.interpreter_factory
        and args.argument_mode == "baseline"
    ):
        parser.error(
            "--preference-text requires an explicitly configured "
            "--interpreter-factory; use --preferences for deterministic "
            "execution without an LLM"
        )

    if args.preferences is not None and args.interpreter_factory:
        parser.error(
            "--interpreter-factory is only valid with --preference-text"
        )

    if (
        args.argument_mode == "baseline"
        and args.hybrid_generator_factory
    ):
        parser.error(
            "--hybrid-generator-factory is only valid in hybrid mode"
        )

    needs_gemini_interpreter = (
        args.preference_text is not None
        and not args.interpreter_factory
    )

    needs_gemini_generator = (
        args.argument_mode == "hybrid"
        and not args.hybrid_generator_factory
    )

    vertex_client = None

    if needs_gemini_interpreter or needs_gemini_generator:
        if not args.gcp_project:
            parser.error(
                "--gcp-project or GOOGLE_CLOUD_PROJECT is required "
                "for Vertex AI"
            )

        try:
            from google import genai
        except ModuleNotFoundError:
            parser.error("google-genai is required for Vertex AI")

        vertex_client = genai.Client(
            vertexai=True,
            project=args.gcp_project,
            location=args.gcp_location,
        )

    try:
        dataset = load_hotel_profiles(args.profiles)

        ontology = FacilityOntology.load(
            args.facility_ontology
        )

        if args.preferences is not None:
            preferences = load_session_preferences(
                args.preferences
            )
        else:
            interpreter = (
                _load_interpreter(args.interpreter_factory)
                if args.interpreter_factory
                else GeminiPreferenceInterpreter(
                    model_name=args.gemini_model,
                    ontology=ontology,
                    client=vertex_client,
                )
            )

            preferences = interpret_session_preferences(
                args.preference_text,
                interpreter,
            )

        hybrid_generator = None

        if args.argument_mode == "hybrid":
            hybrid_generator = (
                _load_hybrid_generator(
                    args.hybrid_generator_factory
                )
                if args.hybrid_generator_factory
                else GeminiHybridArgumentGenerator(
                    model_name=args.gemini_model,
                    client=vertex_client,
                )
            )

        result = evaluate_hotel_by_id(
            dataset,
            args.hotel_id,
            preferences,
            argument_mode=args.argument_mode,
            hybrid_generator=hybrid_generator,
            facility_ontology=ontology,
        )

    except (
        HotelDataValidationError,
        HotelPreferenceValidationError,
        HotelGeminiError,
        HotelHybridValidationError,
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    rendered = json.dumps(
        result.to_dict(),
        indent=2,
        ensure_ascii=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        rendered + "\n",
        encoding="utf-8",
    )

    print(rendered)


if __name__ == "__main__":
    main()