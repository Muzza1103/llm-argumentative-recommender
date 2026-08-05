from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from src.hotel import (
    HotelDataValidationError,
    HotelPreferenceValidationError,
    PreferenceInterpreter,
    evaluate_hotel_by_id,
    interpret_session_preferences,
    load_hotel_profiles,
    load_session_preferences,
)


def _load_interpreter(specification: str) -> PreferenceInterpreter:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "interpreter factory must use the form 'module:attribute'"
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

    if not isinstance(candidate, PreferenceInterpreter):
        raise TypeError(
            "configured interpreter factory did not provide an object with "
            "interpret(text)"
        )
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one hotel against explicit session preferences with "
            "traceable arguments and DF-QuAD."
        )
    )
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--hotel-id", required=True)
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
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.preference_text is not None and not args.interpreter_factory:
        parser.error(
            "--preference-text requires an explicitly configured "
            "--interpreter-factory; use --preferences for deterministic "
            "execution without an LLM"
        )
    if args.preferences is not None and args.interpreter_factory:
        parser.error("--interpreter-factory is only valid with --preference-text")

    try:
        dataset = load_hotel_profiles(args.profiles)
        if args.preferences is not None:
            preferences = load_session_preferences(args.preferences)
        else:
            interpreter = _load_interpreter(args.interpreter_factory)
            preferences = interpret_session_preferences(
                args.preference_text,
                interpreter,
            )
        result = evaluate_hotel_by_id(dataset, args.hotel_id, preferences)
    except (
        HotelDataValidationError,
        HotelPreferenceValidationError,
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    rendered = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
