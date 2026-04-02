from __future__ import annotations

from collections import Counter
from typing import Any


def make_error(code: str, message: str) -> dict[str, str]:
    return {
        "code": code,
        "message": message,
    }


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _extract_known_item_names(example: dict) -> set[str]:
    names = set()

    for item in example.get("history", []):
        name = item.get("name")
        if name:
            names.add(name)

    target_name = example.get("target_item", {}).get("name")
    if target_name:
        names.add(target_name)

    return names


def validate_generated_arguments(example: dict, parsed_json: Any) -> dict[str, Any]:
    errors: list[dict[str, str]] = []

    if parsed_json is None:
        return {
            "is_valid": False,
            "errors": [
                make_error("invalid_json", "Parsed JSON is None."),
            ],
            "error_counts": {
                "invalid_json": 1,
            },
        }

    if not isinstance(parsed_json, dict):
        return {
            "is_valid": False,
            "errors": [
                make_error("invalid_root_object", "Parsed output is not a JSON object."),
            ],
            "error_counts": {
                "invalid_root_object": 1,
            },
        }

    arguments = parsed_json.get("arguments")
    if not isinstance(arguments, list):
        return {
            "is_valid": False,
            "errors": [
                make_error("missing_arguments_list", "'arguments' field is missing or is not a list."),
            ],
            "error_counts": {
                "missing_arguments_list": 1,
            },
        }

    if len(arguments) != 4:
        errors.append(
            make_error(
                "wrong_argument_count",
                f"Expected 4 arguments, got {len(arguments)}.",
            )
        )

    known_item_names = _extract_known_item_names(example)

    ids: list[str] = []
    support_count = 0
    attack_count = 0

    for idx, argument in enumerate(arguments):
        prefix = f"Argument {idx}"

        if not isinstance(argument, dict):
            errors.append(
                make_error(
                    "invalid_argument_object",
                    f"{prefix} is not an object.",
                )
            )
            continue

        for required_key in ["id", "type", "text", "evidence"]:
            if required_key not in argument:
                errors.append(
                    make_error(
                        "missing_argument_key",
                        f"{prefix} is missing key '{required_key}'.",
                    )
                )

        arg_id = argument.get("id")
        arg_type = argument.get("type")
        text = argument.get("text")
        evidence = argument.get("evidence")

        if _is_non_empty_string(arg_id):
            ids.append(arg_id)
        else:
            errors.append(
                make_error(
                    "invalid_argument_id",
                    f"{prefix} has an invalid 'id'.",
                )
            )

        if arg_type not in {"support", "attack"}:
            errors.append(
                make_error(
                    "invalid_argument_type",
                    f"{prefix} has invalid type '{arg_type}'.",
                )
            )
        elif arg_type == "support":
            support_count += 1
        elif arg_type == "attack":
            attack_count += 1

        if not _is_non_empty_string(text):
            errors.append(
                make_error(
                    "invalid_argument_text",
                    f"{prefix} has an empty or invalid 'text'.",
                )
            )

        if not isinstance(evidence, list):
            errors.append(
                make_error(
                    "invalid_evidence_list",
                    f"{prefix} has invalid 'evidence' (must be a list).",
                )
            )
            continue

        if len(evidence) == 0:
            errors.append(
                make_error(
                    "empty_evidence_list",
                    f"{prefix} has empty evidence.",
                )
            )

        if len(evidence) > 2:
            errors.append(
                make_error(
                    "too_many_evidence_items",
                    f"{prefix} has more than 2 evidence strings.",
                )
            )

        for ev_idx, ev in enumerate(evidence):
            ev_prefix = f"{prefix}, evidence {ev_idx}"

            if not _is_non_empty_string(ev):
                errors.append(
                    make_error(
                        "invalid_evidence_string",
                        f"{ev_prefix} is empty or invalid.",
                    )
                )
                continue

            if " | " not in ev:
                errors.append(
                    make_error(
                        "invalid_evidence_format",
                        f"{ev_prefix} does not follow the expected 'item_name | fact' format.",
                    )
                )
                continue

            item_name = ev.split(" | ", 1)[0].strip()

            if item_name not in known_item_names:
                errors.append(
                    make_error(
                        "unknown_item_reference",
                        f"{ev_prefix} references unknown item '{item_name}'.",
                    )
                )

    if len(set(ids)) != len(ids):
        errors.append(
            make_error(
                "duplicate_argument_ids",
                "Argument ids are not unique.",
            )
        )

    if support_count != 2:
        errors.append(
            make_error(
                "wrong_support_count",
                f"Expected 2 support arguments, got {support_count}.",
            )
        )

    if attack_count != 2:
        errors.append(
            make_error(
                "wrong_attack_count",
                f"Expected 2 attack arguments, got {attack_count}.",
            )
        )

    error_counter = Counter(error["code"] for error in errors)

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "error_counts": dict(error_counter),
    }