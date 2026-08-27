from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.hotel.render_hotel_graph import render_hotel_graph
from src.hotel import (
    DEFAULT_FACILITY_ONTOLOGY_PATH,
    HOTEL_ASPECT_SET,
    FacilityOntology,
    evaluate_hotel_by_id,
    load_hotel_profiles,
    session_preferences_from_dict,
)


class ReplayHybridArgumentGenerator:
    #Réutilise les propositions existantes sans appeler de nouveau le LLM.

    def __init__(self, proposed_arguments: list[Any]) -> None:
        self._proposed_arguments = deepcopy(proposed_arguments)
        self.last_trace = {
            "mode": "replay_existing_proposals",
            "request_count": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0,
        }

    def propose_arguments(self, **_: Any) -> dict[str, Any]:
        return {
            "arguments": deepcopy(self._proposed_arguments),
            "relations": [],
        }


def _read_json(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    payload = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"expected a JSON object: {input_path}"
        )

    return payload


def _parse_importance_edits(
    values: list[str],
) -> dict[str, float]:
    edits: dict[str, float] = {}

    for value in values:
        aspect, separator, raw_importance = value.partition("=")
        aspect = aspect.strip()

        if not separator or not aspect:
            raise ValueError(
                "--set-importance expects ASPECT=VALUE, "
                "for example bruit_calme=5"
            )

        if aspect not in HOTEL_ASPECT_SET:
            raise ValueError(
                f"unknown hotel aspect: {aspect}"
            )

        try:
            importance = float(raw_importance)
        except ValueError as exc:
            raise ValueError(
                f"invalid importance: {value}"
            ) from exc

        if not 0.0 <= importance <= 5.0:
            raise ValueError(
                f"importance must be between 0 and 5: {value}"
            )

        edits[aspect] = importance

    return edits


def apply_contestation(
    preferences_payload: dict[str, Any],
    *,
    importance_edits: dict[str, float],
    disabled_aspects: list[str],
    disabled_constraint_ids: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modified = deepcopy(preferences_payload)

    aspect_preferences = modified.get(
        "aspect_preferences"
    )

    if not isinstance(aspect_preferences, dict):
        raise ValueError(
            "session_preferences.aspect_preferences is missing"
        )

    constraints = modified.get("constraints")

    if not isinstance(constraints, list):
        raise ValueError(
            "session_preferences.constraints is missing"
        )

    changes: list[dict[str, Any]] = []

    # Désactivation des contraintes souples.
    for constraint_id in disabled_constraint_ids:
        matches = [
            (index, constraint)
            for index, constraint in enumerate(constraints)
            if isinstance(constraint, dict)
            and constraint.get("constraint_id")
            == constraint_id
        ]

        if not matches:
            raise ValueError(
                f"unknown constraint_id: {constraint_id}"
            )

        index, constraint = matches[0]

        if constraint.get("mode") == "hard":
            raise ValueError(
                "simple_v1 cannot disable a hard "
                f"constraint: {constraint_id}"
            )

        constraints.pop(index)

        changes.append({
            "operation": "disable_soft_constraint",
            "constraint_id": constraint_id,
            "target": constraint.get(
                "target",
                constraint.get("field"),
            ),
            "before_importance": constraint.get(
                "importance_raw"
            ),
            "after_importance": None,
        })

    # Désactivation des préférences par aspect.
    for aspect in disabled_aspects:
        if aspect not in HOTEL_ASPECT_SET:
            raise ValueError(
                f"unknown hotel aspect: {aspect}"
            )

        if aspect not in aspect_preferences:
            raise ValueError(
                "cannot disable an aspect absent from "
                f"the original profile: {aspect}"
            )

        previous = aspect_preferences.pop(aspect)

        changes.append({
            "operation": "disable_aspect",
            "aspect": aspect,
            "before_importance": previous.get(
                "importance_raw"
            ),
            "after_importance": None,
        })

    # Modification des importances.
    for aspect, importance in importance_edits.items():
        if aspect not in aspect_preferences:
            raise ValueError(
                "cannot reweight an aspect absent or "
                "disabled in the original profile: "
                f"{aspect}"
            )

        previous = aspect_preferences[aspect].get(
            "importance_raw"
        )

        aspect_preferences[aspect][
            "importance_raw"
        ] = importance

        changes.append({
            "operation": "set_importance",
            "aspect": aspect,
            "before_importance": previous,
            "after_importance": importance,
        })

    if not changes:
        raise ValueError(
            "at least one contestation edit is required"
        )

    return modified, changes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate one hotel after simple structured "
            "preference edits, reusing the original Gemini "
            "proposals without a new LLM call."
        )
    )

    parser.add_argument(
        "--profiles",
        required=True,
    )

    parser.add_argument(
        "--evaluation",
        required=True,
    )

    parser.add_argument(
        "--set-importance",
        action="append",
        default=[],
        metavar="ASPECT=VALUE",
    )

    parser.add_argument(
        "--disable-aspect",
        action="append",
        default=[],
        metavar="ASPECT",
    )

    parser.add_argument(
        "--disable-constraint",
        action="append",
        default=[],
        metavar="CONSTRAINT_ID",
        help="Disable one existing soft constraint by id.",
    )

    parser.add_argument(
        "--facility-ontology",
        default=str(
            DEFAULT_FACILITY_ONTOLOGY_PATH
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--html-output",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        original = _read_json(args.evaluation)

        if original.get("argument_mode") != "hybrid":
            raise ValueError(
                "the source evaluation must use hybrid mode"
            )

        hotel_id = original.get("hotel_id")

        if not isinstance(hotel_id, str) or not hotel_id:
            raise ValueError(
                "the source evaluation has no valid hotel_id"
            )

        preferences_payload = original.get(
            "session_preferences"
        )

        if not isinstance(preferences_payload, dict):
            raise ValueError(
                "the source evaluation has no "
                "session_preferences"
            )

        hybrid = original.get("hybrid")

        validation = (
            hybrid.get("validation")
            if isinstance(hybrid, dict)
            else None
        )

        proposals = (
            validation.get("proposed_arguments")
            if isinstance(validation, dict)
            else None
        )

        if not isinstance(proposals, list):
            raise ValueError(
                "the source evaluation has no replayable "
                "hybrid proposals"
            )

        importance_edits = _parse_importance_edits(
            args.set_importance
        )

        modified_payload, changes = apply_contestation(
            preferences_payload,
            importance_edits=importance_edits,
            disabled_aspects=args.disable_aspect,
            disabled_constraint_ids=(
                args.disable_constraint
            ),
        )

        preferences = session_preferences_from_dict(
            modified_payload
        )

        dataset = load_hotel_profiles(
            args.profiles
        )

        ontology = FacilityOntology.load(
            args.facility_ontology
        )

        generator = ReplayHybridArgumentGenerator(
            proposals
        )

        result = evaluate_hotel_by_id(
            dataset,
            hotel_id,
            preferences,
            argument_mode="hybrid",
            hybrid_generator=generator,
                       facility_ontology=ontology,
        )

        output_payload = result.to_dict()

        output_payload["contestation"] = {
            "version": "simple_v1",
            "source_evaluation": str(
                Path(args.evaluation)
            ),
            "changes": changes,
            "llm_request_count": 0,
            "reused_proposal_count": len(proposals),
            "before": {
                "eligibility": original.get(
                    "eligibility",
                    {},
                ).get("status"),
                "dfquad_score": original.get(
                    "dfquad_score"
                ),
            },
            "after": {
                "eligibility": output_payload[
                    "eligibility"
                ]["status"],
                "dfquad_score": output_payload[
                    "dfquad_score"
                ],
            },
        }

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                output_payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        if args.html_output:
            render_hotel_graph(
                output_payload,
                args.html_output,
            )

    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            output_payload["contestation"],
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "Contested evaluation:",
        output_path,
    )

    if args.html_output:
        print(
            "HTML report:",
            Path(args.html_output),
        )


if __name__ == "__main__":
    main()