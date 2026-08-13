"""Rank several hotels against one reusable session preference profile.

The user request is interpreted once, then the resulting structured profile is
passed unchanged to every hotel evaluation.  Hybrid mode may still make one
argument-generation request per hotel because each hotel has different evidence.

Example::

    python scripts/hotel/rank_hotel_session.py \
      --profiles data/private/hotel_profiles_complete.json \
      --preference-text "Je cherche un hôtel calme à Londres." \
      --hotel-ids lp1bbc3 lp100e29 \
      --candidate-count 10 \
      --seed 42 \
      --argument-mode hybrid \
      --gcp-project jinko-data \
      --gcp-location us-central1 \
      --output-dir outputs/hotel_ranking
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from scripts.hotel.render_hotel_graph import render_hotel_graph
from src.hotel import (
    ABSOLUTE_5_WEIGHTING_METHOD,
    ARGUMENT_MODES,
    DEFAULT_FACILITY_ONTOLOGY_PATH,
    FacilityOntology,
    GeminiHybridArgumentGenerator,
    GeminiPreferenceInterpreter,
    HotelDataValidationError,
    HotelGeminiError,
    HotelHybridValidationError,
    HotelPreferenceValidationError,
    HotelProfileDataset,
    HybridArgumentGenerator,
    PreferenceInterpreter,
    SessionPreferences,
    evaluate_hotel_by_id,
    interpret_session_preferences,
    load_hotel_profiles,
    load_session_preferences,
)


DEFAULT_CANDIDATE_COUNT = 10
DEFAULT_SEED = 42
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

CSV_FIELDNAMES = (
    "rank",
    "hotel_id",
    "hotel_name",
    "eligibility_status",
    "ranking_status",
    "dfquad_score",
    "linear_empirical_score",
    "n_supports",
    "n_attacks",
    "n_scoring_units",
    "n_counted_units",
    "ineligibility_reasons",
    "unknown_constraints",
    "evaluation_json",
    "report_html",
    "error_type",
    "error_message",
)


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    requested_count: int
    explicit_hotel_ids: tuple[str, ...]
    randomly_selected_hotel_ids: tuple[str, ...]
    seed: int
    final_hotel_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_count": self.requested_count,
            "explicit_hotel_ids": list(self.explicit_hotel_ids),
            "randomly_selected_hotel_ids": list(
                self.randomly_selected_hotel_ids
            ),
            "seed": self.seed,
            "final_hotel_ids": list(self.final_hotel_ids),
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    hotel_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateFailure:
    hotel_id: str
    hotel_name: str | None
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotel_id": self.hotel_id,
            "hotel_name": self.hotel_name,
            "error_type": self.error_type,
            "message": self.message,
        }


def _deduplicate_hotel_ids(hotel_ids: Sequence[str]) -> tuple[str, ...]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for hotel_id in hotel_ids:
        if not isinstance(hotel_id, str) or not hotel_id.strip():
            raise ValueError("hotel IDs must be non-empty strings")
        normalized = hotel_id.strip()
        if normalized not in seen:
            seen.add(normalized)
            deduplicated.append(normalized)
    return tuple(deduplicated)


def select_candidate_hotel_ids(
    dataset: HotelProfileDataset,
    explicit_hotel_ids: Sequence[str] | None = None,
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    seed: int = DEFAULT_SEED,
) -> CandidateSelection:
    """Validate explicit IDs and reproducibly fill the candidate set."""

    if isinstance(candidate_count, bool) or candidate_count < 1:
        raise ValueError("candidate-count must be at least 1")

    available_ids = tuple(hotel.hotel_id for hotel in dataset.hotels)
    if candidate_count > len(available_ids):
        raise ValueError(
            "candidate-count exceeds the number of available hotels "
            f"({candidate_count} requested, {len(available_ids)} available)"
        )

    explicit_ids = _deduplicate_hotel_ids(explicit_hotel_ids or ())
    available_id_set = set(available_ids)
    unknown_ids = [
        hotel_id
        for hotel_id in explicit_ids
        if hotel_id not in available_id_set
    ]
    if unknown_ids:
        raise ValueError(
            "unknown hotel_id(s): " + ", ".join(unknown_ids)
        )

    if len(explicit_ids) > candidate_count:
        raise ValueError(
            "candidate-count is smaller than the number of explicit hotel IDs "
            f"({candidate_count} < {len(explicit_ids)})"
        )

    explicit_id_set = set(explicit_ids)
    remaining_ids = [
        hotel_id
        for hotel_id in available_ids
        if hotel_id not in explicit_id_set
    ]
    random_count = candidate_count - len(explicit_ids)
    random_ids = tuple(
        random.Random(seed).sample(remaining_ids, random_count)
        if random_count
        else ()
    )

    return CandidateSelection(
        requested_count=candidate_count,
        explicit_hotel_ids=explicit_ids,
        randomly_selected_hotel_ids=random_ids,
        seed=seed,
        final_hotel_ids=explicit_ids + random_ids,
    )


def build_vertex_client(
    gcp_project: str | None,
    gcp_location: str = "us-central1",
    *,
    client_factory: Callable[..., object] | None = None,
) -> object:
    """Build one Vertex AI client backed by Application Default Credentials."""

    project = (gcp_project or "").strip()
    if not project:
        raise ValueError(
            "--gcp-project or GOOGLE_CLOUD_PROJECT is required for Vertex AI"
        )
    location = (gcp_location or "us-central1").strip() or "us-central1"

    if client_factory is None:
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise ImportError("google-genai is required for Vertex AI") from exc
        client_factory = genai.Client

    return client_factory(
        vertexai=True,
        project=project,
        location=location,
    )


def load_or_interpret_preferences(
    *,
    preference_text: str | None = None,
    preferences_path: str | Path | None = None,
    interpreter: PreferenceInterpreter | None = None,
) -> tuple[SessionPreferences, dict[str, Any]]:
    """Load a profile or interpret text exactly once and describe its source."""

    if (preference_text is None) == (preferences_path is None):
        raise ValueError(
            "exactly one of preference_text and preferences_path is required"
        )

    if preferences_path is not None:
        path = Path(preferences_path)
        preferences = load_session_preferences(path)
        return preferences, {
            "type": "file",
            "path": str(path),
            "original_text": preferences.original_text,
        }

    if not isinstance(preference_text, str) or not preference_text.strip():
        raise ValueError("preference-text must be a non-empty string")
    if interpreter is None:
        raise ValueError(
            "a preference interpreter is required with preference-text"
        )

    preferences = interpret_session_preferences(
        preference_text,
        interpreter,
    )
    return preferences, {
        "type": "text",
        "original_text": preference_text,
    }


def evaluate_candidates(
    dataset: HotelProfileDataset,
    hotel_ids: Sequence[str],
    preferences: SessionPreferences,
    *,
    argument_mode: str,
    facility_ontology: FacilityOntology,
    hybrid_generator: HybridArgumentGenerator | None = None,
    evaluator: Callable[..., Any] = evaluate_hotel_by_id,
) -> tuple[list[CandidateEvaluation], list[CandidateFailure]]:
    """Evaluate every candidate while isolating genuine per-hotel failures."""

    evaluations: list[CandidateEvaluation] = []
    failures: list[CandidateFailure] = []

    for hotel_id in hotel_ids:
        hotel = dataset.get_hotel(hotel_id)
        hotel_name = hotel.metadata.name if hotel is not None else None
        try:
            result = evaluator(
                dataset,
                hotel_id,
                preferences,
                argument_mode=argument_mode,
                hybrid_generator=hybrid_generator,
                facility_ontology=facility_ontology,
            )
            payload = result.to_dict()
            if not isinstance(payload, Mapping):
                raise TypeError("hotel evaluation to_dict() must return a mapping")
            evaluations.append(
                CandidateEvaluation(
                    hotel_id=hotel_id,
                    payload=dict(payload),
                )
            )
        except Exception as exc:
            failures.append(
                CandidateFailure(
                    hotel_id=hotel_id,
                    hotel_name=hotel_name,
                    error_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                )
            )

    return evaluations, failures


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _artifact_stem(hotel_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", hotel_id).strip("._")
    if safe == hotel_id and safe:
        return safe
    digest = hashlib.sha256(hotel_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'hotel'}-{digest}"


def _ranking_row(evaluation: CandidateEvaluation) -> dict[str, Any]:
    payload = evaluation.payload
    eligibility = payload.get("eligibility")
    eligibility_status = (
        eligibility.get("status")
        if isinstance(eligibility, Mapping)
        else None
    )
    arguments = payload.get("arguments")
    argument_rows = (
        arguments
        if isinstance(arguments, list)
        else []
    )
    scoring_units = payload.get("scoring_units")
    unit_rows = (
        scoring_units
        if isinstance(scoring_units, list)
        else []
    )
    stem = _artifact_stem(evaluation.hotel_id)

    return {
        "rank": None,
        "hotel_id": evaluation.hotel_id,
        "hotel_name": payload.get("hotel_name"),
        "eligibility_status": eligibility_status,
        "ranking_status": "not_ranked",
        "dfquad_score": _finite_number(payload.get("dfquad_score")),
        "linear_empirical_score": _finite_number(
            payload.get("linear_empirical_score")
        ),
        "n_supports": sum(
            isinstance(argument, Mapping)
            and argument.get("arg_type") == "support"
            for argument in argument_rows
        ),
        "n_attacks": sum(
            isinstance(argument, Mapping)
            and argument.get("arg_type") == "attack"
            for argument in argument_rows
        ),
        "n_scoring_units": len(unit_rows),
        "n_counted_units": sum(
            isinstance(unit, Mapping)
            and bool(unit.get("included_in_dfquad"))
            for unit in unit_rows
        ),
        "ineligibility_reasons": (
            payload.get("ineligibility_reasons")
            if isinstance(payload.get("ineligibility_reasons"), list)
            else []
        ),
        "unknown_constraints": (
            payload.get("unknown_constraints")
            if isinstance(payload.get("unknown_constraints"), list)
            else []
        ),
        "evaluation_json": f"evaluations/{stem}.json",
        "report_html": f"reports/{stem}.html",
    }


def build_ranking_rows(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank only eligible hotels with finite DF-QuAD scores."""

    rows = [_ranking_row(evaluation) for evaluation in evaluations]
    ranked = [
        row
        for row in rows
        if row["eligibility_status"] == "eligible"
        and row["dfquad_score"] is not None
    ]
    ranked.sort(
        key=lambda row: (
            -row["dfquad_score"],
            row["hotel_id"],
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["ranking_status"] = "ranked"

    not_ranked = [row for row in rows if row not in ranked]
    for row in not_ranked:
        if row["eligibility_status"] == "ineligible":
            row["ranking_status"] = "ineligible"

    return ranked, not_ranked


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    rendered = {field: row.get(field) for field in CSV_FIELDNAMES}
    for field in ("ineligibility_reasons", "unknown_constraints"):
        rendered[field] = json.dumps(
            rendered[field],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return rendered


def _failure_csv_row(failure: CandidateFailure) -> dict[str, Any]:
    return {
        "rank": None,
        "hotel_id": failure.hotel_id,
        "hotel_name": failure.hotel_name,
        "eligibility_status": None,
        "ranking_status": "failed",
        "dfquad_score": None,
        "linear_empirical_score": None,
        "n_supports": None,
        "n_attacks": None,
        "n_scoring_units": None,
        "n_counted_units": None,
        "ineligibility_reasons": "[]",
        "unknown_constraints": "[]",
        "evaluation_json": None,
        "report_html": None,
        "error_type": failure.error_type,
        "error_message": failure.message,
    }


def write_ranking_outputs(
    output_dir: str | Path,
    *,
    preferences: SessionPreferences,
    preference_source: Mapping[str, Any],
    selection: CandidateSelection,
    argument_mode: str,
    evaluations: Sequence[CandidateEvaluation],
    failures: Sequence[CandidateFailure],
) -> dict[str, Any]:
    """Write the reusable preferences, evaluations, reports, and ranking."""

    root = Path(output_dir)
    evaluations_dir = root / "evaluations"
    reports_dir = root / "reports"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_json(root / "session_preferences.json", preferences.to_dict())

    for evaluation in evaluations:
        stem = _artifact_stem(evaluation.hotel_id)
        _write_json(
            evaluations_dir / f"{stem}.json",
            evaluation.payload,
        )
        render_hotel_graph(
            evaluation.payload,
            reports_dir / f"{stem}.html",
        )

    ranking, not_ranked = build_ranking_rows(evaluations)
    failure_rows = [failure.to_dict() for failure in failures]
    summary = {
        "n_candidates": len(selection.final_hotel_ids),
        "n_evaluated": len(evaluations),
        "n_ranked": len(ranking),
        "n_not_ranked": len(not_ranked),
        "n_ineligible": sum(
            row["eligibility_status"] == "ineligible"
            for row in not_ranked
        ),
        "n_failed": len(failures),
    }
    ranking_payload = {
        "schema_version": "1.0",
        "argument_mode": argument_mode,
        "weighting_method": preferences.weighting_method,
        "preference_source": dict(preference_source),
        "candidate_selection": selection.to_dict(),
        "summary": summary,
        "ranking": ranking,
        "not_ranked": not_ranked,
        "failures": failure_rows,
    }
    _write_json(root / "ranking.json", ranking_payload)

    with (root / "ranking.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in (*ranking, *not_ranked))
        writer.writerows(_failure_csv_row(failure) for failure in failures)

    return ranking_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpret one hotel request once, evaluate several candidates, "
            "and rank eligible hotels by DF-QuAD score."
        )
    )
    parser.add_argument("--profiles", required=True)
    preference_group = parser.add_mutually_exclusive_group(required=True)
    preference_group.add_argument("--preference-text")
    preference_group.add_argument("--preferences")
    parser.add_argument("--hotel-ids", nargs="*", default=())
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--argument-mode",
        choices=sorted(ARGUMENT_MODES),
        default="baseline",
    )
    parser.add_argument(
        "--facility-ontology",
        default=str(DEFAULT_FACILITY_ONTOLOGY_PATH),
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--gcp-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Vertex AI project; defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--gcp-location",
        default=(
            os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        ),
        help=(
            "Vertex AI location; defaults to GOOGLE_CLOUD_LOCATION or "
            "us-central1."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset = load_hotel_profiles(args.profiles)
        selection = select_candidate_hotel_ids(
            dataset,
            args.hotel_ids,
            candidate_count=args.candidate_count,
            seed=args.seed,
        )
        ontology = FacilityOntology.load(args.facility_ontology)

        needs_vertex_client = (
            args.preference_text is not None
            or args.argument_mode == "hybrid"
        )
        vertex_client = (
            build_vertex_client(args.gcp_project, args.gcp_location)
            if needs_vertex_client
            else None
        )

        interpreter = (
            GeminiPreferenceInterpreter(
                model_name=args.gemini_model,
                ontology=ontology,
                client=vertex_client,
            )
            if args.preference_text is not None
            else None
        )
        preferences, preference_source = load_or_interpret_preferences(
            preference_text=args.preference_text,
            preferences_path=args.preferences,
            interpreter=interpreter,
        )

        if preferences.weighting_method != ABSOLUTE_5_WEIGHTING_METHOD:
            raise HotelPreferenceValidationError(
                "ranking requires weighting_method='absolute_5'"
            )

        hybrid_generator = (
            GeminiHybridArgumentGenerator(
                model_name=args.gemini_model,
                client=vertex_client,
            )
            if args.argument_mode == "hybrid"
            else None
        )

        evaluations, failures = evaluate_candidates(
            dataset,
            selection.final_hotel_ids,
            preferences,
            argument_mode=args.argument_mode,
            hybrid_generator=hybrid_generator,
            facility_ontology=ontology,
        )
        if not evaluations:
            details = "; ".join(
                f"{failure.hotel_id}: {failure.message}"
                for failure in failures
            )
            raise RuntimeError(
                "no hotel could be evaluated"
                + (f" ({details})" if details else "")
            )

        ranking_payload = write_ranking_outputs(
            args.output_dir,
            preferences=preferences,
            preference_source=preference_source,
            selection=selection,
            argument_mode=args.argument_mode,
            evaluations=evaluations,
            failures=failures,
        )
    except (
        HotelDataValidationError,
        HotelPreferenceValidationError,
        HotelGeminiError,
        HotelHybridValidationError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "ranking_json": str(Path(args.output_dir) / "ranking.json"),
                "summary": ranking_payload["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
