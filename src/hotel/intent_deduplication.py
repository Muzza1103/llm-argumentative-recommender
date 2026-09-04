from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .aspects import HOTEL_ASPECTS
from .errors import HotelPreferenceValidationError
from .facility_ontology import FacilityOntology


MAX_ASPECT_PREFERENCES = 15
MAX_PREFERENCE_CONSTRAINTS = 12
MAX_UNINTERPRETED_ITEMS = 20

DEFAULT_UNINTERPRETED_REASONS = frozenset(
    {
        "unsupported_facility_target",
        "unsupported_aspect",
        "unsupported_operator",
        "ambiguous_request",
        "insufficient_information",
    }
)

_ROOT_FIELDS = frozenset(
    {"aspect_preferences", "constraints", "uninterpreted_items"}
)
_ASPECT_FIELDS = frozenset(
    {"aspect", "importance_raw", "source_text"}
)
_CONSTRAINT_FIELD_ORDER = (
    "constraint_id",
    "target_type",
    "target",
    "operator",
    "qualifiers",
    "value",
    "hard",
    "importance_raw",
    "source_text",
)
_CONSTRAINT_FIELDS = frozenset(_CONSTRAINT_FIELD_ORDER)
_UNINTERPRETED_FIELDS = frozenset({"text", "reason"})
_METADATA_TARGETS = frozenset({"city"})


def _normalized_excerpt(value: str) -> str:
    """Normalize only whitespace and case for exact excerpt validation."""
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _is_valid_importance(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and 0.0 <= numeric <= 5.0


def _source_text_reason(value: object, original_text: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "invalid_source_text"
    if _normalized_excerpt(value) not in _normalized_excerpt(original_text):
        return "source_text_not_in_original"
    return None


def _qualifiers_are_valid(
    ontology: FacilityOntology,
    target: str,
    qualifiers: dict[str, Any],
) -> bool:
    specification = ontology.capabilities[target].get("qualifiers", {})
    for key, value in qualifiers.items():
        if key not in specification:
            return False
        allowed_values = specification[key]
        if not isinstance(allowed_values, list):
            return False
        if value == "unknown" or not any(
            type(value) is type(allowed) and value == allowed
            for allowed in allowed_values
        ):
            return False
    return True


def _drop_trace(
    *,
    collection: str,
    index: int,
    proposal: object,
    reasons: list[str],
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "action": "drop_invalid_entry",
        "collection": collection,
        "index": index,
        "reasons": list(reasons),
        "proposal": copy.deepcopy(proposal),
    }
    if isinstance(proposal, Mapping):
        constraint_id = proposal.get("constraint_id")
        if isinstance(constraint_id, str) and constraint_id:
            trace["constraint_id"] = constraint_id
        aspect = proposal.get("aspect")
        if isinstance(aspect, str) and aspect:
            trace["aspect"] = aspect
    return trace


def _canonical_json(value: object) -> str:
    """Build a deterministic, type-preserving structural identity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _constraint_identity(entry: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        entry["target_type"],
        entry["target"],
        entry["operator"],
        _canonical_json(entry["value"]),
        _canonical_json(entry["qualifiers"]),
    )


def _validate_root(
    payload: object,
) -> tuple[list[object], list[object], list[object]]:
    if not isinstance(payload, dict):
        raise HotelPreferenceValidationError(
            "interpreter output must be an object"
        )
    if set(payload) != _ROOT_FIELDS:
        raise HotelPreferenceValidationError(
            "interpreter output has missing or unknown root fields"
        )

    aspects = payload["aspect_preferences"]
    constraints = payload["constraints"]
    uninterpreted = payload["uninterpreted_items"]
    if not isinstance(aspects, list) or not isinstance(constraints, list):
        raise HotelPreferenceValidationError(
            "aspect_preferences and constraints must be lists"
        )
    if not isinstance(uninterpreted, list):
        raise HotelPreferenceValidationError(
            "uninterpreted_items must be a list"
        )
    if len(aspects) > MAX_ASPECT_PREFERENCES:
        raise HotelPreferenceValidationError(
            "interpreter output has too many aspect preferences"
        )
    if len(constraints) > MAX_PREFERENCE_CONSTRAINTS:
        raise HotelPreferenceValidationError(
            "interpreter output has too many constraints"
        )
    if len(uninterpreted) > MAX_UNINTERPRETED_ITEMS:
        raise HotelPreferenceValidationError(
            "interpreter output has too many uninterpreted items"
        )
    return aspects, constraints, uninterpreted


def filter_structurally_valid_preferences(
    payload: object,
    *,
    original_text: str,
    ontology: FacilityOntology,
    uninterpreted_reasons: Iterable[str] = DEFAULT_UNINTERPRETED_REASONS,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Filter Gemini output without reinterpreting the user's meaning.

    Invalid individual entries are dropped and recorded. Valid entries retain
    Gemini's aspect/constraint classification, hardness, importance, value,
    qualifiers, and source excerpt. Duplicate removal is based only on closed
    canonical identifiers and exact structured constraint identities.
    """
    aspects, constraints, uninterpreted = _validate_root(payload)
    if not isinstance(original_text, str):
        raise HotelPreferenceValidationError(
            "original user text must be a string"
        )

    trace: list[dict[str, Any]] = []
    aspect_mapping: dict[str, dict[str, Any]] = {}
    seen_aspects: set[str] = set()

    for index, proposal in enumerate(aspects):
        reasons: list[str] = []
        if not isinstance(proposal, dict):
            reasons.append("expected_object")
            trace.append(
                _drop_trace(
                    collection="aspect_preferences",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue

        missing = _ASPECT_FIELDS.difference(proposal)
        unknown = set(proposal).difference(_ASPECT_FIELDS)
        if missing:
            reasons.append("missing_required_field")
        if unknown:
            reasons.append("unknown_field")

        aspect = proposal.get("aspect")
        if not isinstance(aspect, str) or aspect not in HOTEL_ASPECTS:
            reasons.append("unknown_aspect")
        if not _is_valid_importance(proposal.get("importance_raw")):
            reasons.append("invalid_importance_raw")
        source_reason = _source_text_reason(
            proposal.get("source_text"),
            original_text,
        )
        if source_reason is not None:
            reasons.append(source_reason)

        if not reasons and aspect in seen_aspects:
            reasons.append("duplicate_aspect")
        if reasons:
            trace.append(
                _drop_trace(
                    collection="aspect_preferences",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue

        seen_aspects.add(aspect)
        aspect_mapping[aspect] = {
            "importance_raw": proposal["importance_raw"],
            "source_text": proposal["source_text"],
        }

    constraint_rows: list[dict[str, Any]] = []
    seen_constraint_ids: set[str] = set()
    seen_constraint_identities: set[tuple[str, ...]] = set()

    for index, proposal in enumerate(constraints):
        reasons = []
        if not isinstance(proposal, dict):
            reasons.append("expected_object")
            trace.append(
                _drop_trace(
                    collection="constraints",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue

        missing = _CONSTRAINT_FIELDS.difference(proposal)
        unknown = set(proposal).difference(_CONSTRAINT_FIELDS)
        if missing:
            reasons.append("missing_required_field")
        if unknown:
            reasons.append("unknown_field")

        constraint_id = proposal.get("constraint_id")
        if not isinstance(constraint_id, str) or not constraint_id.strip():
            reasons.append("invalid_constraint_id")

        target_type = proposal.get("target_type")
        target = proposal.get("target")
        operator = proposal.get("operator")
        qualifiers = proposal.get("qualifiers")
        value = proposal.get("value")

        if target_type not in {"facility", "metadata"}:
            reasons.append("invalid_target_type")
        if not isinstance(target, str) or not target:
            reasons.append("unknown_canonical_target")
        elif target_type == "facility" and target not in ontology.capabilities:
            reasons.append("unknown_canonical_target")
        elif target_type == "metadata" and target not in _METADATA_TARGETS:
            reasons.append("unknown_canonical_target")

        if not isinstance(operator, str) or operator not in ontology.operators:
            reasons.append("unsupported_operator")
        elif target_type == "facility" and operator != "present":
            reasons.append("incompatible_operator")
        elif target_type == "metadata" and operator != "equals":
            reasons.append("incompatible_operator")

        if not isinstance(qualifiers, dict):
            reasons.append("invalid_qualifiers")
        elif target_type == "facility" and isinstance(target, str):
            if target in ontology.capabilities and not _qualifiers_are_valid(
                ontology,
                target,
                qualifiers,
            ):
                reasons.append("invalid_qualifier")
        elif target_type == "metadata" and qualifiers:
            reasons.append("invalid_qualifier")

        if target_type == "facility" and value is not None:
            reasons.append("invalid_value")
        elif target_type == "metadata" and (
            not isinstance(value, str) or not value.strip()
        ):
            reasons.append("invalid_value")

        if not isinstance(proposal.get("hard"), bool):
            reasons.append("invalid_hard")
        if not _is_valid_importance(proposal.get("importance_raw")):
            reasons.append("invalid_importance_raw")
        source_reason = _source_text_reason(
            proposal.get("source_text"),
            original_text,
        )
        if source_reason is not None:
            reasons.append(source_reason)

        identity: tuple[str, ...] | None = None
        if not reasons:
            identity = _constraint_identity(proposal)
            if constraint_id in seen_constraint_ids:
                reasons.append("duplicate_constraint_id")
            if identity in seen_constraint_identities:
                reasons.append("duplicate_constraint")
        if reasons:
            trace.append(
                _drop_trace(
                    collection="constraints",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue

        seen_constraint_ids.add(constraint_id)
        seen_constraint_identities.add(identity)
        constraint_rows.append(
            {
                field: copy.deepcopy(proposal[field])
                for field in _CONSTRAINT_FIELD_ORDER
            }
        )
        constraint_rows[-1]["text"] = proposal["source_text"]

    clean_uninterpreted: list[dict[str, str]] = []
    allowed_reasons = frozenset(uninterpreted_reasons)
    for index, proposal in enumerate(uninterpreted):
        reasons = []
        if not isinstance(proposal, dict):
            reasons.append("expected_object")
            trace.append(
                _drop_trace(
                    collection="uninterpreted_items",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue

        missing = _UNINTERPRETED_FIELDS.difference(proposal)
        unknown = set(proposal).difference(_UNINTERPRETED_FIELDS)
        if missing:
            reasons.append("missing_required_field")
        if unknown:
            reasons.append("unknown_field")
        if proposal.get("reason") not in allowed_reasons:
            reasons.append("unknown_uninterpreted_reason")
        source_reason = _source_text_reason(
            proposal.get("text"),
            original_text,
        )
        if source_reason is not None:
            reasons.append(source_reason)
        if reasons:
            trace.append(
                _drop_trace(
                    collection="uninterpreted_items",
                    index=index,
                    proposal=proposal,
                    reasons=reasons,
                )
            )
            continue
        clean_uninterpreted.append(
            {"text": proposal["text"], "reason": proposal["reason"]}
        )

    return (
        {
            "aspect_preferences": aspect_mapping,
            "constraints": constraint_rows,
            "uninterpreted_items": clean_uninterpreted,
        },
        tuple(trace),
    )


def deduplicate_preference_intentions(
    preferences: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Remove only exact structural duplicates from validated preferences.

    This compatibility helper never changes semantic fields and never moves
    an intention across the aspect/constraint boundary.
    """
    result = copy.deepcopy(dict(preferences))
    raw_constraints = result.get("constraints", [])
    if not isinstance(raw_constraints, list):
        return result, ()

    kept: list[Any] = []
    seen: set[tuple[str, ...]] = set()
    trace: list[dict[str, Any]] = []
    identity_fields = {
        "target_type",
        "target",
        "operator",
        "value",
        "qualifiers",
    }
    for index, constraint in enumerate(raw_constraints):
        if not isinstance(constraint, Mapping) or not identity_fields.issubset(
            constraint
        ):
            kept.append(copy.deepcopy(constraint))
            continue
        identity = _constraint_identity(constraint)
        if identity in seen:
            trace.append(
                _drop_trace(
                    collection="constraints",
                    index=index,
                    proposal=constraint,
                    reasons=["duplicate_constraint"],
                )
            )
            continue
        seen.add(identity)
        kept.append(copy.deepcopy(constraint))
    result["constraints"] = kept
    return result, tuple(trace)
