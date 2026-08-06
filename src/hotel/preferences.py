from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aspects import HOTEL_ASPECTS, validate_hotel_aspect
from .errors import HotelDataValidationError, HotelPreferenceValidationError


PathLike = str | Path
CONSTRAINT_MODES = frozenset({"hard", "soft"})


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HotelPreferenceValidationError("expected an object", path=path)
    return value


def _sequence(value: object, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise HotelPreferenceValidationError("expected a list", path=path)
    return value


def _required_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HotelPreferenceValidationError(
            "expected a non-empty string",
            path=path,
        )
    return value.strip()


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HotelPreferenceValidationError(
            "expected a string or null",
            path=path,
        )
    return value.strip() or None


def _importance(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HotelPreferenceValidationError(
            "expected a finite number between 0 and 5",
            path=path,
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 5.0:
        raise HotelPreferenceValidationError(
            "expected a finite number between 0 and 5",
            path=path,
        )
    return result


@dataclass(frozen=True, slots=True)
class AspectPreference:
    aspect: str
    importance_raw: float
    source_text: str
    normalized_weight: float

    @property
    def active(self) -> bool:
        return self.importance_raw > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "importance_raw": self.importance_raw,
            "source_text": self.source_text,
            "normalized_weight": self.normalized_weight,
        }


@dataclass(frozen=True, slots=True)
class SessionConstraint:
    text: str
    importance_raw: float
    mode: str
    field: str
    value: Any = None
    uninterpreted: dict[str, Any] = field(default_factory=dict)
    constraint_id: str | None = None
    target_type: str | None = None
    target: str | None = None
    operator: str | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    source_text: str | None = None

    @property
    def hard(self) -> bool:
        return self.mode == "hard"

    @property
    def preference_ref(self) -> str:
        return self.constraint_id or f"constraint::{self.field}"

    @property
    def canonical_target(self) -> str:
        return self.target or self.field

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "importance_raw": self.importance_raw,
            "mode": self.mode,
            "field": self.field,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.constraint_id is not None:
            payload["constraint_id"] = self.constraint_id
        if self.target_type is not None:
            payload["target_type"] = self.target_type
        if self.target is not None:
            payload["target"] = self.target
        if self.operator is not None:
            payload["operator"] = self.operator
        if self.qualifiers:
            payload["qualifiers"] = dict(self.qualifiers)
        if self.source_text is not None:
            payload["source_text"] = self.source_text
        if self.uninterpreted:
            payload["uninterpreted"] = dict(self.uninterpreted)
        return payload


@dataclass(frozen=True, slots=True)
class SessionPreferences:
    aspect_preferences: tuple[AspectPreference, ...]
    constraints: tuple[SessionConstraint, ...] = field(default_factory=tuple)
    original_text: str | None = None
    uninterpreted_items: tuple[Any, ...] = field(default_factory=tuple)
    interpretation_trace: dict[str, Any] | None = None

    @property
    def active_aspect_preferences(self) -> tuple[AspectPreference, ...]:
        return tuple(item for item in self.aspect_preferences if item.active)

    def get_aspect(self, aspect: str) -> AspectPreference | None:
        try:
            validated = validate_hotel_aspect(aspect)
        except HotelDataValidationError as exc:
            raise HotelPreferenceValidationError(str(exc)) from exc
        return next(
            (item for item in self.aspect_preferences if item.aspect == validated),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "original_text": self.original_text,
            "aspect_preferences": {
                item.aspect: item.to_dict()
                for item in self.aspect_preferences
            },
            "constraints": [item.to_dict() for item in self.constraints],
            "uninterpreted_items": list(self.uninterpreted_items),
        }
        if self.interpretation_trace is not None:
            payload["interpretation_trace"] = dict(
                self.interpretation_trace
            )
        return payload


def session_preferences_from_dict(
    raw_preferences: object,
    *,
    path: str = "preferences",
    original_text: str | None = None,
) -> SessionPreferences:
    """Validate, canonicalize, and normalize one session preference payload."""
    preferences = _mapping(raw_preferences, path)
    raw_aspects = _mapping(
        preferences.get("aspect_preferences", {}),
        f"{path}.aspect_preferences",
    )

    validated_entries: dict[str, tuple[float, str]] = {}
    for raw_aspect, raw_entry in raw_aspects.items():
        try:
            aspect = validate_hotel_aspect(
                raw_aspect,
                path=f"{path}.aspect_preferences",
            )
        except HotelDataValidationError as exc:
            raise HotelPreferenceValidationError(str(exc)) from exc

        entry_path = f"{path}.aspect_preferences.{aspect}"
        entry = _mapping(raw_entry, entry_path)
        validated_entries[aspect] = (
            _importance(
                entry.get("importance_raw"),
                f"{entry_path}.importance_raw",
            ),
            _required_string(
                entry.get("source_text"),
                f"{entry_path}.source_text",
            ),
        )

    total_importance = sum(
        importance
        for importance, _ in validated_entries.values()
        if importance > 0.0
    )
    aspect_preferences = tuple(
        AspectPreference(
            aspect=aspect,
            importance_raw=validated_entries[aspect][0],
            source_text=validated_entries[aspect][1],
            normalized_weight=(
                validated_entries[aspect][0] / total_importance
                if validated_entries[aspect][0] > 0.0
                and total_importance > 0.0
                else 0.0
            ),
        )
        for aspect in HOTEL_ASPECTS
        if aspect in validated_entries
    )

    constraints = []
    seen_constraint_ids: set[str] = set()
    mode_counts = {"hard": 0, "soft": 0}
    for index, raw_constraint in enumerate(
        _sequence(preferences.get("constraints", []), f"{path}.constraints")
    ):
        constraint_path = f"{path}.constraints[{index}]"
        constraint = _mapping(raw_constraint, constraint_path)
        raw_mode = constraint.get("mode")
        raw_hard = constraint.get("hard")
        if raw_mode is None and isinstance(raw_hard, bool):
            mode = "hard" if raw_hard else "soft"
        else:
            mode = _required_string(
                raw_mode,
                f"{constraint_path}.mode",
            ).lower()
        if mode not in CONSTRAINT_MODES:
            raise HotelPreferenceValidationError(
                "expected 'hard' or 'soft'",
                path=f"{constraint_path}.mode",
            )

        if isinstance(raw_hard, bool) and raw_hard != (mode == "hard"):
            raise HotelPreferenceValidationError(
                "hard and mode describe different constraint modes",
                path=constraint_path,
            )

        mode_counts[mode] += 1
        supplied_constraint_id = constraint.get("constraint_id")
        constraint_id = (
            _required_string(
                supplied_constraint_id,
                f"{constraint_path}.constraint_id",
            )
            if supplied_constraint_id is not None
            else f"{mode}_{mode_counts[mode]:02d}"
        )
        if constraint_id in seen_constraint_ids:
            raise HotelPreferenceValidationError(
                f"duplicate constraint_id {constraint_id!r}",
                path=f"{constraint_path}.constraint_id",
            )
        seen_constraint_ids.add(constraint_id)

        raw_target = constraint.get("target", constraint.get("field"))
        target = _required_string(
            raw_target,
            f"{constraint_path}.target",
        ).casefold()
        raw_target_type = constraint.get("target_type")
        if raw_target_type is None:
            target_type = "metadata" if target in {"city", "ville"} else "facility"
        else:
            target_type = _required_string(
                raw_target_type,
                f"{constraint_path}.target_type",
            ).casefold()
        if target_type not in {"facility", "metadata"}:
            raise HotelPreferenceValidationError(
                "expected 'facility' or 'metadata'",
                path=f"{constraint_path}.target_type",
            )

        default_operator = "equals" if target in {"city", "ville"} else "present"
        operator = _required_string(
            constraint.get("operator", default_operator),
            f"{constraint_path}.operator",
        ).casefold()
        if operator not in {"present", "equals"}:
            raise HotelPreferenceValidationError(
                "expected 'present' or 'equals'",
                path=f"{constraint_path}.operator",
            )
        qualifiers = constraint.get("qualifiers", {})
        if not isinstance(qualifiers, Mapping):
            raise HotelPreferenceValidationError(
                "expected an object",
                path=f"{constraint_path}.qualifiers",
            )

        known_keys = {
            "text",
            "source_text",
            "importance_raw",
            "mode",
            "hard",
            "field",
            "constraint_id",
            "target_type",
            "target",
            "operator",
            "qualifiers",
            "value",
            "expected_value",
        }
        if "value" in constraint and "expected_value" in constraint:
            raise HotelPreferenceValidationError(
                "use either 'value' or 'expected_value', not both",
                path=constraint_path,
            )
        expected_value = constraint.get(
            "value",
            constraint.get("expected_value"),
        )
        source_text = _required_string(
            constraint.get(
                "source_text",
                constraint.get("text"),
            ),
            f"{constraint_path}.source_text",
        )
        constraints.append(
            SessionConstraint(
                text=_required_string(
                    constraint.get("text", source_text),
                    f"{constraint_path}.text",
                ),
                importance_raw=_importance(
                    constraint.get("importance_raw"),
                    f"{constraint_path}.importance_raw",
                ),
                mode=mode,
                field=target,
                value=expected_value,
                uninterpreted={
                    key: value
                    for key, value in constraint.items()
                    if key not in known_keys
                },
                constraint_id=constraint_id,
                target_type=target_type,
                target=target,
                operator=operator,
                qualifiers=dict(qualifiers),
                source_text=source_text,
            )
        )

    explicit_unknown = []
    for key in ("uninterpreted_items", "unknown_items"):
        if key in preferences:
            explicit_unknown.extend(
                _sequence(preferences[key], f"{path}.{key}")
            )

    known_root_keys = {
        "aspect_preferences",
        "constraints",
        "original_text",
        "uninterpreted_items",
        "unknown_items",
        "interpretation_trace",
    }
    explicit_unknown.extend(
        {"field": key, "value": value}
        for key, value in preferences.items()
        if key not in known_root_keys
    )

    resolved_original_text = original_text
    if resolved_original_text is None:
        resolved_original_text = _optional_string(
            preferences.get("original_text"),
            f"{path}.original_text",
        )

    return SessionPreferences(
        aspect_preferences=aspect_preferences,
        constraints=tuple(constraints),
        original_text=resolved_original_text,
        uninterpreted_items=tuple(explicit_unknown),
        interpretation_trace=(
            dict(preferences["interpretation_trace"])
            if isinstance(preferences.get("interpretation_trace"), Mapping)
            else None
        ),
    )


def load_session_preferences(path: PathLike) -> SessionPreferences:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            raw_preferences = json.load(stream)
    except json.JSONDecodeError as exc:
        raise HotelPreferenceValidationError(
            f"invalid JSON: {exc.msg}",
            path=str(input_path),
        ) from exc
    return session_preferences_from_dict(raw_preferences, path=str(input_path))
