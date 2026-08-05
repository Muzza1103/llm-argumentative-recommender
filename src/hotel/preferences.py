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

    @property
    def hard(self) -> bool:
        return self.mode == "hard"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "importance_raw": self.importance_raw,
            "mode": self.mode,
            "field": self.field,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.uninterpreted:
            payload["uninterpreted"] = dict(self.uninterpreted)
        return payload


@dataclass(frozen=True, slots=True)
class SessionPreferences:
    aspect_preferences: tuple[AspectPreference, ...]
    constraints: tuple[SessionConstraint, ...] = field(default_factory=tuple)
    original_text: str | None = None
    uninterpreted_items: tuple[Any, ...] = field(default_factory=tuple)

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
        return {
            "original_text": self.original_text,
            "aspect_preferences": {
                item.aspect: item.to_dict()
                for item in self.aspect_preferences
            },
            "constraints": [item.to_dict() for item in self.constraints],
            "uninterpreted_items": list(self.uninterpreted_items),
        }


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
    for index, raw_constraint in enumerate(
        _sequence(preferences.get("constraints", []), f"{path}.constraints")
    ):
        constraint_path = f"{path}.constraints[{index}]"
        constraint = _mapping(raw_constraint, constraint_path)
        mode = _required_string(
            constraint.get("mode"),
            f"{constraint_path}.mode",
        ).lower()
        if mode not in CONSTRAINT_MODES:
            raise HotelPreferenceValidationError(
                "expected 'hard' or 'soft'",
                path=f"{constraint_path}.mode",
            )

        known_keys = {
            "text",
            "importance_raw",
            "mode",
            "field",
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
        constraints.append(
            SessionConstraint(
                text=_required_string(
                    constraint.get("text"),
                    f"{constraint_path}.text",
                ),
                importance_raw=_importance(
                    constraint.get("importance_raw"),
                    f"{constraint_path}.importance_raw",
                ),
                mode=mode,
                field=_required_string(
                    constraint.get("field"),
                    f"{constraint_path}.field",
                ).casefold(),
                value=expected_value,
                uninterpreted={
                    key: value
                    for key, value in constraint.items()
                    if key not in known_keys
                },
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
