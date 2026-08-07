from __future__ import annotations

import re
import unicodedata
from types import MappingProxyType
from typing import Mapping


def normalize_city_name(value: str) -> str:
    """Normalize city spelling without guessing a different city.

    The normalization is deliberately lexical only: Unicode compatibility,
    case, accents, punctuation and repeated whitespace are normalized.  No
    fuzzy, phonetic or geographic matching is performed.
    """
    if not isinstance(value, str):
        raise TypeError("city name must be a string")
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", without_marks)
    return re.sub(r"\s+", " ", normalized).strip()


# Explicit, reviewable aliases only.  Keys and values use normalize_city_name.
# Unknown names intentionally fall through unchanged after lexical normalization.
CITY_NAME_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "londres": "london",
    }
)


def canonicalize_city_name(value: str) -> str:
    """Return the conservative canonical city key used for comparisons."""
    normalized = normalize_city_name(value)
    return CITY_NAME_ALIASES.get(normalized, normalized)
