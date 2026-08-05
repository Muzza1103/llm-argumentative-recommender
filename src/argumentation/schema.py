from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ARGUMENT_ASPECT_NORMALIZATION = {
    "alcohol": "drinks",
    "bar": "drinks",
    "full_bar": "drinks",
    "full bar": "drinks",
    "beer": "drinks",
    "wine": "drinks",
    "cocktail": "drinks",
    "cocktails": "drinks",

    "atmosphere": "ambience",
    "ambiance": "ambience",
    "staff": "service",

    "outdoor": "outdoor_seating",
    "outdoor seating": "outdoor_seating",
    "patio": "outdoor_seating",

    "group_friendly": "good_for_groups",
    "group-friendly": "good_for_groups",
    "group friendly": "good_for_groups",
}

ALLOWED_ASPECT_EFFECTS = {
    "present_preferred",
    "missing_preferred",
    "present_disliked",
    "missing_disliked",
    "neutral_or_unclear",
}

ARGUMENT_FAMILIES = {
    "empirical_aspect",
    "structured_fact",
    "semantic_extra",
}

def clean_used_aspects(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned = []
    seen = set()

    for aspect in value:
        if not isinstance(aspect, str):
            continue

        aspect = aspect.strip().lower()
        if not aspect:
            continue
        aspect = ARGUMENT_ASPECT_NORMALIZATION.get(aspect, aspect)

        if aspect in seen:
            continue

        seen.add(aspect)
        cleaned.append(aspect)

    return cleaned


@dataclass
class Argument:
    """
    Internal representation of one generated argument.
    """

    id: str
    arg_type: str  # "support" or "attack"
    text: str
    evidence: list[str]
    aspect_effect: str = "neutral_or_unclear"
    used_aspects: list[str] = field(default_factory=list)

    # Context
    user_id: str | None = None
    target_item_name: str | None = None
    target_item: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    # Scores
    llm_score: float | None = None
    llm_score_reason: str | None = None
    llm_scoring_prompt: str | None = None
    llm_scoring_raw_output: str | None = None

    mf_score: float | None = None
    combined_score: float | None = None

    argument_family: str | None = None
    aspect: str | None = None
    intrinsic_strength: float | None = None
    importance_raw: float | None = None
    normalized_weight: float | None = None
    evidence_score: float | None = None
    n_support: int | None = None
    n_attack: int | None = None
    n_neutral: int | None = None
    review_sources: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def is_support(self) -> bool:
        return self.arg_type == "support"

    def is_attack(self) -> bool:
        return self.arg_type == "attack"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "arg_type": self.arg_type,
            "text": self.text,
            "evidence": self.evidence,
            "aspect_effect": self.aspect_effect,
            "used_aspects": self.used_aspects,
            "user_id": self.user_id,
            "target_item_name": self.target_item_name,
            "llm_score": self.llm_score,
            "llm_score_reason": self.llm_score_reason,
            "llm_scoring_prompt": self.llm_scoring_prompt,
            "llm_scoring_raw_output": self.llm_scoring_raw_output,
            "mf_score": self.mf_score,
            "combined_score": self.combined_score,
            "metadata": self.metadata,
        }

        if self.argument_family is not None:
            payload.update(
                {
                    "argument_family": self.argument_family,
                    "aspect": self.aspect,
                    "intrinsic_strength": self.intrinsic_strength,
                    "importance_raw": self.importance_raw,
                    "normalized_weight": self.normalized_weight,
                    "evidence_score": self.evidence_score,
                    "n_support": self.n_support,
                    "n_attack": self.n_attack,
                    "n_neutral": self.n_neutral,
                    "review_sources": self.review_sources,
                }
            )

        return payload


def build_argument_from_json(
    argument_json: dict[str, Any],
    example: dict[str, Any],
) -> Argument:
    target_item = example.get("target_item", {})

    return Argument(
        id=argument_json["id"],
        arg_type=argument_json["type"],
        text=argument_json["text"],
        evidence=argument_json["evidence"],
        aspect_effect=argument_json.get("aspect_effect", "neutral_or_unclear"),
        used_aspects=clean_used_aspects(argument_json.get("used_aspects", [])),
        user_id=example.get("user_id"),
        target_item_name=target_item.get("name"),
        target_item=target_item,
        history=example.get("history", []),
        argument_family=argument_json.get("argument_family"),
        aspect=argument_json.get("aspect"),
        intrinsic_strength=argument_json.get("intrinsic_strength"),
        importance_raw=argument_json.get("importance_raw"),
        normalized_weight=argument_json.get("normalized_weight"),
        evidence_score=argument_json.get("evidence_score"),
        n_support=argument_json.get("n_support"),
        n_attack=argument_json.get("n_attack"),
        n_neutral=argument_json.get("n_neutral"),
        review_sources=argument_json.get("review_sources", []),
    )


def build_arguments_from_parsed_json(
    parsed_json: dict[str, Any],
    example: dict[str, Any],
) -> list[Argument]:
    arguments_json = parsed_json.get("arguments", [])

    return [
        build_argument_from_json(argument_json, example)
        for argument_json in arguments_json
    ]


def build_arguments_from_scored_json(
    scored_arguments_json: list[dict[str, Any]],
    example: dict[str, Any] | None = None,
) -> list[Argument]:
    arguments = []

    for argument_json in scored_arguments_json:
        argument = Argument(
            id=argument_json["id"],
            arg_type=argument_json["arg_type"],
            text=argument_json["text"],
            evidence=argument_json["evidence"],
            aspect_effect=argument_json.get("aspect_effect", "neutral_or_unclear"),
            used_aspects=clean_used_aspects(argument_json.get("used_aspects", [])),
            user_id=argument_json.get("user_id"),
            target_item_name=argument_json.get("target_item_name"),
            llm_score=argument_json.get("llm_score"),
            llm_score_reason=argument_json.get("llm_score_reason"),
            llm_scoring_prompt=argument_json.get("llm_scoring_prompt"),
            llm_scoring_raw_output=argument_json.get("llm_scoring_raw_output"),
            mf_score=argument_json.get("mf_score"),
            combined_score=argument_json.get("combined_score"),
            argument_family=argument_json.get("argument_family"),
            aspect=argument_json.get("aspect"),
            intrinsic_strength=argument_json.get("intrinsic_strength"),
            importance_raw=argument_json.get("importance_raw"),
            normalized_weight=argument_json.get("normalized_weight"),
            evidence_score=argument_json.get("evidence_score"),
            n_support=argument_json.get("n_support"),
            n_attack=argument_json.get("n_attack"),
            n_neutral=argument_json.get("n_neutral"),
            review_sources=argument_json.get("review_sources", []),
            metadata=argument_json.get("metadata", {}),
        )

        if example is not None:
            argument.target_item = example.get("target_item", {})
            argument.history = example.get("history", [])

        arguments.append(argument)

    return arguments
