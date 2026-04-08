from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Argument:
    """
    Internal representation of one generated argument.
    """

    id: str
    arg_type: str  # "support" or "attack"
    text: str
    evidence: list[str]

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

    # Optional metadata for later use
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_support(self) -> bool:
        return self.arg_type == "support"

    def is_attack(self) -> bool:
        return self.arg_type == "attack"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "arg_type": self.arg_type,
            "text": self.text,
            "evidence": self.evidence,
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


def build_argument_from_json(
    argument_json: dict[str, Any],
    example: dict[str, Any],
) -> Argument:
    """
    Build one Argument object from one generated JSON argument
    and the original dataset example.
    """
    target_item = example.get("target_item", {})

    return Argument(
        id=argument_json["id"],
        arg_type=argument_json["type"],
        text=argument_json["text"],
        evidence=argument_json["evidence"],
        user_id=example.get("user_id"),
        target_item_name=target_item.get("name"),
        target_item=target_item,
        history=example.get("history", []),
    )


def build_arguments_from_parsed_json(
    parsed_json: dict[str, Any],
    example: dict[str, Any],
) -> list[Argument]:
    """
    Build a list of Argument objects from a parsed generated JSON output.
    """
    arguments_json = parsed_json.get("arguments", [])

    return [
        build_argument_from_json(argument_json, example)
        for argument_json in arguments_json
    ]