from .schema import (
    Argument,
    build_argument_from_json,
    build_arguments_from_parsed_json,
    build_arguments_from_scored_json,
)
from .scoring import (
    ScoreConfig,
    BaseLLMScorer,
    BaseMFScorer,
    DummyMFScorer,
    combine_scores,
    score_argument,
    score_arguments,
)
from .graph_builder import (
    ArgumentNode,
    ArgumentEdge,
    ArgumentGraph,
    build_argument_graph,
    get_argument_base_score,
)
from .dfquad import (
    DFQuADResult,
    aggregate_strength,
    dfquad_combine,
    evaluate_root_dfquad,
)
from .mf_scorer import (
    MFScorerConfig,
    GlobalRatingFallbackMFScorer,
    PrecomputedMFScorer,
)

__all__ = [
    "Argument",
    "build_argument_from_json",
    "build_arguments_from_parsed_json",
    "build_arguments_from_scored_json",
    "ScoreConfig",
    "BaseLLMScorer",
    "BaseMFScorer",
    "DummyMFScorer",
    "combine_scores",
    "score_argument",
    "score_arguments",
    "LLMScorerConfig",
    "LocalLLMScorer",
    "ArgumentNode",
    "ArgumentEdge",
    "ArgumentGraph",
    "build_argument_graph",
    "get_argument_base_score",
    "DFQuADResult",
    "MFScorerConfig",
    "GlobalRatingFallbackMFScorer",
    "PrecomputedMFScorer",
    "aggregate_strength",
    "dfquad_combine",
    "evaluate_root_dfquad",
]


def __getattr__(name: str):
    """Load local-LLM scoring only when explicitly requested.

    Importing the deterministic schema, graph, or DF-QuAD core must not require
    optional Torch/Transformers dependencies.  Attribute access remains
    backward compatible for callers importing these scorer names from the
    package root.
    """
    if name in {"LLMScorerConfig", "LocalLLMScorer"}:
        from .llm_scorer import LLMScorerConfig, LocalLLMScorer

        return {
            "LLMScorerConfig": LLMScorerConfig,
            "LocalLLMScorer": LocalLLMScorer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
