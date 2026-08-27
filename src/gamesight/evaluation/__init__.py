"""Offline evaluation helpers for retrieval and coaching quality."""

from gamesight.evaluation.rag import (
    RetrievalCase,
    RetrievalMetrics,
    evaluate_retrieval,
)

__all__ = ["RetrievalCase", "RetrievalMetrics", "evaluate_retrieval"]
