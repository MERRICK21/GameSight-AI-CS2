"""Small, provider-neutral retrieval evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

from gamesight.knowledge.retriever import KnowledgeRetriever


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    case_count: int
    recall_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    retriever: KnowledgeRetriever,
    cases: list[RetrievalCase],
    *,
    top_k: int = 5,
) -> RetrievalMetrics:
    """Measure whether known relevant passages are retrieved and how early."""
    if not cases:
        return RetrievalMetrics(0, 0.0, 0.0)
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        results = retriever.retrieve(case.query, top_k=top_k)
        ranked_ids = [result.chunk.chunk_id for result in results]
        rank = next((
            index for index, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in case.relevant_chunk_ids
        ), None)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return RetrievalMetrics(
        case_count=len(cases),
        recall_at_k=hits / len(cases),
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(cases),
    )
