"""Retrieval service with score filtering and audit-friendly results."""

from __future__ import annotations

from datetime import date, datetime

from gamesight.knowledge.models import KnowledgeLayer, RetrievedKnowledge
from gamesight.knowledge.routing import DecisionContext, KnowledgeQueryRouter
from gamesight.knowledge.store import KnowledgeStore


class KnowledgeRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        *,
        top_k: int = 4,
        min_score: float = 0.15,
        router: KnowledgeQueryRouter | None = None,
        dynamic_max_age_days: int = 365,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_score = min_score
        self.router = router or KnowledgeQueryRouter()
        self.dynamic_max_age_days = dynamic_max_age_days

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        layers: list[KnowledgeLayer] | None = None,
    ) -> list[RetrievedKnowledge]:
        if not query.strip():
            return []
        results = self.store.query(
            query, top_k=top_k or self.top_k, layers=layers,
        )
        return [
            result for result in results
            if result.score >= self.min_score and self._is_usable(result)
        ]

    def retrieve_for_decision(
        self,
        query: str,
        context: DecisionContext | None = None,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedKnowledge]:
        """Retrieve from ordered layers, reserving space for situation guidance."""
        limit = top_k or self.top_k
        if limit <= 0 or not query.strip():
            return []
        routed_query = " | ".join([query, *(context.query_terms() if context else [])])
        layers = self.router.plan(routed_query, context)
        selected: dict[str, RetrievedKnowledge] = {}

        # One candidate per layer prevents a generic weapon passage from
        # displacing a matching 1v4/post-plant decision rule.
        for layer in layers[:limit]:
            hits = self.retrieve(
                routed_query, top_k=max(4, self.top_k), layers=[layer],
            )
            if hits:
                selected[hits[0].chunk.chunk_id] = hits[0]
        if len(selected) < limit:
            for hit in self.retrieve(routed_query, top_k=limit * 4, layers=layers):
                selected.setdefault(hit.chunk.chunk_id, hit)
                if len(selected) >= limit:
                    break

        priority = {layer: index for index, layer in enumerate(layers)}
        results = list(selected.values())
        results.sort(key=lambda hit: (
            priority.get(hit.chunk.layer, len(priority)),
            -hit.score,
            hit.chunk.chunk_id,
        ))
        return results[:limit]

    @property
    def chunk_count(self) -> int:
        return self.store.count()

    def _is_usable(self, result: RetrievedKnowledge) -> bool:
        chunk = result.chunk
        if not chunk.version_sensitive:
            return True
        if not chunk.last_verified or not chunk.source_urls:
            return False
        try:
            verified = datetime.strptime(chunk.last_verified, "%Y-%m-%d").date()
            expires = (
                datetime.strptime(chunk.expires_at, "%Y-%m-%d").date()
                if chunk.expires_at else None
            )
        except ValueError:
            return False
        today = date.today()
        return (
            (today - verified).days <= self.dynamic_max_age_days
            and (expires is None or today <= expires)
        )
