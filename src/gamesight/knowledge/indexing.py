"""Reproducible knowledge-index construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from gamesight.knowledge.chunking import KnowledgeChunker
from gamesight.knowledge.models import KnowledgeDocument
from gamesight.knowledge.store import KnowledgeStore


@dataclass(frozen=True)
class KnowledgeIndexResult:
    document_count: int
    chunk_count: int


def index_documents(
    documents: Iterable[KnowledgeDocument],
    store: KnowledgeStore,
    *,
    chunker: KnowledgeChunker | None = None,
) -> KnowledgeIndexResult:
    chunker = chunker or KnowledgeChunker()
    documents = list(documents)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    store.upsert(chunks)
    return KnowledgeIndexResult(
        document_count=len(documents), chunk_count=len(chunks),
    )
