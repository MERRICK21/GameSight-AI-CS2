"""Vector-store ports with local Chroma and deterministic in-memory adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gamesight.knowledge.embeddings import EmbeddingProvider
from gamesight.knowledge.models import KnowledgeChunk, KnowledgeLayer, RetrievedKnowledge


class KnowledgeStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[KnowledgeChunk]) -> int:
        """Insert or replace chunks and return the number processed."""

    @abstractmethod
    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        layers: Sequence[KnowledgeLayer] | None = None,
    ) -> list[RetrievedKnowledge]:
        """Return semantically similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return indexed chunk count."""


class InMemoryKnowledgeStore(KnowledgeStore):
    """Small-corpus store used by tests and dependency-free development."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, chunks: Sequence[KnowledgeChunk]) -> int:
        chunks = list(chunks)
        vectors = self.embedder.embed_documents([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            array = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(array))
            self._vectors[chunk.chunk_id] = array / norm if norm else array
        return len(chunks)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        layers: Sequence[KnowledgeLayer] | None = None,
    ) -> list[RetrievedKnowledge]:
        if not self._chunks or top_k <= 0:
            return []
        query = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        allowed = set(layers) if layers else None
        scored = [
            (float(np.dot(query, self._vectors[chunk_id])), chunk)
            for chunk_id, chunk in self._chunks.items()
            if allowed is None or chunk.layer in allowed
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievedKnowledge(chunk=chunk, score=max(-1.0, min(1.0, score)))
            for score, chunk in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._chunks)


class ChromaKnowledgeStore(KnowledgeStore):
    """Persistent local Chroma collection with caller-owned embeddings."""

    def __init__(
        self,
        path: str | Path,
        embedder: EmbeddingProvider,
        *,
        collection_name: str = "gamesight_cs2_knowledge",
        reset: bool = False,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Persistent RAG requires chromadb. Install with: pip install chromadb"
            ) from exc
        self.embedder = embedder
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        if reset:
            try:
                self._client.delete_collection(collection_name)
            except Exception:
                pass
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Evidence-gated CS2 coaching knowledge",
                "embedding_model": embedder.model_id,
                "hnsw:space": "cosine",
            },
        )

    def upsert(self, chunks: Sequence[KnowledgeChunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([chunk.content for chunk in chunks])
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "document_id": chunk.document_id,
                "title": chunk.title,
                "source_uri": chunk.source_uri,
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading or "",
                "language": chunk.language or "",
                "layer": chunk.layer.value,
                "rule_strength": chunk.rule_strength.value,
                "version_sensitive": chunk.version_sensitive,
                "last_verified": chunk.last_verified or "",
                "effective_from": chunk.effective_from or "",
                "expires_at": chunk.expires_at or "",
                "source_urls_json": json.dumps(chunk.source_urls, ensure_ascii=False),
                "exceptions_json": json.dumps(chunk.exceptions, ensure_ascii=False),
                "metadata_json": json.dumps(
                    chunk.metadata, ensure_ascii=False, sort_keys=True,
                ),
            })
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        layers: Sequence[KnowledgeLayer] | None = None,
    ) -> list[RetrievedKnowledge]:
        if top_k <= 0 or self.count() == 0:
            return []
        where = None
        if layers:
            values = [layer.value for layer in layers]
            where = {"layer": values[0]} if len(values) == 1 else {"layer": {"$in": values}}
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        retrieved: list[RetrievedKnowledge] = []
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances,
        ):
            extra = json.loads(metadata.get("metadata_json") or "{}")
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                document_id=metadata["document_id"],
                title=metadata["title"],
                source_uri=metadata["source_uri"],
                content=content,
                chunk_index=int(metadata["chunk_index"]),
                heading=metadata.get("heading") or None,
                language=metadata.get("language") or None,
                layer=metadata.get("layer") or KnowledgeLayer.TACTICAL_FUNDAMENTALS,
                rule_strength=metadata.get("rule_strength") or "strategic_principle",
                version_sensitive=bool(metadata.get("version_sensitive", False)),
                last_verified=metadata.get("last_verified") or None,
                effective_from=metadata.get("effective_from") or None,
                expires_at=metadata.get("expires_at") or None,
                source_urls=json.loads(metadata.get("source_urls_json") or "[]"),
                exceptions=json.loads(metadata.get("exceptions_json") or "[]"),
                metadata=extra,
            )
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            retrieved.append(RetrievedKnowledge(chunk=chunk, score=score))
        return retrieved

    def count(self) -> int:
        return int(self._collection.count())


class LayeredKnowledgeStore(KnowledgeStore):
    """Route writes and queries to one physical store per knowledge layer."""

    def __init__(self, stores: dict[KnowledgeLayer, KnowledgeStore]) -> None:
        missing = set(KnowledgeLayer) - set(stores)
        if missing:
            raise ValueError(f"Missing knowledge stores for: {sorted(item.value for item in missing)}")
        self.stores = stores

    def upsert(self, chunks: Sequence[KnowledgeChunk]) -> int:
        grouped = {layer: [] for layer in KnowledgeLayer}
        for chunk in chunks:
            grouped[chunk.layer].append(chunk)
        return sum(
            self.stores[layer].upsert(layer_chunks)
            for layer, layer_chunks in grouped.items()
            if layer_chunks
        )

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        layers: Sequence[KnowledgeLayer] | None = None,
    ) -> list[RetrievedKnowledge]:
        selected = list(layers or KnowledgeLayer)
        if top_k <= 0 or not selected:
            return []
        # Ask each collection independently so a dense generic collection cannot
        # crowd the high-value situation layer out of the candidate set.
        candidates = [
            hit
            for layer in selected
            for hit in self.stores[layer].query(text, top_k=top_k, layers=[layer])
        ]
        candidates.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return candidates[:top_k]

    def count(self) -> int:
        return sum(store.count() for store in self.stores.values())

    def counts_by_layer(self) -> dict[KnowledgeLayer, int]:
        return {layer: store.count() for layer, store in self.stores.items()}


class LayeredChromaKnowledgeStore(LayeredKnowledgeStore):
    """Persistent four-collection Chroma index."""

    def __init__(
        self,
        path: str | Path,
        embedder: EmbeddingProvider,
        *,
        reset: bool = False,
    ) -> None:
        stores = {
            layer: ChromaKnowledgeStore(
                path,
                embedder,
                collection_name=f"gamesight_cs2_{layer.value}",
                reset=reset,
            )
            for layer in KnowledgeLayer
        }
        super().__init__(stores)
