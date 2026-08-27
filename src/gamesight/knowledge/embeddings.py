"""Embedding-provider ports and the selected multilingual MiniLM adapter."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any


DEFAULT_MULTILINGUAL_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable model identifier stored with the knowledge index."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document passages."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy Sentence Transformers adapter with normalized output vectors."""

    def __init__(
        self,
        model_id: str = DEFAULT_MULTILINGUAL_MODEL,
        *,
        device: str | None = None,
        cache_folder: str | Path | None = None,
        model: Any | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._cache_folder = Path(cache_folder) if cache_folder else (
            Path(__file__).resolve().parents[3] / "models" / "huggingface"
        )
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self):
        if self._model is None:
            if os.name == "nt":
                try:
                    import truststore
                    truststore.inject_into_ssl()
                except ImportError:
                    pass
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RAG embeddings require sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            kwargs = {"cache_folder": str(self._cache_folder)}
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self._model_id, **kwargs)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        encoder = getattr(model, "encode_document", model.encode)
        vectors = encoder(
            list(texts), normalize_embeddings=True, show_progress_bar=False,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        encoder = getattr(model, "encode_query", model.encode)
        vector = encoder(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)
