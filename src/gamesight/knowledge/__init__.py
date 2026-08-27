"""Local, source-attributed knowledge retrieval for the AI coach."""

from gamesight.knowledge.chunking import KnowledgeChunker
from gamesight.knowledge.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from gamesight.knowledge.loaders import (
    load_knowledge_bytes,
    load_knowledge_document,
)
from gamesight.knowledge.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeLayer,
    RetrievedKnowledge,
    RuleStrength,
)
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.knowledge.routing import DecisionContext, KnowledgeQueryRouter
from gamesight.knowledge.store import (
    ChromaKnowledgeStore,
    InMemoryKnowledgeStore,
    LayeredChromaKnowledgeStore,
    LayeredKnowledgeStore,
)

__all__ = [
    "ChromaKnowledgeStore",
    "EmbeddingProvider",
    "InMemoryKnowledgeStore",
    "KnowledgeChunk",
    "KnowledgeChunker",
    "KnowledgeDocument",
    "KnowledgeLayer",
    "KnowledgeQueryRouter",
    "KnowledgeRetriever",
    "LayeredChromaKnowledgeStore",
    "LayeredKnowledgeStore",
    "RetrievedKnowledge",
    "RuleStrength",
    "DecisionContext",
    "SentenceTransformerEmbeddingProvider",
    "load_knowledge_bytes",
    "load_knowledge_document",
]
