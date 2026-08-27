from gamesight.evaluation.rag import RetrievalCase, evaluate_retrieval
from gamesight.knowledge.embeddings import EmbeddingProvider
from gamesight.knowledge.models import KnowledgeChunk
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.knowledge.store import InMemoryKnowledgeStore


class KeywordEmbedding(EmbeddingProvider):
    @property
    def model_id(self) -> str:
        return "keyword-test"

    def _one(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("flash" in lowered or "闪光" in lowered),
            float("economy" in lowered or "经济" in lowered),
        ]

    def embed_documents(self, texts):
        return [self._one(text) for text in texts]

    def embed_query(self, text):
        return self._one(text)


def test_retrieval_metrics_report_recall_and_rank() -> None:
    store = InMemoryKnowledgeStore(KeywordEmbedding())
    store.upsert([
        KnowledgeChunk(
            chunk_id="flash", document_id="doc", title="Flash",
            source_uri="test://policy", content="flash avoidance 闪光躲避",
            chunk_index=0,
        ),
        KnowledgeChunk(
            chunk_id="economy", document_id="doc", title="Economy",
            source_uri="test://policy", content="economy 经济管理",
            chunk_index=1,
        ),
    ])
    metrics = evaluate_retrieval(
        KnowledgeRetriever(store, min_score=0.1),
        [
            RetrievalCase("如何躲避闪光", frozenset({"flash"})),
            RetrievalCase("经济管理", frozenset({"economy"})),
        ],
        top_k=1,
    )
    assert metrics.case_count == 2
    assert metrics.recall_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
