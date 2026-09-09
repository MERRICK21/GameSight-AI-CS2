"""Evidence and citation gates for RAG + LLM coaching."""

from __future__ import annotations

from pathlib import Path

from gamesight.coach.engine import RuleBasedCoach
from gamesight.coach.rag_engine import EvidenceBoundRagCoach
from gamesight.domain.models import (
    AnalysisResult,
    EventType,
    Evidence,
    GameEvent,
    RoundAnalysis,
    RoundContextEvidence,
    VideoInput,
    VideoMetadata,
)
from gamesight.knowledge.chunking import KnowledgeChunker
from gamesight.knowledge.embeddings import EmbeddingProvider
from gamesight.knowledge.indexing import index_documents
from gamesight.knowledge.loaders import load_knowledge_bytes
from gamesight.knowledge.models import KnowledgeLayer, RuleStrength
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.knowledge.store import InMemoryKnowledgeStore
from gamesight.llm.client import JsonLLMClient
from gamesight.llm.models import JsonGenerationResult, LLMUsage
from gamesight.reporting.builder import EvidenceReportBuilder


class ConstantEmbedding(EmbeddingProvider):
    @property
    def model_id(self) -> str:
        return "constant"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeLLM(JsonLLMClient):
    def __init__(self, content: dict, available: bool = True) -> None:
        self.content = content
        self._available = available

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-json"

    @property
    def available(self) -> bool:
        return self._available

    def generate_json(self, system_prompt, user_prompt):
        return JsonGenerationResult(
            content=self.content,
            provider=self.provider,
            model=self.model,
            latency_ms=12,
            usage=LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )


def _analysis_and_report():
    event = GameEvent(
        event_id="enemy_first_visible_test",
        event_type=EventType.ENEMY_FIRST_VISIBLE,
        start_sec=21.5,
        confidence=0.9,
        evidence=[Evidence(timestamp_sec=21.5, frame_index=645, source="test")],
    )
    analysis = AnalysisResult(
        video=VideoInput(video_id="test", path=Path("match.mp4")),
        metadata=VideoMetadata(duration_sec=50.0),
        rounds=[RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=50.0, events=[event],
        )],
        capabilities={"analysis_complete": True, "personal_combat": False},
    )
    return analysis, EvidenceReportBuilder().build(analysis)


def _retriever():
    document = load_knowledge_bytes(
        b"Review first contact using cover and crosshair evidence. "
        b"Contact time alone does not prove passive play.",
        "policy.txt",
    )
    store = InMemoryKnowledgeStore(ConstantEmbedding())
    index_documents([document], store, chunker=KnowledgeChunker(max_chars=400))
    return KnowledgeRetriever(store, min_score=0.0), store


def _payload(suggestion_id: str, chunk_id: str, *, reasoning: str = "Contact evidence defines a neutral review window."):
    return {
        "suggestions": [{
            "source_suggestion_id": suggestion_id,
            "reasoning": reasoning,
            "action": "Review cover and crosshair placement using the linked clip.",
            "knowledge_chunk_ids": [chunk_id],
            "evaluation_basis": "decision_quality",
        }],
        "summary": {
            "strengths": ["The analysis retains auditable contact evidence."],
            "weaknesses": ["The clip still requires player review."],
            "practice_drills": ["Review contact clips from cover."],
            "focus_areas": ["Evidence-backed contact review."],
            "overall_assessment": "The available evidence supports a neutral review.",
            "knowledge_chunk_ids": [chunk_id],
        },
    }


def test_valid_generation_preserves_video_evidence_and_adds_citation() -> None:
    analysis, report = _analysis_and_report()
    retriever, store = _retriever()
    base = RuleBasedCoach().run(analysis, report)
    suggestion_id = base.suggestions[0].suggestion_id
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    run = EvidenceBoundRagCoach(
        retriever, FakeLLM(_payload(suggestion_id, chunk_id)),
    ).run(analysis, report)
    assert run.diagnostics.mode == "rag_llm"
    assert run.diagnostics.accepted_enrichments == 1
    assert run.suggestions[0].generated_by == "fake_rag"
    assert run.suggestions[0].evidence == base.suggestions[0].evidence
    assert run.suggestions[0].knowledge_citations[0].chunk_id == chunk_id
    assert run.summary.generated_by == "fake_rag"


def test_unknown_citation_is_rejected_and_rules_remain() -> None:
    analysis, report = _analysis_and_report()
    retriever, _ = _retriever()
    base = RuleBasedCoach().run(analysis, report)
    payload = _payload(base.suggestions[0].suggestion_id, "chunk_not_retrieved")
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.summary.generated_by == "rules"
    assert run.diagnostics.fallback_reason == "all_llm_claims_rejected"


def test_llm_cannot_add_unseen_numeric_claims() -> None:
    analysis, report = _analysis_and_report()
    retriever, store = _retriever()
    base = RuleBasedCoach().run(analysis, report)
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    payload = _payload(
        base.suggestions[0].suggestion_id,
        chunk_id,
        reasoning="The player was exactly 99 percent too slow.",
    )
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.diagnostics.rejected_enrichments >= 1


def test_unavailable_llm_falls_back_without_calling_generation() -> None:
    analysis, report = _analysis_and_report()
    retriever, _ = _retriever()
    run = EvidenceBoundRagCoach(retriever, FakeLLM({}, available=False)).run(
        analysis, report,
    )
    assert run.diagnostics.mode == "rules"
    assert run.diagnostics.fallback_reason == "llm_unavailable"
    assert run.suggestions


def test_strategic_principle_cannot_be_rewritten_as_absolute_rule() -> None:
    analysis, report = _analysis_and_report()
    retriever, store = _retriever()
    base = RuleBasedCoach().run(analysis, report)
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    payload = _payload(
        base.suggestions[0].suggestion_id,
        chunk_id,
        reasoning="You must never take this contact.",
    )
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.diagnostics.rejected_enrichments >= 1


def test_hard_rule_may_use_categorical_language() -> None:
    analysis, report = _analysis_and_report()
    document = load_knowledge_bytes(
        b"CT must defuse a planted bomb before it explodes to win by defusal.",
        "hard-rule.txt",
        metadata={
            "knowledge_layer": KnowledgeLayer.GAME_RULES.value,
            "rule_strength": RuleStrength.HARD_RULE.value,
        },
    )
    store = InMemoryKnowledgeStore(ConstantEmbedding())
    index_documents([document], store, chunker=KnowledgeChunker(max_chars=400))
    retriever = KnowledgeRetriever(store, min_score=0.0)
    base = RuleBasedCoach().run(analysis, report)
    chunk_id = store.query("defuse", top_k=1)[0].chunk.chunk_id
    payload = _payload(
        base.suggestions[0].suggestion_id,
        chunk_id,
        reasoning="CT must defuse a planted bomb before it explodes to win by defusal.",
    )
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "fake_rag"


def test_outcome_does_not_prove_decision_quality() -> None:
    analysis, report = _analysis_and_report()
    retriever, store = _retriever()
    base = RuleBasedCoach().run(analysis, report)
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    payload = _payload(
        base.suggestions[0].suggestion_id,
        chunk_id,
        reasoning="Because you got the kill, this was the correct decision.",
    )
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.diagnostics.rejected_enrichments >= 1


def test_stale_dynamic_fact_is_rejected() -> None:
    analysis, report = _analysis_and_report()
    analysis = analysis.model_copy(update={
        "round_contexts": [RoundContextEvidence(round_id="round_001", money=300)],
    })
    document = load_knowledge_bytes(
        b"A version-sensitive purchase costs $300.",
        "economy.txt",
        metadata={
            "knowledge_layer": KnowledgeLayer.DYNAMIC_GAME_DATA.value,
            "rule_strength": RuleStrength.HARD_RULE.value,
            "version_sensitive": True,
            "last_verified": "2020-01-01",
            "source_urls": ["https://example.test/official"],
        },
    )
    store = InMemoryKnowledgeStore(ConstantEmbedding())
    index_documents([document], store, chunker=KnowledgeChunker(max_chars=400))
    retriever = KnowledgeRetriever(store, min_score=0.0)
    base = RuleBasedCoach().run(analysis, report)
    chunk_id = store.query("purchase", top_k=1)[0].chunk.chunk_id
    payload = _payload(base.suggestions[0].suggestion_id, chunk_id)
    run = EvidenceBoundRagCoach(retriever, FakeLLM(payload)).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.summary.generated_by == "rules"
