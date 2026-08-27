"""Bounded tool use, traceability and fallback for the single Replay Coach Agent."""

from __future__ import annotations

from pathlib import Path

from gamesight.agent.models import CoachToolName
from gamesight.agent.tools import AgentToolContext, CoachToolRegistry
from gamesight.coach.agent_engine import SingleAgentCoach
from gamesight.coach.engine import RuleBasedCoach
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


class SequentialLLM(JsonLLMClient):
    def __init__(self, responses: list[dict], *, available: bool = True) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self._available = available

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-agent"

    @property
    def available(self) -> bool:
        return self._available

    def generate_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return JsonGenerationResult(
            content=self.responses.pop(0),
            provider=self.provider,
            model=self.model,
            latency_ms=7,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


def _analysis_report_retriever():
    event = GameEvent(
        event_id="enemy_visible",
        event_type=EventType.ENEMY_FIRST_VISIBLE,
        start_sec=21.5,
        confidence=0.92,
        evidence=[Evidence(
            timestamp_sec=21.5,
            frame_index=645,
            asset_path="private/frame.png",
            source="native_enemy",
        )],
    )
    analysis = AnalysisResult(
        video=VideoInput(video_id="agent-test", path=Path("match.mp4")),
        metadata=VideoMetadata(duration_sec=50.0),
        rounds=[RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=50.0, events=[event],
        )],
        round_contexts=[RoundContextEvidence(
            round_id="round_001", player_side="ct", weapon="awp",
            weapon_categories=["sniper"], native_round_clock_sec=93.0,
        )],
        capabilities={"analysis_complete": True, "personal_combat": False},
    )
    report = EvidenceReportBuilder().build(analysis)
    document = load_knowledge_bytes(
        (
            "A contact should be reviewed using cover, crosshair placement and "
            "information available before the engagement. The outcome alone does "
            "not establish decision quality."
        ).encode(),
        "decision.txt",
    )
    store = InMemoryKnowledgeStore(ConstantEmbedding())
    index_documents([document], store, chunker=KnowledgeChunker(max_chars=400))
    return analysis, report, KnowledgeRetriever(store, min_score=0.0), store


def _final_payload(suggestion_id: str, chunk_id: str) -> dict:
    return {
        "suggestions": [{
            "source_suggestion_id": suggestion_id,
            "reasoning": "The observed contact defines a review window, not an automatic mistake.",
            "action": "Review cover and crosshair placement before the contact.",
            "knowledge_chunk_ids": [chunk_id],
            "evaluation_basis": "decision_quality",
        }],
        "summary": {
            "strengths": ["The replay contains auditable contact evidence."],
            "weaknesses": ["Decision context remains partly unavailable."],
            "practice_drills": ["Review the linked contact from available cover."],
            "focus_areas": ["Evidence-based contact review."],
            "overall_assessment": "The available evidence supports a bounded review.",
            "knowledge_chunk_ids": [chunk_id],
        },
    }


def test_single_agent_calls_tools_then_produces_validated_coaching() -> None:
    analysis, report, retriever, store = _analysis_report_retriever()
    base = RuleBasedCoach().run(analysis, report)
    suggestion_id = base.suggestions[0].suggestion_id
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    llm = SequentialLLM([
        {
            "status": "tool_calls",
            "tool_calls": [
                {"call_id": "c1", "tool_name": "list_coaching_candidates", "arguments": {}},
                {"call_id": "c2", "tool_name": "get_round_evidence", "arguments": {"round_id": "round_001"}},
                {"call_id": "c3", "tool_name": "get_decision_context", "arguments": {"round_id": "round_001"}},
            ],
            "final": None,
        },
        {
            "status": "tool_calls",
            "tool_calls": [{
                "call_id": "c4",
                "tool_name": "search_knowledge",
                "arguments": {
                    "suggestion_id": suggestion_id,
                    "query": "first contact cover decision quality",
                    "top_k": 2,
                },
            }],
            "final": None,
        },
        {"status": "final", "tool_calls": [], "final": _final_payload(suggestion_id, chunk_id)},
    ])
    run = SingleAgentCoach(retriever, llm).run(analysis, report)
    assert run.diagnostics.mode == "agent_llm"
    assert run.diagnostics.agent_iterations == 3
    assert run.diagnostics.agent_tool_calls == 4
    assert run.diagnostics.agent_tool_failures == 0
    assert run.diagnostics.total_tokens == 45
    assert run.suggestions[0].generated_by == "fake_agent"
    assert run.suggestions[0].evidence == base.suggestions[0].evidence
    assert run.suggestions[0].knowledge_citations[0].chunk_id == chunk_id
    assert "untrusted data" in llm.calls[0][0]
    assert "unlisted tool" in llm.calls[0][0]


def test_agent_cannot_finish_without_using_knowledge_tool() -> None:
    analysis, report, retriever, store = _analysis_report_retriever()
    base = RuleBasedCoach().run(analysis, report)
    suggestion_id = base.suggestions[0].suggestion_id
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    llm = SequentialLLM([
        {"status": "final", "tool_calls": [], "final": _final_payload(suggestion_id, chunk_id)},
    ])
    run = SingleAgentCoach(retriever, llm).run(analysis, report)
    assert run.diagnostics.mode == "rules"
    assert run.diagnostics.fallback_reason == "agent_error:final_without_knowledge_tool"
    assert run.diagnostics.agent_iterations == 1
    assert run.diagnostics.agent_stop_reason == "final_without_knowledge_tool"
    assert run.diagnostics.total_tokens == 15
    assert run.suggestions == base.suggestions


def test_agent_tool_registry_exposes_only_read_only_allowlist() -> None:
    names = {
        schema["tool_name"] for schema in CoachToolRegistry().schemas()
    }
    assert names == {item.value for item in CoachToolName}
    assert not names & {"shell", "filesystem", "network", "write_report", "modify_events"}


def test_round_evidence_tool_omits_private_asset_paths() -> None:
    analysis, report, retriever, _ = _analysis_report_retriever()
    base = RuleBasedCoach().run(analysis, report)
    context = AgentToolContext(
        analysis=analysis,
        report=report,
        base_run=base,
        selected=base.suggestions,
        retriever=retriever,
    )
    observation = CoachToolRegistry().execute(
        "round",
        CoachToolName.GET_ROUND_EVIDENCE,
        {"round_id": "round_001"},
        context,
    )
    assert observation.ok
    assert "asset_path" not in str(observation.data)
    assert "private/frame.png" not in str(observation.data)


def test_invalid_tool_arguments_are_reported_without_execution() -> None:
    analysis, report, retriever, _ = _analysis_report_retriever()
    base = RuleBasedCoach().run(analysis, report)
    context = AgentToolContext(
        analysis=analysis,
        report=report,
        base_run=base,
        selected=base.suggestions,
        retriever=retriever,
    )
    observation = CoachToolRegistry().execute(
        "bad",
        CoachToolName.GET_MATCH_OVERVIEW,
        {"unexpected": True},
        context,
    )
    assert not observation.ok
    assert observation.error == "invalid_tool_arguments"


def test_agent_output_still_passes_outcome_bias_gate() -> None:
    analysis, report, retriever, store = _analysis_report_retriever()
    base = RuleBasedCoach().run(analysis, report)
    suggestion_id = base.suggestions[0].suggestion_id
    chunk_id = store.query("contact", top_k=1)[0].chunk.chunk_id
    bad_final = _final_payload(suggestion_id, chunk_id)
    bad_final["suggestions"][0]["reasoning"] = (
        "Because you got the kill, this was the correct decision."
    )
    llm = SequentialLLM([
        {
            "status": "tool_calls",
            "tool_calls": [{
                "call_id": "search",
                "tool_name": "search_knowledge",
                "arguments": {
                    "suggestion_id": suggestion_id,
                    "query": "contact decision quality",
                    "top_k": 1,
                },
            }],
            "final": None,
        },
        {"status": "final", "tool_calls": [], "final": bad_final},
    ])
    run = SingleAgentCoach(retriever, llm).run(analysis, report)
    assert run.suggestions[0].generated_by == "rules"
    assert run.diagnostics.rejected_enrichments >= 1
    assert run.summary.generated_by == "fake_agent"
