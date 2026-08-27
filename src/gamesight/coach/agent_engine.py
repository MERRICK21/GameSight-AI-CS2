"""Single-Agent replay coach using bounded, read-only tool calls."""

from __future__ import annotations

from dataclasses import dataclass

from gamesight.agent.runtime import (
    AgentExecutionError,
    AgentRuntimeConfig,
    ReplayCoachAgent,
)
from gamesight.agent.tools import AgentToolContext, CoachToolRegistry
from gamesight.coach.engine import RuleBasedCoach
from gamesight.coach.models import CoachDiagnostics, CoachRun
from gamesight.coach.rag_engine import EvidenceBoundRagCoach, RagCoachConfig
from gamesight.domain.models import AnalysisResult
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.llm.client import JsonLLMClient
from gamesight.reporting.models import MatchReport


@dataclass(frozen=True)
class AgentCoachConfig(RagCoachConfig):
    max_iterations: int = 3
    max_tool_calls: int = 12
    max_observation_chars: int = 48_000


class SingleAgentCoach(EvidenceBoundRagCoach):
    """Agent-ready orchestration without introducing a third-party framework."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        llm: JsonLLMClient,
        *,
        base_coach: RuleBasedCoach | None = None,
        locale: str = "en",
        config: AgentCoachConfig | None = None,
        registry: CoachToolRegistry | None = None,
    ) -> None:
        self.agent_config = config or AgentCoachConfig()
        super().__init__(
            retriever,
            llm,
            base_coach=base_coach,
            locale=locale,
            config=self.agent_config,
        )
        self.registry = registry or CoachToolRegistry()

    def run(self, analysis: AnalysisResult, report: MatchReport) -> CoachRun:
        base_run = self.base_coach.run(analysis, report)
        diagnostics = CoachDiagnostics(
            mode="rules",
            provider=self.llm.provider,
            model=self.llm.model,
            knowledge_chunks=self.retriever.chunk_count,
            knowledge_layers=self._knowledge_layer_counts(),
        )
        fallback = self._preflight_fallback(analysis, base_run)
        if fallback:
            diagnostics.fallback_reason = fallback
            return base_run.model_copy(update={"diagnostics": diagnostics})

        selected = sorted(
            base_run.suggestions,
            key=lambda item: (-item.confidence, item.timestamp_sec, item.suggestion_id),
        )[:self.agent_config.max_suggestions]
        tool_context = AgentToolContext(
            analysis=analysis,
            report=report,
            base_run=base_run,
            selected=selected,
            retriever=self.retriever,
        )
        agent = ReplayCoachAgent(
            self.llm,
            self.registry,
            locale=self.locale,
            config=AgentRuntimeConfig(
                max_iterations=self.agent_config.max_iterations,
                max_tool_calls=self.agent_config.max_tool_calls,
                max_observation_chars=self.agent_config.max_observation_chars,
            ),
        )
        try:
            result = agent.run(tool_context)
        except AgentExecutionError as exc:
            self._apply_trace(diagnostics, exc.trace)
            diagnostics.fallback_reason = f"agent_error:{exc.reason}"
            return base_run.model_copy(update={"diagnostics": diagnostics})

        trace = result.trace
        self._apply_trace(diagnostics, trace)
        diagnostics.retrieved_chunks = len(tool_context.unique_knowledge)

        # _apply_payload adds the final model call, so seed diagnostics with only
        # the earlier planning/tool turns to keep token totals exact.
        final = result.final_generation
        diagnostics.latency_ms = max(0, trace.latency_ms - final.latency_ms)
        diagnostics.prompt_tokens = max(
            0, trace.prompt_tokens - final.usage.prompt_tokens,
        )
        diagnostics.completion_tokens = max(
            0, trace.completion_tokens - final.usage.completion_tokens,
        )
        diagnostics.total_tokens = max(
            0, trace.total_tokens - final.usage.total_tokens,
        )
        knowledge = sorted(
            tool_context.unique_knowledge.values(),
            key=lambda hit: (-hit.score, hit.chunk.chunk_id),
        )[:self.agent_config.max_unique_chunks]
        return self._apply_payload(
            payload=result.payload,
            generated=final,
            base_run=base_run,
            report=report,
            analysis=analysis,
            retrieved_by_suggestion=tool_context.retrieved_by_suggestion,
            knowledge=knowledge,
            diagnostics=diagnostics,
            mode="agent_llm",
            generated_by_suffix="agent",
        )

    @staticmethod
    def _apply_trace(diagnostics: CoachDiagnostics, trace) -> None:
        diagnostics.agent_iterations = trace.iterations
        diagnostics.agent_tool_calls = trace.tool_calls
        diagnostics.agent_tool_failures = trace.tool_failures
        diagnostics.agent_tools = list(dict.fromkeys(trace.tool_names))
        diagnostics.agent_stop_reason = trace.stop_reason
        diagnostics.latency_ms = trace.latency_ms
        diagnostics.prompt_tokens = trace.prompt_tokens
        diagnostics.completion_tokens = trace.completion_tokens
        diagnostics.total_tokens = trace.total_tokens

    def _preflight_fallback(
        self,
        analysis: AnalysisResult,
        base_run: CoachRun,
    ) -> str | None:
        if not analysis.capabilities.get("analysis_complete", True):
            return "analysis_incomplete"
        if not self.llm.available:
            return "llm_unavailable"
        if not base_run.suggestions:
            return "no_evidence_suggestions"
        if self.retriever.chunk_count == 0:
            return "knowledge_index_empty"
        return None
