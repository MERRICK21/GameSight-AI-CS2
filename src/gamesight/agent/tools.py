"""Read-only, schema-validated tools exposed to the Replay Coach Agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gamesight.agent.models import CoachToolName, ToolObservation
from gamesight.domain.models import AnalysisResult, EventType, RoundContextEvidence
from gamesight.knowledge.models import RetrievedKnowledge
from gamesight.knowledge.retriever import KnowledgeRetriever
from gamesight.knowledge.routing import DecisionContext
from gamesight.reporting.models import MatchReport

if TYPE_CHECKING:
    from gamesight.coach.models import CoachRun, CoachSuggestion


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArguments(ToolArguments):
    pass


class RoundArguments(ToolArguments):
    round_id: str = Field(min_length=1, max_length=64)


class SearchKnowledgeArguments(ToolArguments):
    suggestion_id: str = Field(min_length=1, max_length=160)
    query: str = Field(min_length=3, max_length=800)
    top_k: int = Field(default=3, ge=1, le=4)


@dataclass
class AgentToolContext:
    analysis: AnalysisResult
    report: MatchReport
    base_run: CoachRun
    selected: list[CoachSuggestion]
    retriever: KnowledgeRetriever
    retrieved_by_suggestion: dict[str, list[RetrievedKnowledge]] = field(default_factory=dict)
    unique_knowledge: dict[str, RetrievedKnowledge] = field(default_factory=dict)

    @property
    def candidates(self) -> dict[str, CoachSuggestion]:
        return {item.suggestion_id: item for item in self.selected}

    def round_context(self, round_id: str) -> RoundContextEvidence | None:
        return next(
            (item for item in self.analysis.round_contexts if item.round_id == round_id),
            None,
        )

    def decision_context(self, round_id: str) -> DecisionContext:
        context = self.round_context(round_id)
        round_ = next(
            (item for item in self.analysis.rounds if item.round_id == round_id),
            None,
        )
        bomb_state = None
        if round_ is not None:
            event_types = [event.event_type for event in round_.events]
            if EventType.BOMB_DEFUSED in event_types:
                bomb_state = "defused"
            elif EventType.BOMB_PLANTED in event_types:
                bomb_state = "planted"
        return DecisionContext(
            side=context.player_side if context else None,
            bomb_state=bomb_state,
            weapon=context.weapon if context else None,
            time_remaining_sec=context.native_round_clock_sec if context else None,
            money=context.money if context else None,
            utility=context.utility if context else [],
            map_name=context.map_name if context else None,
            map_position=context.map_position if context else None,
        )


class CoachTool(ABC):
    name: CoachToolName
    description: str
    arguments_model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        return {
            "tool_name": self.name.value,
            "description": self.description,
            "arguments_schema": self.arguments_model.model_json_schema(),
        }

    def execute(
        self,
        call_id: str,
        arguments: dict[str, Any],
        context: AgentToolContext,
    ) -> ToolObservation:
        try:
            parsed = self.arguments_model.model_validate(arguments)
            data = self._execute(parsed, context)
            return ToolObservation(
                call_id=call_id, tool_name=self.name, ok=True, data=data,
            )
        except ValidationError:
            return ToolObservation(
                call_id=call_id,
                tool_name=self.name,
                ok=False,
                error="invalid_tool_arguments",
            )
        except LookupError as exc:
            return ToolObservation(
                call_id=call_id,
                tool_name=self.name,
                ok=False,
                error=str(exc)[:160],
            )

    @abstractmethod
    def _execute(self, arguments: BaseModel, context: AgentToolContext) -> dict[str, Any]:
        pass


class GetMatchOverviewTool(CoachTool):
    name = CoachToolName.GET_MATCH_OVERVIEW
    description = "Return the deterministic match overview and capability flags."
    arguments_model = EmptyArguments

    def _execute(self, arguments: EmptyArguments, context: AgentToolContext) -> dict[str, Any]:
        return {
            "overview": context.report.overview.model_dump(mode="json"),
            "capabilities": context.analysis.capabilities,
            "round_count": len(context.analysis.rounds),
        }


class ListCoachingCandidatesTool(CoachTool):
    name = CoachToolName.LIST_COACHING_CANDIDATES
    description = "List immutable rule-based coaching candidates eligible for enrichment."
    arguments_model = EmptyArguments

    def _execute(self, arguments: EmptyArguments, context: AgentToolContext) -> dict[str, Any]:
        return {"candidates": [
            {
                "suggestion_id": item.suggestion_id,
                "round_id": item.round_id,
                "timestamp_sec": item.timestamp_sec,
                "category": item.category.value,
                "reasoning": item.reasoning,
                "action": item.action,
                "confidence": item.confidence,
            }
            for item in context.selected
        ]}


class GetRoundEvidenceTool(CoachTool):
    name = CoachToolName.GET_ROUND_EVIDENCE
    description = "Return bounded, already-detected events and evidence for one round."
    arguments_model = RoundArguments

    def _execute(self, arguments: RoundArguments, context: AgentToolContext) -> dict[str, Any]:
        round_ = next(
            (item for item in context.analysis.rounds if item.round_id == arguments.round_id),
            None,
        )
        if round_ is None:
            raise LookupError("unknown_round_id")
        return {
            "round_id": round_.round_id,
            "start_sec": round_.start_sec,
            "end_sec": round_.end_sec,
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "start_sec": event.start_sec,
                    "end_sec": event.end_sec,
                    "confidence": event.confidence,
                    "attributes": event.attributes,
                    "evidence": [
                        {
                            "frame_index": evidence.frame_index,
                            "timestamp_sec": evidence.timestamp_sec,
                            "source": evidence.source,
                        }
                        for evidence in event.evidence[:4]
                    ],
                }
                for event in round_.events[:40]
            ],
        }


class GetDecisionContextTool(CoachTool):
    name = CoachToolName.GET_DECISION_CONTEXT
    description = "Return known decision context and an explicit list of unavailable fields."
    arguments_model = RoundArguments

    def _execute(self, arguments: RoundArguments, context: AgentToolContext) -> dict[str, Any]:
        if not any(item.round_id == arguments.round_id for item in context.analysis.rounds):
            raise LookupError("unknown_round_id")
        decision = context.decision_context(arguments.round_id)
        values = decision.model_dump(mode="json")
        unavailable = [
            key for key, value in values.items()
            if value is None or value == []
        ]
        return {
            "round_id": arguments.round_id,
            "known": {key: value for key, value in values.items() if key not in unavailable},
            "unavailable": unavailable,
        }


class SearchKnowledgeTool(CoachTool):
    name = CoachToolName.SEARCH_KNOWLEDGE
    description = (
        "Search the four CS2 knowledge layers for one immutable coaching candidate. "
        "The suggestion_id binds retrieved citations to that candidate."
    )
    arguments_model = SearchKnowledgeArguments

    def _execute(
        self,
        arguments: SearchKnowledgeArguments,
        context: AgentToolContext,
    ) -> dict[str, Any]:
        suggestion = context.candidates.get(arguments.suggestion_id)
        if suggestion is None:
            raise LookupError("unknown_suggestion_id")
        decision = context.decision_context(suggestion.round_id)
        hits = context.retriever.retrieve_for_decision(
            arguments.query,
            decision,
            top_k=arguments.top_k,
        )
        combined = {
            hit.chunk.chunk_id: hit
            for hit in context.retrieved_by_suggestion.get(arguments.suggestion_id, [])
        }
        for hit in hits:
            existing = combined.get(hit.chunk.chunk_id)
            if existing is None or hit.score > existing.score:
                combined[hit.chunk.chunk_id] = hit
        context.retrieved_by_suggestion[arguments.suggestion_id] = sorted(
            combined.values(), key=lambda hit: (-hit.score, hit.chunk.chunk_id),
        )[:8]
        for hit in hits:
            current = context.unique_knowledge.get(hit.chunk.chunk_id)
            if current is None or hit.score > current.score:
                context.unique_knowledge[hit.chunk.chunk_id] = hit
        return {
            "suggestion_id": arguments.suggestion_id,
            "round_id": suggestion.round_id,
            "results": [
                {
                    "knowledge_chunk_id": hit.chunk.chunk_id,
                    "title": hit.chunk.title,
                    "heading": hit.chunk.heading,
                    "knowledge_layer": hit.chunk.layer.value,
                    "rule_strength": hit.chunk.rule_strength.value,
                    "version_sensitive": hit.chunk.version_sensitive,
                    "last_verified": hit.chunk.last_verified,
                    "source_uri": hit.chunk.source_uri,
                    "score": round(hit.score, 4),
                    "content": hit.chunk.content,
                    "exceptions": hit.chunk.exceptions,
                }
                for hit in hits
            ],
        }


class CoachToolRegistry:
    """Fixed allowlist; no filesystem, shell, network or mutation tools exist."""

    def __init__(self) -> None:
        tools: list[CoachTool] = [
            GetMatchOverviewTool(),
            ListCoachingCandidatesTool(),
            GetRoundEvidenceTool(),
            GetDecisionContextTool(),
            SearchKnowledgeTool(),
        ]
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(
        self,
        call_id: str,
        tool_name: CoachToolName,
        arguments: dict[str, Any],
        context: AgentToolContext,
    ) -> ToolObservation:
        return self._tools[tool_name].execute(call_id, arguments, context)
