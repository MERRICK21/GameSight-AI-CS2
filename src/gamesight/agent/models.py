"""Typed protocol for the bounded single Replay Coach Agent."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class CoachToolName(StrEnum):
    GET_MATCH_OVERVIEW = "get_match_overview"
    LIST_COACHING_CANDIDATES = "list_coaching_candidates"
    GET_ROUND_EVIDENCE = "get_round_evidence"
    GET_DECISION_CONTEXT = "get_decision_context"
    SEARCH_KNOWLEDGE = "search_knowledge"


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    tool_name: CoachToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentTurn(BaseModel):
    """One model decision: request tools or return the final coaching payload."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["tool_calls", "final"]
    tool_calls: list[AgentToolCall] = Field(default_factory=list, max_length=8)
    final: dict[str, Any] | None = None


class ToolObservation(BaseModel):
    call_id: str
    tool_name: CoachToolName
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AgentTrace(BaseModel):
    iterations: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    tool_names: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_config = ConfigDict(extra="forbid")
