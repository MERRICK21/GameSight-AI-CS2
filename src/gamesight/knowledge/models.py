"""Typed knowledge documents, chunks and retrieval results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class KnowledgeLayer(StrEnum):
    """Four independently queryable CS2 knowledge bases."""

    GAME_RULES = "game_rules"
    TACTICAL_FUNDAMENTALS = "tactical_fundamentals"
    SITUATION_DECISIONS = "situation_decisions"
    DYNAMIC_GAME_DATA = "dynamic_game_data"


class RuleStrength(StrEnum):
    """How strongly a retrieved passage may be worded by the coach."""

    HARD_RULE = "hard_rule"
    STRATEGIC_PRINCIPLE = "strategic_principle"
    CONTEXTUAL_RECOMMENDATION = "contextual_recommendation"


class KnowledgeDocument(BaseModel):
    """One local source document before chunking."""

    document_id: str
    title: str
    source_uri: str
    content: str
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    """A stable, independently retrievable passage."""

    chunk_id: str
    document_id: str
    title: str
    source_uri: str
    content: str
    chunk_index: int
    heading: str | None = None
    language: str | None = None
    layer: KnowledgeLayer = KnowledgeLayer.TACTICAL_FUNDAMENTALS
    rule_strength: RuleStrength = RuleStrength.STRATEGIC_PRINCIPLE
    version_sensitive: bool = False
    last_verified: str | None = None
    effective_from: str | None = None
    expires_at: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedKnowledge(BaseModel):
    """One passage returned by semantic retrieval."""

    chunk: KnowledgeChunk
    score: float = Field(ge=-1.0, le=1.0)
