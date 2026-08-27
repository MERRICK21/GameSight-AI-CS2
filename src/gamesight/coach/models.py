"""Evidence-based coaching suggestion models.

Every suggestion carries explicit evidence references so the coach
output is auditable, whether the engine is rule-based or LLM-powered.
"""

from __future__ import annotations

from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass

from pydantic import BaseModel, Field

from gamesight.reporting.models import EvidenceLink
from gamesight.knowledge.models import KnowledgeLayer, RuleStrength


class KnowledgeCitation(BaseModel):
    """A retrieved CS2 knowledge passage cited by generated coaching text."""

    chunk_id: str
    title: str
    source_uri: str
    heading: str | None = None
    score: float = Field(ge=-1.0, le=1.0)
    layer: KnowledgeLayer = KnowledgeLayer.TACTICAL_FUNDAMENTALS
    rule_strength: RuleStrength = RuleStrength.STRATEGIC_PRINCIPLE
    version_sensitive: bool = False
    last_verified: str | None = None


class CoachCategory(StrEnum):
    AIM = "aim"
    POSITIONING = "positioning"
    GAME_SENSE = "game_sense"
    UTILITY = "utility"
    ECONOMY = "economy"
    TEAMPLAY = "teamplay"


class CoachSuggestion(BaseModel):
    """One evidence-based coaching suggestion for a specific round."""

    suggestion_id: str
    category: CoachCategory
    round_id: str
    timestamp_sec: float
    reasoning: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    generated_by: str = "rules"
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)


class CoachSummary(BaseModel):
    """Post-match coaching summary with actionable practice recommendations.

    Generated after all per-round suggestions are collected.  Provides
    a high-level view of what the player did well, what needs work,
    and concrete drills to improve.
    """

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    practice_drills: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
    generated_by: str = "rules"
    knowledge_citations: list[KnowledgeCitation] = Field(default_factory=list)


class CoachDiagnostics(BaseModel):
    """Serializable RAG/LLM trace without prompts, secrets or source text."""

    mode: str = "rules"
    provider: str | None = None
    model: str | None = None
    fallback_reason: str | None = None
    knowledge_chunks: int = 0
    knowledge_layers: dict[str, int] = Field(default_factory=dict)
    retrieved_chunks: int = 0
    accepted_enrichments: int = 0
    rejected_enrichments: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CoachRun(BaseModel):
    suggestions: list[CoachSuggestion] = Field(default_factory=list)
    summary: CoachSummary = Field(default_factory=CoachSummary)
    diagnostics: CoachDiagnostics = Field(default_factory=CoachDiagnostics)
