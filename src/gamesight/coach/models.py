"""Evidence-based coaching suggestion models.

Every suggestion carries explicit evidence references so the coach
output is auditable, whether the engine is rule-based or LLM-powered.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from gamesight.reporting.models import EvidenceLink


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
