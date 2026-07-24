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
    """One evidence-based coaching suggestion for a specific round.

    Parameters
    ----------
    suggestion_id:
        Unique identifier (e.g. ``"aim_round_001_01"``).
    category:
        Coaching category.
    round_id:
        Round this suggestion applies to.
    timestamp_sec:
        Approximate timestamp in the video.
    reasoning:
        Why this suggestion is being made — grounded in pipeline evidence.
    action:
        Concrete recommended action the player should take.
    confidence:
        Confidence score (0-1).
    evidence:
        Evidence links backing this suggestion.
    """

    suggestion_id: str
    category: CoachCategory
    round_id: str
    timestamp_sec: float
    reasoning: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceLink] = Field(default_factory=list)
