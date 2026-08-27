"""AI Coach module for GameSight — evidence-based coaching suggestions."""

from gamesight.coach.engine import CoachEngine, RuleBasedCoach
from gamesight.coach.models import (
    CoachCategory,
    CoachDiagnostics,
    CoachRun,
    CoachSuggestion,
    CoachSummary,
    KnowledgeCitation,
)
from gamesight.coach.rag_engine import EvidenceBoundRagCoach, RagCoachConfig

__all__ = [
    "CoachCategory",
    "CoachDiagnostics",
    "CoachEngine",
    "CoachRun",
    "CoachSuggestion",
    "CoachSummary",
    "EvidenceBoundRagCoach",
    "KnowledgeCitation",
    "RagCoachConfig",
    "RuleBasedCoach",
]
