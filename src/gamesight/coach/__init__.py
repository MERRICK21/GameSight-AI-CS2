"""AI Coach module for GameSight — evidence-based coaching suggestions."""

from gamesight.coach.engine import CoachEngine, RuleBasedCoach
from gamesight.coach.models import CoachCategory, CoachSuggestion, CoachSummary

__all__ = ["CoachCategory", "CoachEngine", "CoachSuggestion", "CoachSummary", "RuleBasedCoach"]
