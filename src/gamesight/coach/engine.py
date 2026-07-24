"""Coaching engine — rule-based evidence analysis.

Produces ``CoachSuggestion`` objects from pipeline outputs.  The current
implementation is a deterministic rule engine.  The ``CoachEngine`` ABC
makes it straightforward to swap in an LLM-based engine in the future.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from gamesight.coach.models import CoachCategory, CoachSuggestion
from gamesight.domain.models import AnalysisResult, EventType, GameEvent, RoundAnalysis
from gamesight.reporting.models import EvidenceLink, MatchReport, RoundStats


class CoachEngine(ABC):
    """Abstract coaching engine — generate suggestions from analysis data."""

    @abstractmethod
    def generate(
        self,
        analysis: AnalysisResult,
        report: MatchReport,
    ) -> list[CoachSuggestion]:
        """Produce coaching suggestions from pipeline outputs."""


class RuleBasedCoach(CoachEngine):
    """Deterministic rule engine that generates coaching suggestions.

    Rules are based on observable patterns in the pipeline output:
    kill/death ratios, enemy encounter timing, round duration anomalies,
    and combat density.  Every suggestion includes evidence links back
    to specific frames and pipeline sources.

    The engine is intentionally modular: add or remove rules without
    affecting the interface.
    """

    def __init__(self) -> None:
        self._counter = 0

    def generate(
        self,
        analysis: AnalysisResult,
        report: MatchReport,
    ) -> list[CoachSuggestion]:
        self._counter = 0
        suggestions: list[CoachSuggestion] = []

        for ra, rr in zip(analysis.rounds, report.rounds):
            stats = rr.stats
            suggestions.extend(self._analyse_round(ra, stats, rr.findings))

        return suggestions

    # -- per-round rules -----------------------------------------------------

    def _analyse_round(
        self,
        ra: RoundAnalysis,
        stats: RoundStats,
        findings,
    ) -> list[CoachSuggestion]:
        result: list[CoachSuggestion] = []

        result.extend(self._check_death_heavy_round(ra, stats))
        result.extend(self._check_aggressive_round(ra, stats))
        result.extend(self._check_late_enemy_contact(ra, stats))
        result.extend(self._check_early_enemy_contact(ra, stats))
        result.extend(self._check_no_combat_round(ra, stats))
        result.extend(self._check_combat_density(ra, stats))

        return result

    # Rule: death-heavy round — more deaths than kills.
    def _check_death_heavy_round(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        if stats.deaths_detected <= 0 or stats.deaths_detected <= stats.kills_detected:
            return []

        death_events = [e for e in ra.events if e.event_type == EventType.PLAYER_DEATH]
        ts = death_events[0].start_sec if death_events else ra.start_sec

        return [self._make(
            f"death_heavy_{ra.round_id}",
            CoachCategory.POSITIONING,
            ra.round_id,
            ts,
            f"You died {stats.deaths_detected} time(s) but only secured {stats.kills_detected} kill(s) "
            f"in this round. This suggests your positioning may have been too aggressive or exposed.",
            "Try holding angles from cover and only peeking when you have an advantage. "
            "Consider using utility (flash/smoke) before peeking common angles.",
            0.75,
            [self._ev_link(e) for e in death_events],
        )]

    # Rule: aggressive/dominant round — many kills, no deaths.
    def _check_aggressive_round(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        if stats.kills_detected < 2 or stats.deaths_detected > 0:
            return []

        kill_events = [e for e in ra.events if e.event_type == EventType.PLAYER_KILL]
        ts = kill_events[0].start_sec if kill_events else ra.start_sec

        return [self._make(
            f"aggressive_{ra.round_id}",
            CoachCategory.AIM,
            ra.round_id,
            ts,
            f"Strong round! You secured {stats.kills_detected} kill(s) without dying. "
            "Your aim and engagement timing were effective.",
            "Keep up the good crosshair placement. Review your positioning after each kill "
            "to ensure you did not over-extend.",
            0.80,
            [self._ev_link(e) for e in kill_events],
        )]

    # Rule: late first enemy contact.
    def _check_late_enemy_contact(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        if stats.enemy_first_visible_sec is None:
            return []
        if stats.enemy_first_visible_sec < 20.0:
            return []
        if stats.duration_sec is None:
            return []

        efv_events = [e for e in ra.events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
        ts = efv_events[0].start_sec if efv_events else stats.enemy_first_visible_sec

        return [self._make(
            f"late_contact_{ra.round_id}",
            CoachCategory.GAME_SENSE,
            ra.round_id,
            ts,
            f"First enemy contact at {stats.enemy_first_visible_sec:.1f}s — "
            f"over {stats.enemy_first_visible_sec / stats.duration_sec * 100:.0f}% into the round. "
            "This may indicate overly passive play or poor map control.",
            "Try taking map control earlier. Use your utility to contest key areas "
            "in the first 15-20 seconds of the round.",
            0.65,
            [self._ev_link(e) for e in efv_events],
        )]

    # Rule: very early enemy contact.
    def _check_early_enemy_contact(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        if stats.enemy_first_visible_sec is None:
            return []
        if stats.enemy_first_visible_sec >= 8.0:
            return []

        efv_events = [e for e in ra.events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
        ts = efv_events[0].start_sec if efv_events else stats.enemy_first_visible_sec

        return [self._make(
            f"early_contact_{ra.round_id}",
            CoachCategory.GAME_SENSE,
            ra.round_id,
            ts,
            f"Enemy contact at {stats.enemy_first_visible_sec:.1f}s — very early in the round. "
            "You may be rushing into contested areas without adequate utility support.",
            "Consider using a flashbang or smoke before pushing into common contact points. "
            "Coordinate with teammates for trade potential.",
            0.70,
            [self._ev_link(e) for e in efv_events],
        )]

    # Rule: no combat in the round.
    def _check_no_combat_round(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        if stats.kills_detected > 0 or stats.deaths_detected > 0:
            return []
        if stats.duration_sec is None or stats.duration_sec < 30:
            return []

        return [self._make(
            f"no_combat_{ra.round_id}",
            CoachCategory.TEAMPLAY,
            ra.round_id,
            ra.start_sec,
            f"No kills or deaths detected in a {stats.duration_sec:.0f}s round. "
            "You may have been too passive or isolated from team engagements.",
            "Communicate with your team and position yourself where trades can happen. "
            "If playing a support/lurk role, ensure your timing aligns with team executes.",
            0.60,
            [],
        )]

    # Rule: high combat density — many kills in a short round.
    def _check_combat_density(
        self, ra: RoundAnalysis, stats: RoundStats
    ) -> list[CoachSuggestion]:
        total_combat = stats.kills_detected + stats.deaths_detected
        if total_combat < 2:
            return []
        if stats.duration_sec is None or stats.duration_sec <= 0:
            return []

        density = total_combat / stats.duration_sec * 60  # events per minute
        if density < 3.0:
            return []

        kill_events = [e for e in ra.events if e.event_type == EventType.PLAYER_KILL]
        ts = kill_events[0].start_sec if kill_events else ra.start_sec

        return [self._make(
            f"combat_density_{ra.round_id}",
            CoachCategory.POSITIONING,
            ra.round_id,
            ts,
            f"High combat density: {total_combat} engagements in {stats.duration_sec:.0f}s "
            f"({density:.1f} events/min). The round was very active.",
            "In high-density rounds, prioritise staying alive over chasing kills. "
            "Hold advantageous positions and let opponents make mistakes.",
            0.70,
            [self._ev_link(e) for e in kill_events],
        )]

    # -- helpers -------------------------------------------------------------

    def _make(
        self,
        suffix: str,
        category: CoachCategory,
        round_id: str,
        ts: float,
        reasoning: str,
        action: str,
        confidence: float,
        evidence: list[EvidenceLink],
    ) -> CoachSuggestion:
        self._counter += 1
        return CoachSuggestion(
            suggestion_id=f"{category.value}_{suffix}_{self._counter:02d}",
            category=category,
            round_id=round_id,
            timestamp_sec=round(ts, 1),
            reasoning=reasoning,
            action=action,
            confidence=confidence,
            evidence=evidence,
        )

    @staticmethod
    def _ev_link(event: GameEvent) -> EvidenceLink:
        ev = event.evidence[0] if event.evidence else None
        return EvidenceLink(
            frame_index=ev.frame_index if ev else None,
            timestamp_sec=event.start_sec,
            source=ev.source if ev else "RuleBasedCoach",
        )
