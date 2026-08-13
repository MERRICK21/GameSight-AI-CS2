"""Coaching engine — rule-based evidence analysis with post-match summary.

Produces ``CoachSuggestion`` objects and ``CoachSummary``.  Supports
i18n via an optional ``I18nLoader`` so that suggestions are generated
in the user's language.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gamesight.coach.models import CoachCategory, CoachSuggestion, CoachSummary
from gamesight.domain.models import AnalysisResult, EventType, GameEvent, RoundAnalysis
from gamesight.i18n.loader import I18nLoader
from gamesight.reporting.models import EvidenceLink, MatchReport, RoundStats


class CoachEngine(ABC):
    @abstractmethod
    def generate(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        """Produce per-round coaching suggestions."""

    @abstractmethod
    def summarize(self, suggestions: list[CoachSuggestion], analysis: AnalysisResult, report: MatchReport) -> CoachSummary:
        """Produce post-match summary."""


class RuleBasedCoach(CoachEngine):
    """Deterministic rule engine with i18n support.

    Pass an ``I18nLoader`` to generate suggestions in the target language.
    When ``loader`` is None, English is used.
    """

    def __init__(self, loader: I18nLoader | None = None) -> None:
        self._counter = 0
        self._t = loader or I18nLoader("en")

    def generate(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        self._counter = 0
        suggestions: list[CoachSuggestion] = []
        for ra, rr in zip(analysis.rounds, report.rounds):
            if report.overview.personal_combat_available:
                suggestions.extend(self._analyse_round(ra, rr.stats))
            else:
                suggestions.extend(self._analyse_first_person(ra, rr.stats))
        if not report.overview.personal_combat_available:
            return suggestions
        suggestions.extend(self._check_kd_trend(analysis, report))
        suggestions.extend(self._check_survival_pattern(analysis, report))
        suggestions.extend(self._check_momentum(analysis, report))
        suggestions.extend(self._check_round_consistency(analysis, report))
        return suggestions

    def summarize(self, suggestions: list[CoachSuggestion], analysis: AnalysisResult, report: MatchReport) -> CoachSummary:
        return self._build_summary(suggestions, analysis, report)

    # -- per-round rules -----------------------------------------------------

    def _analyse_round(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        result: list[CoachSuggestion] = []
        result.extend(self._check_death_heavy_round(ra, stats))
        result.extend(self._check_aggressive_round(ra, stats))
        result.extend(self._check_late_enemy_contact(ra, stats))
        result.extend(self._check_early_enemy_contact(ra, stats))
        result.extend(self._check_no_combat_round(ra, stats))
        result.extend(self._check_combat_density(ra, stats))
        return result

    def _analyse_first_person(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        """Coach only from viewport geometry; never from names or watermarks."""
        result: list[CoachSuggestion] = []
        engagements = [
            event for event in ra.events
            if event.event_type == EventType.ENGAGEMENT_CANDIDATE
        ]
        for index, event in enumerate(engagements[:3], start=1):
            elapsed = max(0.0, event.start_sec - ra.start_sec)
            result.append(self._make(
                f"engagement_{ra.round_id}_{index:02d}",
                CoachCategory.GAME_SENSE,
                ra.round_id,
                event.start_sec,
                self._t.t("coach_reasoning.engagement_window", time=elapsed),
                self._t.t("coach_action.engagement_review"),
                event.confidence,
                [self._ev_link(event)],
            ))
        moments = [
            event for event in ra.events
            if event.event_type == EventType.FIRST_PERSON_MOMENT
        ]
        for event in moments:
            kind = event.attributes.get("moment_kind")
            duration = float(event.attributes.get("duration_sec", 0.0))
            evidence = [self._ev_link(event)]
            if kind == "flash" and duration >= 2.0:
                result.append(self._make(
                    f"flash_{ra.round_id}", CoachCategory.UTILITY, ra.round_id,
                    event.start_sec,
                    self._t.t("coach_reasoning.flash_episode", seconds=duration),
                    self._t.t("coach_action.flash_exposure"), 0.84, evidence,
                ))
            elif kind == "scope" and duration >= 8.0:
                result.append(self._make(
                    f"scope_hold_{ra.round_id}", CoachCategory.POSITIONING,
                    ra.round_id, event.start_sec,
                    self._t.t("coach_reasoning.scope_episode", seconds=duration),
                    self._t.t("coach_action.scope_hold"), 0.82, evidence,
                ))
        return result

    def _check_death_heavy_round(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        if stats.deaths_detected <= 0 or stats.deaths_detected <= stats.kills_detected:
            return []
        death_events = [e for e in ra.events if e.event_type == EventType.PLAYER_DEATH]
        ts = death_events[0].start_sec if death_events else ra.start_sec
        return [self._make(f"death_heavy_{ra.round_id}", CoachCategory.POSITIONING, ra.round_id, ts,
            self._t.t("coach_reasoning.death_heavy", deaths=stats.deaths_detected, kills=stats.kills_detected),
            self._t.t("coach_action.death_heavy"), 0.75,
            [self._ev_link(e) for e in death_events])]

    def _check_aggressive_round(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        if stats.kills_detected < 2 or stats.deaths_detected > 0:
            return []
        kill_events = [e for e in ra.events if e.event_type == EventType.PLAYER_KILL]
        ts = kill_events[0].start_sec if kill_events else ra.start_sec
        return [self._make(f"aggressive_{ra.round_id}", CoachCategory.AIM, ra.round_id, ts,
            self._t.t("coach_reasoning.aggressive", kills=stats.kills_detected),
            self._t.t("coach_action.aggressive"), 0.80,
            [self._ev_link(e) for e in kill_events])]

    def _check_late_enemy_contact(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        if stats.enemy_first_visible_sec is None or stats.duration_sec is None:
            return []
        if stats.enemy_first_visible_sec < 20.0:
            return []
        pct = stats.enemy_first_visible_sec / stats.duration_sec * 100
        efv_events = [e for e in ra.events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
        ts = efv_events[0].start_sec if efv_events else stats.enemy_first_visible_sec
        return [self._make(f"late_contact_{ra.round_id}", CoachCategory.GAME_SENSE, ra.round_id, ts,
            self._t.t("coach_reasoning.late_contact", time=stats.enemy_first_visible_sec, pct=pct),
            self._t.t("coach_action.late_contact"), 0.65,
            [self._ev_link(e) for e in efv_events])]

    def _check_early_enemy_contact(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        if stats.enemy_first_visible_sec is None or stats.enemy_first_visible_sec >= 8.0:
            return []
        efv_events = [e for e in ra.events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
        ts = efv_events[0].start_sec if efv_events else stats.enemy_first_visible_sec
        return [self._make(f"early_contact_{ra.round_id}", CoachCategory.GAME_SENSE, ra.round_id, ts,
            self._t.t("coach_reasoning.early_contact", time=stats.enemy_first_visible_sec),
            self._t.t("coach_action.early_contact"), 0.70,
            [self._ev_link(e) for e in efv_events])]

    def _check_no_combat_round(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        if stats.kills_detected > 0 or stats.deaths_detected > 0:
            return []
        if stats.duration_sec is None or stats.duration_sec < 30:
            return []
        return [self._make(f"no_combat_{ra.round_id}", CoachCategory.TEAMPLAY, ra.round_id, ra.start_sec,
            self._t.t("coach_reasoning.no_combat", duration=stats.duration_sec),
            self._t.t("coach_action.no_combat"), 0.60, [])]

    def _check_combat_density(self, ra: RoundAnalysis, stats: RoundStats) -> list[CoachSuggestion]:
        total = stats.kills_detected + stats.deaths_detected
        if total < 2 or stats.duration_sec is None or stats.duration_sec <= 0:
            return []
        density = total / stats.duration_sec * 60
        if density < 3.0:
            return []
        kill_events = [e for e in ra.events if e.event_type == EventType.PLAYER_KILL]
        ts = kill_events[0].start_sec if kill_events else ra.start_sec
        return [self._make(f"combat_density_{ra.round_id}", CoachCategory.POSITIONING, ra.round_id, ts,
            self._t.t("coach_reasoning.combat_density", total=total, duration=stats.duration_sec, density=density),
            self._t.t("coach_action.combat_density"), 0.70,
            [self._ev_link(e) for e in kill_events])]

    # -- cross-round rules ---------------------------------------------------

    def _check_kd_trend(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        if len(report.rounds) < 3:
            return []
        mid = len(report.rounds) // 2
        fh = report.rounds[:mid]
        sh = report.rounds[mid:]
        fh_k = sum(r.stats.kills_detected for r in fh)
        fh_d = sum(r.stats.deaths_detected for r in fh)
        sh_k = sum(r.stats.kills_detected for r in sh)
        sh_d = sum(r.stats.deaths_detected for r in sh)
        er = fh_k / max(fh_d, 1)
        lr = sh_k / max(sh_d, 1)
        diff = lr - er
        ts = analysis.rounds[mid].start_sec
        if diff > 0.5:
            return [self._make("kd_improving", CoachCategory.AIM, "all", ts,
                self._t.t("coach_reasoning.kd_improving", early=er, late=lr),
                self._t.t("coach_action.kd_improving"), 0.75, [])]
        if diff < -0.5:
            return [self._make("kd_declining", CoachCategory.GAME_SENSE, "all", ts,
                self._t.t("coach_reasoning.kd_declining", early=er, late=lr),
                self._t.t("coach_action.kd_declining"), 0.70, [])]
        return []

    def _check_survival_pattern(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        early_deaths = 0
        total = 0
        for ra in analysis.rounds:
            if ra.end_sec is None:
                continue
            total += 1
            for e in ra.events:
                if e.event_type == EventType.PLAYER_DEATH and (e.start_sec - ra.start_sec) < 30.0:
                    early_deaths += 1
        if total < 3 or early_deaths < 2:
            return []
        return [self._make("early_deaths", CoachCategory.POSITIONING, "all", analysis.rounds[0].start_sec,
            self._t.t("coach_reasoning.early_deaths", count=early_deaths, total=total, pct=early_deaths/total*100),
            self._t.t("coach_action.early_deaths"), 0.80, [])]

    def _check_momentum(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        if len(report.rounds) < 3:
            return []
        outcomes = []
        for r in report.rounds:
            s = r.stats
            outcomes.append("win" if s.kills_detected > s.deaths_detected else "loss")
        loss_streak = 0
        max_streak = 0
        for o in outcomes:
            if o == "loss":
                loss_streak += 1
                max_streak = max(max_streak, loss_streak)
            else:
                loss_streak = 0
        if max_streak >= 3:
            return [self._make("loss_streak", CoachCategory.GAME_SENSE, "all",
                analysis.rounds[-max_streak].start_sec,
                self._t.t("coach_reasoning.loss_streak", streak=max_streak),
                self._t.t("coach_action.loss_streak"), 0.75, [])]
        return []

    def _check_round_consistency(self, analysis: AnalysisResult, report: MatchReport) -> list[CoachSuggestion]:
        if len(report.rounds) < 3:
            return []
        kills = [r.stats.kills_detected for r in report.rounds]
        avg_k = sum(kills) / len(kills)
        max_dev = max(abs(k - avg_k) for k in kills)
        if max_dev >= 2 and avg_k > 0:
            return [self._make("inconsistent", CoachCategory.AIM, "all", analysis.rounds[0].start_sec,
                self._t.t("coach_reasoning.inconsistent", min=min(kills), max=max(kills), avg=avg_k),
                self._t.t("coach_action.inconsistent"), 0.70, [])]
        return []

    # -- summary -------------------------------------------------------------

    def _build_summary(self, suggestions: list[CoachSuggestion], analysis: AnalysisResult, report: MatchReport) -> CoachSummary:
        if not report.overview.personal_combat_available:
            rounds = report.overview.total_rounds
            has_flash = any("flash_" in item.suggestion_id for item in suggestions)
            has_scope = any("scope_hold_" in item.suggestion_id for item in suggestions)
            has_engagement = any("engagement_" in item.suggestion_id for item in suggestions)
            weaknesses = [self._t.t("combat_unavailable.weakness")]
            drills = [self._t.t("combat_unavailable.drill")]
            focus = [self._t.t("combat_unavailable.focus")]
            if has_flash:
                weaknesses.append(self._t.t("first_person_summary.flash_weakness"))
                drills.append(self._t.t("first_person_summary.flash_drill"))
            if has_scope:
                focus.append(self._t.t("first_person_summary.scope_focus"))
            if has_engagement:
                strengths.append(self._t.t("first_person_summary.engagement_strength"))
                focus.append(self._t.t("first_person_summary.engagement_focus"))
            return CoachSummary(
                strengths=[self._t.t("combat_unavailable.strength", rounds=rounds)],
                weaknesses=weaknesses,
                practice_drills=drills,
                focus_areas=focus,
                overall_assessment=self._t.t("combat_unavailable.assessment", rounds=rounds),
            )
        strengths: list[str] = []
        weaknesses: list[str] = []
        drills: list[str] = []
        focus: list[str] = []
        ov = report.overview
        total_k = ov.total_kills_detected
        total_d = ov.total_deaths_detected
        total_r = max(ov.total_rounds, 1)
        kd_ratio = total_k / max(total_d, 1)

        if kd_ratio >= 1.5:
            strengths.append(f"K/D ratio strong ({kd_ratio:.1f}) -- aim and engagement selection above average.")
        elif kd_ratio < 0.8:
            weaknesses.append(f"K/D ratio below 1.0 ({kd_ratio:.1f}) -- focus on survival and smarter peeks.")
        else:
            strengths.append(f"Balanced K/D ratio ({kd_ratio:.1f}) -- you trade effectively.")

        cats: dict[str, int] = {}
        for s in suggestions:
            cats[s.category.value] = cats.get(s.category.value, 0) + 1
        if cats.get("aim", 0) >= 2:
            strengths.append("Aim is a standout -- multiple strong shooting rounds detected.")
        if cats.get("positioning", 0) >= 2:
            weaknesses.append("Positioning needs work -- frequently caught in disadvantageous angles.")
            drills.append("Practice counter-strafing and jiggle-peeking 10 min/day in a local server.")
            focus.append("Improve positioning -- hold off-angles and use cover effectively.")
        if cats.get("game_sense", 0) >= 2:
            weaknesses.append("Game sense flagged -- contact timing or map awareness could improve.")
            drills.append("Watch 1-2 pro POV demos per week; note timing and pathing decisions.")
            focus.append("Develop game sense -- learn timing and pre-aim spots on each map.")
        if cats.get("teamplay", 0) >= 1:
            weaknesses.append("Teamplay could improve -- you had rounds with no engagement.")
            focus.append("Coordinate with team -- communicate position and intentions clearly.")
        if any("early_deaths" in s.suggestion_id for s in suggestions):
            weaknesses.append("You tend to die early -- work on opening survival.")
            drills.append("Practice safe pathing: walk each map and identify safe routes.")
        if any("inconsistent" in s.suggestion_id for s in suggestions):
            weaknesses.append("Round-to-round consistency is an issue.")
            drills.append("Warm up with 10 min deathmatch before every competitive session.")
        if any("streak" in s.suggestion_id for s in suggestions):
            weaknesses.append("Prone to loss streaks -- practice mental reset techniques.")
            focus.append("Mental resilience -- after 2 losses, take a breath and reset your approach.")

        if len(strengths) >= len(weaknesses):
            assessment = f"Across {total_r} rounds, you showed solid fundamentals. To reach the next level, focus on {' and '.join(focus[:2])}."
        else:
            assessment = f"Across {total_r} rounds, there is clear room for growth. Primary development areas: {' and '.join(focus[:2])}."

        return CoachSummary(
            strengths=strengths or ["Continue recording matches for deeper analysis."],
            weaknesses=weaknesses or ["No major weaknesses flagged -- solid all-around play."],
            practice_drills=drills or ["Continue playing and recording matches."],
            focus_areas=focus or ["Build consistency across all aspects of play."],
            overall_assessment=assessment,
        )

    # -- helpers -------------------------------------------------------------

    def _make(self, suffix, category, round_id, ts, reasoning, action, confidence, evidence) -> CoachSuggestion:
        self._counter += 1
        return CoachSuggestion(
            suggestion_id=f"{category.value}_{suffix}_{self._counter:02d}",
            category=category, round_id=round_id, timestamp_sec=round(ts, 1),
            reasoning=reasoning, action=action, confidence=confidence, evidence=evidence,
        )

    @staticmethod
    def _ev_link(event: GameEvent) -> EvidenceLink:
        ev = event.evidence[0] if event.evidence else None
        return EvidenceLink(
            frame_index=ev.frame_index if ev else None,
            timestamp_sec=event.start_sec,
            source=ev.source if ev else "RuleBasedCoach",
        )
