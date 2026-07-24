"""Tests for AI Coach engine and models."""

from pathlib import Path
from unittest import TestCase

from gamesight.coach.engine import RuleBasedCoach
from gamesight.coach.models import CoachCategory, CoachSuggestion, CoachSummary
from gamesight.domain.models import (
    AnalysisResult,
    EventType,
    Evidence,
    GameEvent,
    RoundAnalysis,
    VideoInput,
    VideoMetadata,
)
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.reporting.models import EvidenceLink


def _event(etype: EventType, ts: float, source: str = "test", frame: int | None = None) -> GameEvent:
    return GameEvent(
        event_id=f"{etype.value}_test",
        event_type=etype,
        start_sec=ts,
        confidence=0.9,
        evidence=[Evidence(timestamp_sec=ts, frame_index=frame, source=source)],
    )


class CoachSuggestionModelTests(TestCase):
    def test_minimal_construction(self) -> None:
        s = CoachSuggestion(suggestion_id="aim_001", category=CoachCategory.AIM,
                            round_id="round_001", timestamp_sec=5.0,
                            reasoning="Good.", action="Keep.", confidence=0.8)
        self.assertEqual(s.category, CoachCategory.AIM)

    def test_with_evidence(self) -> None:
        s = CoachSuggestion(suggestion_id="pos_001", category=CoachCategory.POSITIONING,
                            round_id="round_001", timestamp_sec=10.0,
                            reasoning="Exposed.", action="Cover.", confidence=0.7,
                            evidence=[EvidenceLink(timestamp_sec=10.0, source="test")])
        self.assertEqual(len(s.evidence), 1)


class CoachSummaryModelTests(TestCase):
    def test_default_construction(self) -> None:
        cs = CoachSummary()
        self.assertEqual(cs.strengths, [])
        self.assertEqual(cs.weaknesses, [])
        self.assertEqual(cs.practice_drills, [])
        self.assertEqual(cs.focus_areas, [])
        self.assertEqual(cs.overall_assessment, "")

    def test_full_construction(self) -> None:
        cs = CoachSummary(
            strengths=["Good aim"],
            weaknesses=["Poor positioning"],
            practice_drills=["DM 10 min/day"],
            focus_areas=["Improve peeking"],
            overall_assessment="Solid fundamentals.",
        )
        self.assertEqual(len(cs.strengths), 1)
        self.assertEqual(len(cs.weaknesses), 1)


class RuleBasedCoachTests(TestCase):
    def setUp(self) -> None:
        self.coach = RuleBasedCoach()
        self.builder = EvidenceReportBuilder()

    def _analysis(self, rounds: list[RoundAnalysis]) -> AnalysisResult:
        return AnalysisResult(
            video=VideoInput(video_id="test", path=Path("x.mp4")),
            metadata=VideoMetadata(), rounds=rounds,
        )

    def _report(self, analysis: AnalysisResult):
        return self.builder.build(analysis)

    # Per-round rules
    def test_empty_analysis(self) -> None:
        a = self._analysis([])
        s = self.coach.generate(a, self._report(a))
        self.assertEqual(s, [])

    def test_death_heavy_round(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0),
            _event(EventType.PLAYER_DEATH, 30.0), _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.POSITIONING for x in s))

    def test_aggressive_round(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_KILL, 10.0),
            _event(EventType.PLAYER_KILL, 25.0), _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.AIM for x in s))

    def test_late_enemy_contact(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=90.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.ENEMY_FIRST_VISIBLE, 45.0),
            _event(EventType.ROUND_END, 90.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.GAME_SENSE for x in s))

    def test_early_enemy_contact(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.ENEMY_FIRST_VISIBLE, 3.0),
            _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.GAME_SENSE for x in s))

    def test_no_combat_round(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=80.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.ROUND_END, 80.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.TEAMPLAY for x in s))

    def test_combat_density(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_KILL, 10.0),
            _event(EventType.PLAYER_KILL, 20.0), _event(EventType.PLAYER_DEATH, 30.0),
            _event(EventType.PLAYER_KILL, 40.0), _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any(x.category == CoachCategory.POSITIONING for x in s))

    # Cross-round rules
    def test_kd_trend_improving(self) -> None:
        r1 = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0), _event(EventType.ROUND_END, 60.0)])
        r2 = RoundAnalysis(round_id="r2", start_sec=70.0, end_sec=130.0, events=[
            _event(EventType.ROUND_START, 70.0), _event(EventType.PLAYER_KILL, 90.0), _event(EventType.ROUND_END, 130.0)])
        r3 = RoundAnalysis(round_id="r3", start_sec=140.0, end_sec=200.0, events=[
            _event(EventType.ROUND_START, 140.0), _event(EventType.PLAYER_KILL, 160.0),
            _event(EventType.PLAYER_KILL, 180.0), _event(EventType.ROUND_END, 200.0)])
        a = self._analysis([r1, r2, r3])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any("improving" in x.suggestion_id for x in s))

    def test_survival_pattern_early_deaths(self) -> None:
        rounds = []
        for i in range(4):
            rounds.append(RoundAnalysis(round_id=f"r{i+1}", start_sec=i*70.0, end_sec=i*70+60.0, events=[
                _event(EventType.ROUND_START, i*70.0),
                _event(EventType.PLAYER_DEATH, i*70.0 + 10.0),  # dies 10s in
                _event(EventType.ROUND_END, i*70+60.0),
            ]))
        a = self._analysis(rounds)
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any("early_deaths" in x.suggestion_id for x in s))

    def test_round_consistency(self) -> None:
        r1 = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_KILL, 10.0),
            _event(EventType.PLAYER_KILL, 20.0), _event(EventType.PLAYER_KILL, 30.0), _event(EventType.ROUND_END, 60.0)])
        r2 = RoundAnalysis(round_id="r2", start_sec=70.0, end_sec=130.0, events=[
            _event(EventType.ROUND_START, 70.0), _event(EventType.PLAYER_DEATH, 90.0), _event(EventType.ROUND_END, 130.0)])
        r3 = RoundAnalysis(round_id="r3", start_sec=140.0, end_sec=200.0, events=[
            _event(EventType.ROUND_START, 140.0), _event(EventType.PLAYER_KILL, 160.0),
            _event(EventType.PLAYER_KILL, 170.0), _event(EventType.PLAYER_KILL, 180.0), _event(EventType.ROUND_END, 200.0)])
        a = self._analysis([r1, r2, r3])
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any("inconsistent" in x.suggestion_id for x in s))

    def test_momentum_loss_streak(self) -> None:
        rounds = []
        for i in range(4):
            rounds.append(RoundAnalysis(round_id=f"r{i+1}", start_sec=i*70.0, end_sec=i*70+60.0, events=[
                _event(EventType.ROUND_START, i*70.0),
                _event(EventType.PLAYER_DEATH, i*70.0 + 20.0),
                _event(EventType.ROUND_END, i*70+60.0),
            ]))
        a = self._analysis(rounds)
        s = self.coach.generate(a, self._report(a))
        self.assertTrue(any("streak" in x.suggestion_id for x in s))

    # Summarize
    def test_summarize_returns_coach_summary(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_KILL, 20.0),
            _event(EventType.PLAYER_KILL, 40.0), _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        r = self._report(a)
        suggestions = self.coach.generate(a, r)
        summary = self.coach.summarize(suggestions, a, r)
        self.assertIsInstance(summary, CoachSummary)
        self.assertTrue(len(summary.overall_assessment) > 0)

    def test_summarize_has_strengths_or_default(self) -> None:
        a = self._analysis([])
        r = self._report(a)
        summary = self.coach.summarize([], a, r)
        self.assertTrue(len(summary.strengths) > 0)
        self.assertTrue(len(summary.practice_drills) > 0)

    def test_suggestions_unique_ids(self) -> None:
        r1 = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0), _event(EventType.ROUND_END, 60.0)])
        r2 = RoundAnalysis(round_id="r2", start_sec=70.0, end_sec=130.0, events=[
            _event(EventType.ROUND_START, 70.0), _event(EventType.PLAYER_KILL, 90.0),
            _event(EventType.PLAYER_KILL, 100.0), _event(EventType.ROUND_END, 130.0)])
        a = self._analysis([r1, r2])
        s = self.coach.generate(a, self._report(a))
        ids = [x.suggestion_id for x in s]
        self.assertEqual(len(ids), len(set(ids)))

    def test_confidence_in_range(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0),
            _event(EventType.PLAYER_KILL, 30.0), _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        for s in self.coach.generate(a, self._report(a)):
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)
