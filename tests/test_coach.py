"""Tests for AI Coach engine and models."""

from pathlib import Path
from unittest import TestCase

from gamesight.coach.engine import RuleBasedCoach
from gamesight.coach.models import CoachCategory, CoachSuggestion
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
from gamesight.reporting.models import EvidenceLink, FindingCategory, FindingSeverity, ReportFinding


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
        s = CoachSuggestion(
            suggestion_id="aim_001",
            category=CoachCategory.AIM,
            round_id="round_001",
            timestamp_sec=5.0,
            reasoning="Good crosshair placement.",
            action="Keep it up.",
            confidence=0.8,
        )
        self.assertEqual(s.category, CoachCategory.AIM)
        self.assertEqual(s.round_id, "round_001")

    def test_with_evidence(self) -> None:
        s = CoachSuggestion(
            suggestion_id="pos_001",
            category=CoachCategory.POSITIONING,
            round_id="round_001",
            timestamp_sec=10.0,
            reasoning="Exposed.",
            action="Use cover.",
            confidence=0.7,
            evidence=[EvidenceLink(timestamp_sec=10.0, source="test")],
        )
        self.assertEqual(len(s.evidence), 1)

    def test_categories_enum(self) -> None:
        self.assertEqual(CoachCategory.AIM.value, "aim")
        self.assertEqual(CoachCategory.POSITIONING.value, "positioning")
        self.assertEqual(CoachCategory.GAME_SENSE.value, "game_sense")


class RuleBasedCoachTests(TestCase):
    def setUp(self) -> None:
        self.coach = RuleBasedCoach()
        self.builder = EvidenceReportBuilder()

    def _build_analysis(self, rounds: list[RoundAnalysis]) -> AnalysisResult:
        return AnalysisResult(
            video=VideoInput(video_id="test", path=Path("x.mp4")),
            metadata=VideoMetadata(),
            rounds=rounds,
        )

    def test_empty_analysis(self) -> None:
        analysis = self._build_analysis([])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertEqual(suggestions, [])

    def test_death_heavy_round(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_DEATH, 15.0, frame=150),
                _event(EventType.PLAYER_DEATH, 30.0, frame=300),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(len(suggestions) > 0)
        self.assertEqual(suggestions[0].category, CoachCategory.POSITIONING)

    def test_aggressive_round(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 10.0),
                _event(EventType.PLAYER_KILL, 25.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(any(s.category == CoachCategory.AIM for s in suggestions))

    def test_late_enemy_contact(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=90.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 45.0, frame=450),
                _event(EventType.ROUND_END, 90.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(any(s.category == CoachCategory.GAME_SENSE for s in suggestions))

    def test_early_enemy_contact(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ENEMY_FIRST_VISIBLE, 3.0, frame=30),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(any(s.category == CoachCategory.GAME_SENSE for s in suggestions))

    def test_no_combat_round(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=80.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.ROUND_END, 80.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(any(s.category == CoachCategory.TEAMPLAY for s in suggestions))

    def test_combat_density(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 10.0),
                _event(EventType.PLAYER_KILL, 20.0),
                _event(EventType.PLAYER_DEATH, 30.0),
                _event(EventType.PLAYER_KILL, 40.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(any(s.category == CoachCategory.POSITIONING for s in suggestions))

    def test_suggestions_have_evidence(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_DEATH, 15.0, frame=150),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        self.assertTrue(len(suggestions) > 0)
        self.assertTrue(len(suggestions[0].evidence) > 0)

    def test_suggestions_have_unique_ids(self) -> None:
        ra1 = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0), _event(EventType.ROUND_END, 60.0)],
        )
        ra2 = RoundAnalysis(
            round_id="round_002", start_sec=70.0, end_sec=130.0,
            events=[_event(EventType.ROUND_START, 70.0), _event(EventType.PLAYER_KILL, 90.0), _event(EventType.PLAYER_KILL, 100.0), _event(EventType.ROUND_END, 130.0)],
        )
        analysis = self._build_analysis([ra1, ra2])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        ids = [s.suggestion_id for s in suggestions]
        self.assertEqual(len(ids), len(set(ids)))  # all unique

    def test_confidence_in_range(self) -> None:
        ra = RoundAnalysis(
            round_id="round_001", start_sec=0.0, end_sec=60.0,
            events=[_event(EventType.ROUND_START, 0.0), _event(EventType.PLAYER_DEATH, 15.0), _event(EventType.PLAYER_KILL, 30.0), _event(EventType.ROUND_END, 60.0)],
        )
        analysis = self._build_analysis([ra])
        report = self.builder.build(analysis)
        suggestions = self.coach.generate(analysis, report)
        for s in suggestions:
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)
