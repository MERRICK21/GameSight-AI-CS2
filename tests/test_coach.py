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
from gamesight.i18n.loader import I18nLoader
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

    def test_contact_time_is_neutral_and_not_divided_by_short_round(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=50.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.ENEMY_FIRST_VISIBLE, 21.5),
            _event(EventType.ROUND_END, 50.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        contact = next(x for x in s if "contact_context_" in x.suggestion_id)
        combined = f"{contact.reasoning} {contact.action}".lower()
        self.assertIn("21.5", combined)
        self.assertNotIn("43%", combined)
        self.assertNotIn("passive", combined)
        self.assertNotIn("slow map control", combined)
        self.assertIn("map, side, spawn, and route evidence", combined)

    def test_early_contact_does_not_infer_unsupported_rush(self) -> None:
        ra = RoundAnalysis(round_id="r1", start_sec=0.0, end_sec=60.0, events=[
            _event(EventType.ROUND_START, 0.0), _event(EventType.ENEMY_FIRST_VISIBLE, 3.0),
            _event(EventType.ROUND_END, 60.0)])
        a = self._analysis([ra])
        s = self.coach.generate(a, self._report(a))
        contact = next(x for x in s if "contact_context_" in x.suggestion_id)
        combined = f"{contact.reasoning} {contact.action}".lower()
        self.assertNotIn("rush", combined)
        self.assertNotIn("without utility", combined)

    def test_first_person_engagement_gets_grounded_review_not_aim_verdict(self) -> None:
        event = _event(EventType.ENGAGEMENT_CANDIDATE, 35.0, frame=1050)
        ra = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0, events=[event],
        )
        analysis = self._analysis([ra])
        analysis.capabilities = {"personal_combat": False, "enemy_contact": True}
        suggestions = self.coach.generate(analysis, self._report(analysis))
        engagement = next(
            item for item in suggestions if "engagement_" in item.suggestion_id
        )
        self.assertEqual(engagement.timestamp_sec, 35.0)
        self.assertIn("review", engagement.action.lower())
        self.assertNotIn("unstable", engagement.reasoning.lower())

    def test_likely_firefight_names_visual_candidates_not_confirmed_hits(self) -> None:
        event = _event(EventType.ENGAGEMENT_CANDIDATE, 35.0, frame=1050)
        event.attributes.update({
            "engagement_level": "likely_firefight",
            "shot_candidate_count": 2,
            "damage_candidate_count": 1,
            "visible_sample_count": 3,
            "observed_span_sec": 1.0,
            "first_shot_offset_sec": 0.5,
        })
        ra = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0, events=[event],
        )
        analysis = self._analysis([ra])
        analysis.capabilities = {"personal_combat": False, "enemy_contact": True}
        suggestion = next(
            item for item in self.coach.generate(analysis, self._report(analysis))
            if "engagement_" in item.suggestion_id
        )
        self.assertIn("muzzle-flash candidate", suggestion.reasoning)
        self.assertNotIn("confirmed hit", suggestion.reasoning.lower())
        self.assertIn("movement had stopped", suggestion.action)

    def test_contact_only_advice_does_not_infer_a_shot(self) -> None:
        event = _event(EventType.ENGAGEMENT_CANDIDATE, 35.0, frame=1050)
        event.attributes.update({
            "engagement_level": "visual_contact",
            "visible_sample_count": 2,
            "observed_span_sec": 0.5,
        })
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0, events=[event],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {"personal_combat": False, "enemy_contact": True}
        suggestion = next(
            item for item in self.coach.generate(analysis, self._report(analysis))
            if "engagement_" in item.suggestion_id
        )
        self.assertIn("2 sampled frame", suggestion.reasoning)
        self.assertIn("no shot is inferred", suggestion.action)

    def test_native_kd_keeps_rich_first_person_analysis(self) -> None:
        first = _event(EventType.ENGAGEMENT_CANDIDATE, 25.0, frame=750)
        first.attributes.update({
            "engagement_level": "visual_contact",
            "visible_sample_count": 2,
            "observed_span_sec": 0.5,
        })
        second = _event(EventType.ENGAGEMENT_CANDIDATE, 40.0, frame=1200)
        second.attributes.update({
            "engagement_level": "likely_firefight",
            "shot_candidate_count": 1,
            "damage_candidate_count": 0,
            "visible_sample_count": 2,
            "observed_span_sec": 0.5,
            "first_shot_offset_sec": 0.5,
        })
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=0.0, end_sec=60.0,
            events=[
                first,
                second,
                _event(EventType.PLAYER_KILL, 41.0),
                _event(EventType.PLAYER_DEATH, 50.0),
            ],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {
            "personal_combat": True,
            "personal_kills": True,
            "personal_deaths": True,
            "enemy_contact": True,
        }
        suggestions = self.coach.generate(analysis, self._report(analysis))
        engagement_cards = [
            item for item in suggestions if "engagement_" in item.suggestion_id
        ]
        self.assertEqual(len(engagement_cards), 2)
        self.assertFalse(any(
            "contact_context_" in item.suggestion_id for item in suggestions
        ))

    def test_precontact_candidate_gets_prefire_review(self) -> None:
        event = _event(EventType.ENGAGEMENT_CANDIDATE, 35.0, frame=1050)
        event.attributes.update({
            "engagement_level": "likely_firefight",
            "shot_candidate_count": 1,
            "damage_candidate_count": 0,
            "first_shot_offset_sec": -0.5,
        })
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0, events=[event],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {"personal_combat": False, "enemy_contact": True}
        suggestion = next(
            item for item in self.coach.generate(analysis, self._report(analysis))
            if "engagement_" in item.suggestion_id
        )
        self.assertIn("precedes first enemy visibility", suggestion.action)

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

    def test_full_combat_summary_is_localized_in_chinese(self) -> None:
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=0.0, end_sec=60.0,
            events=[
                _event(EventType.ROUND_START, 0.0),
                _event(EventType.PLAYER_KILL, 20.0),
                _event(EventType.PLAYER_KILL, 40.0),
                _event(EventType.PLAYER_DEATH, 50.0),
                _event(EventType.ROUND_END, 60.0),
            ],
        )
        analysis = self._analysis([round_analysis])
        loader = I18nLoader("zh-CN")
        report = EvidenceReportBuilder(loader=loader).build(analysis)
        coach = RuleBasedCoach(loader)
        suggestions = coach.generate(analysis, report)
        summary = coach.summarize(suggestions, analysis, report)

        self.assertIn("共分析 1 个回合", summary.overall_assessment)
        self.assertIn("个人击杀", summary.strengths[0])
        rendered = " ".join([
            summary.overall_assessment,
            *summary.strengths,
            *summary.weaknesses,
            *summary.practice_drills,
            *summary.focus_areas,
        ])
        self.assertNotIn("Across", rendered)
        self.assertNotIn("Positioning needs work", rendered)

    def test_summarize_engagement_without_personal_combat(self) -> None:
        engagement = _event(
            EventType.ENGAGEMENT_CANDIDATE, 35.0, frame=1050,
        )
        engagement.attributes.update({
            "engagement_level": "likely_firefight",
            "shot_candidate_count": 1,
            "damage_candidate_count": 0,
        })
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0,
            events=[engagement],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {
            "personal_combat": False,
            "enemy_contact": True,
        }
        report = self._report(analysis)
        suggestions = self.coach.generate(analysis, report)
        summary = self.coach.summarize(suggestions, analysis, report)
        self.assertEqual(len(summary.strengths), 2)
        self.assertTrue(any("enemy-visible" in item for item in summary.strengths))

    def test_native_death_gets_clip_review_without_enabling_kd(self) -> None:
        death = _event(EventType.PLAYER_DEATH, 35.0, frame=1050)
        death.attributes.update({
            "method": "native_health_hud_disappearance",
            "hud_missing_duration_sec": 1.0,
        })
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0, events=[death],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {
            "personal_combat": False,
            "personal_kills": False,
            "personal_deaths": True,
        }
        report = self._report(analysis)
        suggestions = self.coach.generate(analysis, report)
        suggestion = next(
            item for item in suggestions
            if "native_death_" in item.suggestion_id
        )
        self.assertIn("native health HUD disappeared", suggestion.reasoning)
        self.assertIn("death clip", suggestion.action)
        summary = self.coach.summarize(suggestions, analysis, report)
        self.assertIn("1 POV death", summary.overall_assessment)

    def test_lower_bound_kills_are_acknowledged_without_exact_kd(self) -> None:
        round_analysis = RoundAnalysis(
            round_id="r1", start_sec=10.0, end_sec=70.0,
            events=[
                _event(EventType.PLAYER_KILL, 30.0),
                _event(EventType.PLAYER_DEATH, 55.0),
            ],
        )
        analysis = self._analysis([round_analysis])
        analysis.capabilities = {
            "personal_combat": False,
            "personal_kills": True,
            "personal_deaths": True,
        }

        report = self._report(analysis)
        summary = self.coach.summarize([], analysis, report)

        self.assertIn("at least 1 POV kill", summary.overall_assessment)
        self.assertIn("exact K/D", summary.weaknesses[0])

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
