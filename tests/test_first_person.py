import unittest
from pathlib import Path

import numpy as np

from gamesight.coach.engine import RuleBasedCoach
from gamesight.domain.models import (
    AnalysisResult, EventType, RoundAnalysis, VideoInput, VideoMetadata,
)
from gamesight.perception.first_person import (
    FirstPersonAnalyzer,
    FirstPersonSample,
    build_first_person_summary_events,
)
from gamesight.reporting.builder import EvidenceReportBuilder


class FirstPersonAnalyzerTests(unittest.TestCase):
    def test_detects_full_flash(self):
        image = np.full((180, 320, 3), 245, dtype=np.uint8)
        sample = FirstPersonAnalyzer().update(image, 1, 1.0)
        self.assertTrue(sample.flashed)

    def test_detects_scope_geometry(self):
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[65:115, 125:195] = 100
        sample = FirstPersonAnalyzer().update(image, 1, 1.0)
        self.assertTrue(sample.scoped)

    def test_motion_ignores_bottom_watermark_band(self):
        analyzer = FirstPersonAnalyzer()
        first = np.zeros((180, 320, 3), dtype=np.uint8)
        second = first.copy()
        second[165:, :] = 255
        analyzer.update(first, 0, 0.0)
        sample = analyzer.update(second, 30, 1.0)
        self.assertEqual(sample.motion_score, 0.0)

    def test_reads_team_only_from_native_bottom_centre_hud(self):
        terrorist = np.zeros((180, 320, 3), dtype=np.uint8)
        terrorist[155:179, 138:182] = (0, 220, 220)
        counter_terrorist = np.zeros((180, 320, 3), dtype=np.uint8)
        counter_terrorist[155:179, 138:182] = (220, 220, 0)
        self.assertEqual(
            FirstPersonAnalyzer().update(terrorist, 1, 1.0).player_team, "t",
        )
        self.assertEqual(
            FirstPersonAnalyzer().update(counter_terrorist, 1, 1.0).player_team,
            "ct",
        )


class FirstPersonSummaryTests(unittest.TestCase):
    def test_aggregates_round_metrics(self):
        rounds = [RoundAnalysis(round_id="r1", start_sec=0, end_sec=4)]
        samples = [
            FirstPersonSample(0, 0, False, False, None),
            FirstPersonSample(30, 1, True, False, .2),
            FirstPersonSample(60, 2, True, True, .1),
            FirstPersonSample(90, 3, False, True, .05),
        ]
        events = build_first_person_summary_events(rounds, samples)
        summary = next(
            event for event in events
            if event.event_type == EventType.FIRST_PERSON_SUMMARY
        )
        self.assertEqual(summary.attributes["flash_count"], 1)
        self.assertEqual(summary.attributes["flash_exposure_sec"], 2.0)
        self.assertEqual(summary.attributes["scoped_sec"], 2.0)
        moments = [
            event for event in events
            if event.event_type == EventType.FIRST_PERSON_MOMENT
        ]
        self.assertEqual(len(moments), 1)
        self.assertEqual(moments[0].attributes["moment_kind"], "flash")

    def test_report_and_coach_use_viewport_metrics_without_combat_identity(self):
        round_analysis = RoundAnalysis(round_id="r1", start_sec=0, end_sec=10)
        samples = [
            FirstPersonSample(0, 0, False, False, None),
            FirstPersonSample(30, 1, True, False, .5),
            FirstPersonSample(60, 2, True, True, .5),
            FirstPersonSample(90, 3, False, True, .5),
            FirstPersonSample(120, 4, False, True, .5),
            FirstPersonSample(150, 5, False, True, .5),
            FirstPersonSample(180, 6, False, True, .5),
            FirstPersonSample(210, 7, False, True, .5),
            FirstPersonSample(240, 8, False, True, .5),
            FirstPersonSample(270, 9, False, True, .5),
        ]
        round_analysis.events = build_first_person_summary_events(
            [round_analysis], samples,
        )
        analysis = AnalysisResult(
            video=VideoInput(video_id="v", path=Path("v.mp4")),
            metadata=VideoMetadata(duration_sec=10, fps=30, width=320, height=180),
            rounds=[round_analysis], capabilities={"personal_combat": False},
        )
        report = EvidenceReportBuilder().build(analysis)
        stats = report.rounds[0].stats
        self.assertEqual(stats.flash_count, 1)
        self.assertEqual(stats.flash_exposure_sec, 2.0)
        self.assertEqual(stats.scoped_sec, 8.0)
        self.assertTrue(any("describes running and turning" in item.text
                            for item in report.rounds[0].findings))

        suggestions = RuleBasedCoach().generate(analysis, report)
        suggestion_ids = {item.suggestion_id for item in suggestions}
        self.assertTrue(any("flash_" in item for item in suggestion_ids))
        self.assertTrue(any("scope_hold_" in item for item in suggestion_ids))
        self.assertFalse(any("view_motion_" in item for item in suggestion_ids))

    def test_summary_evidence_does_not_choose_opening_motion_peak(self):
        rounds = [RoundAnalysis(round_id="r1", start_sec=50, end_sec=90)]
        samples = [
            FirstPersonSample(1500 + i * 30, 50 + i, False, False,
                              .9 if i == 2 else .1)
            for i in range(40)
        ]
        events = build_first_person_summary_events(rounds, samples)
        summary = next(
            event for event in events
            if event.event_type == EventType.FIRST_PERSON_SUMMARY
        )
        self.assertGreaterEqual(summary.start_sec, 65.0)
        self.assertNotEqual(summary.start_sec, 52.0)


if __name__ == "__main__":
    unittest.main()
