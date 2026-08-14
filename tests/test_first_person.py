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
    _native_local_kill_row_fingerprints,
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

    def test_detects_transient_muzzle_flash_candidate(self):
        analyzer = FirstPersonAnalyzer()
        first = np.zeros((180, 320, 3), dtype=np.uint8)
        second = first.copy()
        # A short, bright warm-colour change inside the central weapon/view
        # area is a visual candidate only; it is not presented as a hit.
        second[60:125, 105:215] = (0, 180, 255)
        analyzer.update(first, 0, 0.0)
        sample = analyzer.update(second, 3, 0.1)
        self.assertTrue(sample.shot_candidate)
        self.assertGreaterEqual(sample.shot_signal_score, 0.04)

    def test_detects_two_sided_damage_overlay_candidate(self):
        analyzer = FirstPersonAnalyzer()
        first = np.zeros((180, 320, 3), dtype=np.uint8)
        second = first.copy()
        second[30:145, 12:58] = (0, 0, 255)
        second[30:145, 262:308] = (0, 0, 255)
        analyzer.update(first, 0, 0.0)
        sample = analyzer.update(second, 3, 0.1)
        self.assertTrue(sample.damage_candidate)
        self.assertGreaterEqual(sample.damage_signal_score, 0.025)

    def test_full_flash_is_not_a_combat_signal(self):
        analyzer = FirstPersonAnalyzer()
        analyzer.update(np.zeros((180, 320, 3), dtype=np.uint8), 0, 0.0)
        sample = analyzer.update(
            np.full((180, 320, 3), 245, dtype=np.uint8), 3, 0.1,
        )
        self.assertTrue(sample.flashed)
        self.assertFalse(sample.shot_candidate)
        self.assertFalse(sample.damage_candidate)

    def test_detects_native_health_hud_cluster(self):
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        # Pink native-HUD-like glyph blocks inside the narrow health cluster.
        image[164:177, 82:88] = (220, 80, 220)
        image[164:177, 92:99] = (220, 80, 220)
        image[164:177, 103:110] = (220, 80, 220)
        sample = FirstPersonAnalyzer().update(image, 1, 1.0)
        self.assertTrue(sample.health_hud_visible)
        self.assertGreater(sample.health_hud_score, 0.025)

    def test_blank_health_hud_region_is_not_visible(self):
        sample = FirstPersonAnalyzer().update(
            np.zeros((180, 320, 3), dtype=np.uint8), 1, 1.0,
        )
        self.assertFalse(sample.health_hud_visible)

    def test_detects_native_local_kill_highlight_without_reading_text(self):
        image = np.full((180, 320, 3), 150, dtype=np.uint8)
        # Native top-right dark feed row with the local-player red outline.
        image[13:32, 254:319] = 30
        image[13:15, 254:319] = (0, 0, 255)
        image[30:32, 254:319] = (0, 0, 255)
        image[13:32, 254:256] = (0, 0, 255)
        image[13:32, 317:319] = (0, 0, 255)
        image[19:27, 280:292] = 240

        sample = FirstPersonAnalyzer().update(image, 1, 1.0)

        self.assertTrue(sample.local_kill_highlight)
        self.assertGreaterEqual(sample.local_kill_highlight_score, .45)

    def test_plain_kill_feed_row_is_not_a_local_kill(self):
        image = np.full((180, 320, 3), 150, dtype=np.uint8)
        image[13:32, 254:319] = 30

        sample = FirstPersonAnalyzer().update(image, 1, 1.0)

        self.assertFalse(sample.local_kill_highlight)

    def test_warm_texture_without_feed_glyphs_is_not_a_local_kill(self):
        image = np.full((180, 320, 3), (20, 70, 140), dtype=np.uint8)
        image[13:25, 254:319] = (0, 0, 145)

        sample = FirstPersonAnalyzer().update(image, 1, 1.0)

        self.assertFalse(sample.local_kill_highlight)

    def test_complete_native_row_produces_a_content_fingerprint(self):
        image = np.full((360, 640, 3), 150, dtype=np.uint8)
        image[40:58, 470:638] = 30
        image[40:42, 470:638] = (0, 0, 120)
        image[56:58, 470:638] = (0, 0, 120)
        image[40:58, 470:472] = (0, 0, 120)
        image[40:58, 636:638] = (0, 0, 120)
        image[46:53, 570:610] = 230

        self.assertTrue(_native_local_kill_row_fingerprints(image))


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
