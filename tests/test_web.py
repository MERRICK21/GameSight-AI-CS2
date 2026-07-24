"""Tests for the Streamlit web application — demo data and pipeline integration."""

from __future__ import annotations

from unittest import TestCase

from gamesight.domain.models import (
    AnalysisResult,
    Detection,
    EventType,
    GameEvent,
    Track,
    VideoInput,
    VideoMetadata,
)
from gamesight.events.aggregator import aggregate_events
from gamesight.reporting.builder import EvidenceReportBuilder
from gamesight.serialization.timeline import TimelineBuilder
from gamesight.web.demo import generate_demo_events, generate_demo_tracks, run_demo_pipeline


class DemoEventGenerationTests(TestCase):
    """Tests for synthetic demo data generators."""

    def test_generate_demo_events_returns_list(self) -> None:
        events = generate_demo_events(rounds=3)
        self.assertIsInstance(events, list)
        self.assertTrue(len(events) > 0)

    def test_generate_demo_events_all_are_game_events(self) -> None:
        events = generate_demo_events(rounds=3)
        for e in events:
            self.assertIsInstance(e, GameEvent)

    def test_generate_demo_events_default_5_rounds(self) -> None:
        events = generate_demo_events()
        round_starts = [e for e in events if e.event_type == EventType.ROUND_START]
        self.assertEqual(len(round_starts), 5)

    def test_generate_demo_events_custom_rounds(self) -> None:
        events = generate_demo_events(rounds=2)
        round_starts = [e for e in events if e.event_type == EventType.ROUND_START]
        self.assertEqual(len(round_starts), 2)

    def test_generate_demo_events_rounds_paired(self) -> None:
        events = generate_demo_events(rounds=3)
        starts = [e for e in events if e.event_type == EventType.ROUND_START]
        ends = [e for e in events if e.event_type == EventType.ROUND_END]
        self.assertEqual(len(starts), len(ends))

    def test_generate_demo_events_chronological(self) -> None:
        events = generate_demo_events(rounds=3)
        for i in range(len(events) - 1):
            self.assertLessEqual(events[i].start_sec, events[i + 1].start_sec)

    def test_generate_demo_events_contains_kills(self) -> None:
        events = generate_demo_events(rounds=5)
        kills = [e for e in events if e.event_type == EventType.PLAYER_KILL]
        self.assertTrue(len(kills) > 0)

    def test_generate_demo_events_contains_enemy_visible(self) -> None:
        events = generate_demo_events(rounds=3)
        efv = [e for e in events if e.event_type == EventType.ENEMY_FIRST_VISIBLE]
        self.assertTrue(len(efv) > 0)

    def test_generate_demo_events_each_event_has_evidence(self) -> None:
        events = generate_demo_events(rounds=2)
        for e in events:
            self.assertTrue(len(e.evidence) > 0, f"Event {e.event_id} has no evidence")

    def test_generate_demo_events_round_ids_consistent(self) -> None:
        events = generate_demo_events(rounds=3)
        for e in events:
            rid = e.attributes.get("round_id")
            if rid is not None:
                self.assertIn("round_", str(rid))


class DemoTrackGenerationTests(TestCase):
    """Tests for synthetic demo track data."""

    def test_generate_demo_tracks_returns_list(self) -> None:
        tracks = generate_demo_tracks()
        self.assertIsInstance(tracks, list)
        self.assertTrue(len(tracks) > 0)

    def test_generate_demo_tracks_all_are_tracks(self) -> None:
        tracks = generate_demo_tracks()
        for t in tracks:
            self.assertIsInstance(t, Track)

    def test_generate_demo_tracks_have_detections(self) -> None:
        tracks = generate_demo_tracks()
        for t in tracks:
            self.assertTrue(len(t.detections) > 0)
            self.assertIsInstance(t.detections[0], Detection)

    def test_generate_demo_tracks_contains_enemy_and_teammate(self) -> None:
        tracks = generate_demo_tracks()
        labels = {t.label for t in tracks}
        self.assertIn("enemy", labels)
        self.assertIn("teammate", labels)

    def test_generate_demo_tracks_detections_chronological(self) -> None:
        tracks = generate_demo_tracks()
        for t in tracks:
            for i in range(len(t.detections) - 1):
                self.assertLessEqual(
                    t.detections[i].timestamp_sec,
                    t.detections[i + 1].timestamp_sec,
                )


class PipelineIntegrationTests(TestCase):
    """End-to-end pipeline integration using demo data."""

    def setUp(self) -> None:
        self.demo_events = generate_demo_events(rounds=3)
        self.demo_tracks = generate_demo_tracks()

    def test_aggregate_demo_events(self) -> None:
        rounds = aggregate_events(self.demo_events)
        self.assertEqual(len(rounds), 3)
        for r in rounds:
            self.assertTrue(r.start_sec >= 0)
            self.assertTrue(len(r.events) > 0)

    def test_build_timeline_from_demo_data(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        timeline = TimelineBuilder().build(analysis, self.demo_tracks)
        self.assertEqual(timeline.total_rounds, 3)
        self.assertEqual(timeline.video_id, "demo_test")

    def test_build_report_from_demo_data(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        report = EvidenceReportBuilder().build(analysis, self.demo_tracks)
        self.assertEqual(report.overview.total_rounds, 3)
        self.assertEqual(len(report.rounds), 3)
        self.assertTrue(report.overview.total_kills_detected > 0)

    def test_demo_report_has_match_findings(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        report = EvidenceReportBuilder().build(analysis, self.demo_tracks)
        self.assertTrue(len(report.match_findings) > 0)

    def test_demo_report_rounds_have_findings(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        report = EvidenceReportBuilder().build(analysis, self.demo_tracks)
        for rr in report.rounds:
            self.assertTrue(len(rr.findings) > 0, f"Round {rr.round_id} has no findings")

    def test_demo_timeline_json_roundtrip(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        timeline = TimelineBuilder().build(analysis, self.demo_tracks)
        d = timeline.model_dump(mode="json")
        self.assertIsInstance(d, dict)
        self.assertEqual(d["total_rounds"], 3)

    def test_demo_report_json_roundtrip(self) -> None:
        rounds = aggregate_events(self.demo_events)
        analysis = AnalysisResult(
            video=VideoInput(video_id="demo_test", path="demo.mp4"),
            metadata=VideoMetadata(),
            rounds=rounds,
        )
        report = EvidenceReportBuilder().build(analysis, self.demo_tracks)
        d = report.model_dump(mode="json")
        self.assertIsInstance(d, dict)
        self.assertEqual(d["overview"]["total_rounds"], 3)


class RunDemoPipelineTests(TestCase):
    """Tests for run_demo_pipeline (the main pipeline runner)."""

    def test_returns_analysis_and_tracks(self) -> None:
        analysis, tracks = run_demo_pipeline("test.mp4", sample_fps=10.0)
        self.assertIsInstance(analysis, AnalysisResult)
        self.assertIsInstance(tracks, list)
        self.assertTrue(len(tracks) > 0)

    def test_rounds_populated(self) -> None:
        analysis, _ = run_demo_pipeline("test.mp4", sample_fps=10.0)
        self.assertTrue(len(analysis.rounds) > 0)

    def test_video_id_set(self) -> None:
        analysis, _ = run_demo_pipeline("my_cs2_clip.mp4", sample_fps=10.0)
        self.assertEqual(analysis.video.video_id, "my_cs2_clip")

    def test_metadata_set(self) -> None:
        analysis, _ = run_demo_pipeline("test.mp4", sample_fps=10.0)
        self.assertIsNotNone(analysis.metadata.duration_sec)
        self.assertIsNotNone(analysis.metadata.fps)
        self.assertEqual(analysis.metadata.width, 1920)
        self.assertEqual(analysis.metadata.height, 1080)

    def test_full_integration(self) -> None:
        """Full round-trip: run pipeline → build timeline → build report → serialise."""
        analysis, tracks = run_demo_pipeline("integration_test.mp4", sample_fps=10.0)

        timeline = TimelineBuilder().build(analysis, tracks)
        self.assertIsNotNone(timeline)
        tl_json = timeline.model_dump(mode="json")
        self.assertIsInstance(tl_json, dict)

        report = EvidenceReportBuilder().build(analysis, tracks)
        self.assertIsNotNone(report)
        rpt_json = report.model_dump(mode="json")
        self.assertIsInstance(rpt_json, dict)
